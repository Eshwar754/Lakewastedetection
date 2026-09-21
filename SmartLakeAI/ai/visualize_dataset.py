"""
SmartLake AI - Dataset Visualization Tool
------------------------------------------
Draws YOLO instance segmentation polygon masks and bounding boxes onto sample images
to visually confirm annotation accuracy before training.
"""

import os
import sys
import random
import cv2
import json
import numpy as np
from pathlib import Path

# Add project root to sys.path
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))



def draw_polygons_on_image(img_path, lbl_path, mapping_config, output_path):
    """Draws segmentation polygon overlays on image with bold image name header."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[!] Could not load image: {img_path}")
        return

    h, w, _ = img.shape
    orig_names = mapping_config['original_classes']
    mapping = mapping_config['mapping']
    target_classes = mapping_config.get('target_classes', [])

    if not lbl_path.exists():
        cv2.imwrite(str(output_path), img)
        return

    overlay = img.copy()
    colors = [
        (255, 144, 30), (50, 205, 50), (255, 69, 0), (0, 215, 255),
        (218, 112, 214), (255, 215, 0), (147, 112, 219), (0, 250, 154),
        (255, 105, 180), (30, 144, 255), (127, 255, 0), (255, 160, 122)
    ]

    with open(lbl_path, 'r') as f:
        lines = f.readlines()

    object_count = 0
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if not parts:
            continue
        object_count += 1

        cls_id_str = parts[0]
        if cls_id_str.isdigit() and int(cls_id_str) < len(target_classes):
            mapped_label = target_classes[int(cls_id_str)]
        else:
            mapped_label = mapping.get(cls_id_str, orig_names.get(cls_id_str, f"Class_{cls_id_str}"))

        # Convert normalized coordinates to pixel coordinates
        coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
        pixel_coords = (coords * np.array([w, h])).astype(np.int32)

        color = colors[int(cls_id_str) % len(colors)]

        # Draw filled polygon overlay
        cv2.fillPoly(overlay, [pixel_coords], color)

        # Draw polygon contour border with bold thickness
        cv2.polylines(img, [pixel_coords], isClosed=True, color=color, thickness=3)

        # Bounding box coordinates for label placement
        x_min, y_min = np.min(pixel_coords, axis=0)
        
        # Label text with black outline stroke for bold clarity
        label_text = f"{mapped_label}"
        tx, ty = x_min, max(y_min - 5, 20)
        cv2.putText(img, label_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, label_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Alpha blend overlay for semi-transparent masks
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Add Bold Top Header Banner marking Image Name
    header_h = max(46, int(h * 0.075))
    header_overlay = img.copy()
    cv2.rectangle(header_overlay, (0, 0), (w, header_h), (15, 23, 42), -1)
    cv2.addWeighted(header_overlay, 0.85, img, 0.15, 0, img)
    cv2.line(img, (0, header_h), (w, header_h), (0, 242, 254), 3)

    font_scale = max(0.55, min(0.85, w / 850.0))
    y_pos = int(header_h * 0.68)
    title_text = f"IMAGE NAME: {img_path.name}"
    cv2.putText(img, title_text, (16, y_pos + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, title_text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"[+] Saved visualization: {output_path}")


def visualize_samples(dataset_dir, mapping_config, num_samples=5):
    """Selects random sample images and visualizes annotations."""
    dataset_path = Path(dataset_dir)
    img_dir = dataset_path / 'train' / 'images'
    lbl_dir = dataset_path / 'train' / 'labels'
    output_dir = dataset_path.parent.parent / 'dataset' / 'sample_visualizations'

    images = list(img_dir.glob('*.*'))
    if not images:
        print(f"[!] No images found in: {img_dir}")
        return

    sample_images = random.sample(images, min(num_samples, len(images)))

    for img_path in sample_images:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        out_path = output_dir / f"vis_{img_path.name}"
        draw_polygons_on_image(img_path, lbl_path, mapping_config, out_path)

    print(f"[+] Successfully generated {len(sample_images)} sample visualizations in: {output_dir}")


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent
    mapping_path = base_dir / 'dataset' / 'class_mapping.json'

    from ai.dataset_check import find_dataset_dir
    trash_dataset = find_dataset_dir(base_dir)

    if mapping_path.exists():
        with open(mapping_path, 'r') as f:
            mapping_config = json.load(f)
        visualize_samples(trash_dataset, mapping_config, num_samples=5)
    else:
        print(f"[ERROR] Class mapping file not found at: {mapping_path}")

