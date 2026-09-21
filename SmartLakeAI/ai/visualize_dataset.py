"""
SmartLake AI - Dataset Visualization Tool
------------------------------------------
Draws YOLO instance segmentation polygon masks and bounding boxes onto sample images
to visually confirm annotation accuracy before training.
"""

import os
import random
import cv2
import json
import numpy as np
from pathlib import Path


def draw_polygons_on_image(img_path, lbl_path, mapping_config, output_path):
    """Draws segmentation polygon overlays on image."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[!] Could not load image: {img_path}")
        return

    h, w, _ = img.shape
    orig_names = mapping_config['original_classes']
    mapping = mapping_config['mapping']

    if not lbl_path.exists():
        cv2.imwrite(str(output_path), img)
        return

    overlay = img.copy()
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        (0, 128, 255), (128, 0, 255), (255, 0, 128), (0, 255, 128)
    ]

    with open(lbl_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if not parts:
            continue

        cls_id = parts[0]
        mapped_label = mapping.get(cls_id, orig_names.get(cls_id, f"Class_{cls_id}"))

        # Convert normalized coordinates to pixel coordinates
        coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
        pixel_coords = (coords * np.array([w, h])).astype(np.int32)

        color = colors[int(cls_id) % len(colors)]

        # Draw filled polygon overlay
        cv2.fillPoly(overlay, [pixel_coords], color)

        # Draw polygon contour border
        cv2.polylines(img, [pixel_coords], isClosed=True, color=color, thickness=2)

        # Bounding box coordinates for label placement
        x_min, y_min = np.min(pixel_coords, axis=0)
        
        # Label text
        label_text = f"{mapped_label}"
        cv2.putText(img, label_text, (x_min, max(y_min - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Alpha blend overlay for semi-transparent masks
    alpha = 0.35
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"[+] Saved visualization: {output_path}")


def visualize_samples(dataset_dir, mapping_config, num_samples=5):
    """Selects random sample images and visualizes annotations."""
    dataset_path = Path(dataset_dir)
    img_dir = dataset_path / 'train' / 'images'
    lbl_dir = dataset_path / 'train' / 'labels'
    output_dir = dataset_path.parent / 'dataset' / 'sample_visualizations'

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

    trash_dataset = Path(r"c:\Users\smesh\Downloads\trash_inst_material\trash_inst_material")
    if not trash_dataset.exists():
        trash_dataset = base_dir / 'dataset'

    if mapping_path.exists():
        with open(mapping_path, 'r') as f:
            mapping_config = json.load(f)
        visualize_samples(trash_dataset, mapping_config, num_samples=5)
    else:
        print(f"[ERROR] Class mapping file not found at: {mapping_path}")
