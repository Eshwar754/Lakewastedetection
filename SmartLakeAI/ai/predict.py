"""
SmartLake AI - Single Image Inference Engine
---------------------------------------------
Performs YOLO instance segmentation on input images, displaying bounding boxes,
masks, confidence percentages, class names, and returning detailed waste statistics.
"""

import cv2
import yaml
import json
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from ultralytics import YOLO

# Import local statistics module
from ai.statistics import calculate_waste_statistics


class SmartLakePredictor:
    def __init__(self, weights_path=None, config_path=None):
        base_dir = Path(__file__).resolve().parent.parent

        if config_path is None:
            config_path = base_dir / 'config.yaml'

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        if weights_path is None:
            custom_weights = base_dir / self.config['model'].get('custom_weights', 'models/best.pt')
            if custom_weights.exists():
                weights_path = str(custom_weights)
            else:
                raise FileNotFoundError(
                    f"Trained lake model not found at {custom_weights}. "
                    "Run ai/train.py first or set model.custom_weights to a trained checkpoint."
                )

        self.conf_threshold = self.config['model'].get('conf_threshold', 0.25)
        self.iou_threshold = self.config['model'].get('iou_threshold', 0.45)
        self.img_size = self.config['model'].get('img_size', 640)

        requested_device = self.config['model'].get('device', 'auto')
        if requested_device == 'auto':
            self.device = '0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = requested_device

        print(f"[+] Loading SmartLake AI Model from: {weights_path} (Device: {self.device})")
        self.model = YOLO(weights_path)
        self.class_names = self.model.names

        dataset_config_path = base_dir / 'dataset' / 'dataset.yaml'
        with open(dataset_config_path, 'r') as f:
            dataset_config = yaml.safe_load(f)

        expected_names = dataset_config.get('names', {})
        expected_names = [expected_names[key] for key in sorted(expected_names, key=int)]
        loaded_names = [self.class_names[key] for key in sorted(self.class_names, key=int)]
        if loaded_names != expected_names:
            raise ValueError(
                "The loaded checkpoint does not match the lake dataset classes. "
                f"Expected {expected_names}, got {loaded_names}."
            )

    def predict_image(self, image_input, save_path=None):
        """
        Runs segmentation on image input (file path string or numpy array image).
        Returns annotated numpy image and detection metadata dict.
        """
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Could not load image file: {image_input}")
        else:
            img = image_input.copy()

        h, w, _ = img.shape

        # Run inference
        results = self.model.predict(
            source=img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            verbose=False
        )[0]

        detections = []
        annotated_img = img.copy()
        mask_overlay = np.zeros_like(img, dtype=np.uint8)

        colors = [
            (0, 165, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
            (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 255, 128)
        ]

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            masks = None
            if results.masks is not None:
                masks = results.masks.xy

            for i in range(len(boxes)):
                box = boxes[i].tolist()
                conf = float(confidences[i])
                cls_id = class_ids[i]
                cls_name = self.class_names.get(cls_id, f"Class_{cls_id}")

                color = colors[cls_id % len(colors)]

                # Format coordinates [x1, y1, x2, y2]
                x1, y1, x2, y2 = [int(v) for v in box]

                mask_coords = []
                if masks is not None and i < len(masks):
                    polygon_coords = masks[i]
                    if len(polygon_coords) > 0:
                        pts = np.int32([polygon_coords])
                        cv2.fillPoly(mask_overlay, pts, color)
                        cv2.polylines(annotated_img, pts, isClosed=True, color=color, thickness=2)
                        mask_coords = polygon_coords.tolist()

                # Draw bounding box
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

                # Label background & text
                label_str = f"{cls_name} {int(conf * 100)}%"
                (lbl_w, lbl_h), baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_img, (x1, max(y1 - lbl_h - 10, 0)), (x1 + lbl_w + 6, max(y1, lbl_h + 10)), color, -1)
                cv2.putText(annotated_img, label_str, (x1 + 3, max(y1 - 5, lbl_h + 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                detections.append({
                    "class": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2],
                    "mask_points": mask_coords
                })

        # Apply semi-transparent mask layer
        cv2.addWeighted(mask_overlay, 0.4, annotated_img, 0.6, 0, annotated_img)

        # Calculate statistics
        stats = calculate_waste_statistics(detections)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), annotated_img)
            print(f"[+] Saved annotated prediction to: {save_path}")

        return annotated_img, {
            "detections": detections,
            "statistics": stats
        }


if __name__ == '__main__':
    predictor = SmartLakePredictor()
    base_dir = Path(__file__).resolve().parent.parent

    # Try sample image first
    sample_img = base_dir / 'samples' / 'sample_lake_trash.jpg'
    if not sample_img.exists():
        val_dir = Path(r"c:\Users\smesh\Downloads\trash_inst_material\trash_inst_material\val\images")
        images = list(val_dir.glob('*.*')) if val_dir.exists() else []
        if images:
            sample_img = images[0]

    if sample_img.exists():
        out_path = base_dir / 'prediction_sample.jpg'
        out_img, meta = predictor.predict_image(sample_img, save_path=out_path)
        print(f"\n==================================================")
        print(f"  SmartLake AI: Prediction Result for {sample_img.name}")
        print(f"==================================================")
        print(json.dumps(meta['statistics'], indent=2))
        print(f"  Saved annotated prediction output to: {out_path}\n")
    else:
        print(f"[!] No sample image found to test prediction.")
