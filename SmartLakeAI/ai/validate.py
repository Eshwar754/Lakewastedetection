"""
SmartLake AI - Model Validation Pipeline
----------------------------------------
Evaluates the trained segmentation model performance on validation dataset,
calculating Precision, Recall, mAP50, mAP50-95, and generating confusion matrix plots.
"""

import sys
import yaml
from pathlib import Path
from ultralytics import YOLO


def validate_model(weights_path, dataset_yaml_path, img_size=640, conf=0.25):
    """Run validation evaluation on trained model."""
    print(f"==================================================")
    print(f"  SmartLake AI: Model Validation")
    print(f"  Model Weights: {weights_path}")
    print(f"  Dataset YAML: {dataset_yaml_path}")
    print(f"==================================================")

    if not Path(weights_path).exists():
        print(f"[!] Warning: Custom weights file {weights_path} not found.")
        print(f"    Falling back to default pretrained 'yolov8n-seg.pt'...")
        weights_path = 'yolov8n-seg.pt'

    model = YOLO(weights_path)
    metrics = model.val(
        data=str(dataset_yaml_path),
        imgsz=img_size,
        conf=conf,
        split='val',
        plots=True
    )

    print(f"\n==================================================")
    print(f"  Validation Performance Metrics:")
    print(f"  - Precision (Bbox): {metrics.box.mp:.4f}")
    print(f"  - Recall (Bbox):    {metrics.box.mr:.4f}")
    print(f"  - mAP50 (Bbox):     {metrics.box.map50:.4f}")
    print(f"  - mAP50-95 (Bbox):  {metrics.box.map:.4f}")
    
    if hasattr(metrics, 'seg') and metrics.seg is not None:
        print(f"\n  Segmentation Mask Metrics:")
        print(f"  - Precision (Mask): {metrics.seg.mp:.4f}")
        print(f"  - Recall (Mask):    {metrics.seg.mr:.4f}")
        print(f"  - mAP50 (Mask):     {metrics.seg.map50:.4f}")
        print(f"  - mAP50-95 (Mask):  {metrics.seg.map:.4f}")
        
    print(f"==================================================\n")

    return metrics


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent
    weights_path = base_dir / 'models' / 'best.pt'
    dataset_yaml_path = base_dir / 'dataset' / 'dataset.yaml'

    validate_model(weights_path, dataset_yaml_path)
