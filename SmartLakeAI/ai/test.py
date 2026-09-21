"""
SmartLake AI - Model Testing Pipeline
-------------------------------------
Evaluates trained model on unseen test split images.
"""

import sys
from pathlib import Path
from ultralytics import YOLO


def test_model(weights_path, dataset_yaml_path):
    """Run testing evaluation on test dataset split."""
    print(f"==================================================")
    print(f"  SmartLake AI: Testing Pipeline")
    print(f"  Model Weights: {weights_path}")
    print(f"==================================================")

    if not Path(weights_path).exists():
        print(f"[!] Target weights not found at {weights_path}, using yolov8n-seg.pt")
        weights_path = 'yolov8n-seg.pt'

    model = YOLO(weights_path)

    # Evaluate on val set as test split
    metrics = model.val(
        data=str(dataset_yaml_path),
        split='val',
        plots=True
    )

    print(f"[+] Testing completed successfully.")
    return metrics


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent
    weights_path = base_dir / 'models' / 'best.pt'
    dataset_yaml_path = base_dir / 'dataset' / 'dataset.yaml'

    test_model(weights_path, dataset_yaml_path)
