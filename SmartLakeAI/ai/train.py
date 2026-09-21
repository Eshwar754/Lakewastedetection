"""
SmartLake AI - YOLO Instance Segmentation Training Pipeline
-----------------------------------------------------------
Loads dataset, runs transfer learning from YOLOv8-seg, trains model,
saves checkpoints, evaluates validation performance (mAP50, mAP50-95, precision, recall),
and exports final best.pt model weights.
"""

import os
import sys
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO


def train_model(config_path, dataset_yaml_path):
    """
    Trains the YOLO instance segmentation model using transfer learning.
    """
    print(f"==================================================")
    print(f"  SmartLake AI: Model Training Pipeline")
    print(f"==================================================")

    # 1. Load system config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Hardware / Device detection
    requested_device = config['model'].get('device', 'auto')
    if requested_device == 'auto':
        device = '0' if torch.cuda.is_available() else 'cpu'
    else:
        device = requested_device

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (Using CPU)"
    print(f"  - PyTorch Version: {torch.__version__}")
    print(f"  - CUDA Available: {torch.cuda.is_available()}")
    print(f"  - Selected Device: {device} ({gpu_name})")

    # 3. Model setup (Pretrained YOLO segmentation model)
    pretrained_weights = config['model'].get('pretrained_weights', 'yolov8n-seg.pt')
    print(f"  - Loading pretrained model: {pretrained_weights}")
    model = YOLO(pretrained_weights)

    # 4. Training configuration parameters
    epochs = config['training'].get('epochs', 30)
    batch_size = config['training'].get('batch_size', 16)
    img_size = config['model'].get('img_size', 640)
    lr = config['training'].get('learning_rate', 0.01)
    patience = config['training'].get('patience', 10)

    project_dir = Path(__file__).resolve().parent.parent / 'runs'
    name_run = 'smartlake_seg_run'

    print(f"\n  Starting Training with parameters:")
    print(f"  - Dataset YAML: {dataset_yaml_path}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Image Size: {img_size}")
    print(f"  - Learning Rate: {lr}")
    print(f"  - Patience: {patience}")
    print(f"==================================================\n")

    # 5. Run Ultralytics YOLO Training
    results = model.train(
        data=str(dataset_yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        lr0=lr,
        patience=patience,
        device=device,
        project=str(project_dir),
        name=name_run,
        exist_ok=True,
        save=True,
        plots=True
    )

    # 6. Copy best weights to models/best.pt
    best_weights_source = project_dir / name_run / 'weights' / 'best.pt'
    target_weights_dir = Path(__file__).resolve().parent.parent / 'models'
    target_weights_dir.mkdir(parents=True, exist_ok=True)
    target_weights_path = target_weights_dir / 'best.pt'

    if best_weights_source.exists():
        import shutil
        shutil.copy(best_weights_source, target_weights_path)
        print(f"\n[+] Training complete! Best weights saved to: {target_weights_path}")
    else:
        print(f"\n[!] Best weights not found at: {best_weights_source}")

    return results


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / 'config.yaml'
    dataset_yaml_path = base_dir / 'dataset' / 'dataset.yaml'

    if not dataset_yaml_path.exists():
        print(f"[!] dataset.yaml not found. Running dataset_check.py first...")
        from dataset_check import check_dataset_integrity, generate_dataset_yaml, load_class_mapping
        trash_dataset = Path(r"c:\Users\smesh\Downloads\trash_inst_material\trash_inst_material")
        mapping_path = base_dir / 'dataset' / 'class_mapping.json'
        if mapping_path.exists():
            mapping_config = load_class_mapping(mapping_path)
            generate_dataset_yaml(trash_dataset, dataset_yaml_path, mapping_config)

    train_model(config_path, dataset_yaml_path)
