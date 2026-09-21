import shutil
import torch
import yaml
from pathlib import Path
from ultralytics import YOLO

def quick_train_lake_model():
    base_dir = Path(__file__).resolve().parent.parent
    dataset_yaml = base_dir / 'dataset' / 'dataset.yaml'

    print("[+] Initializing SmartLake AI fine-tuning for target waste classes...")
    model = YOLO('yolov8n-seg.pt')

    # Train 1 fast epoch with optimized settings and fraction=0.04 for rapid checkpoint generation
    results = model.train(
        data=str(dataset_yaml),
        epochs=1,
        batch=16,
        imgsz=320,
        fraction=0.04,
        device='cpu',
        project=str(base_dir / 'runs'),
        name='quick_lake_run',
        exist_ok=True,
        save=True,
        plots=False,
        verbose=True
    )


    best_weights_source = base_dir / 'runs' / 'quick_lake_run' / 'weights' / 'best.pt'
    target_dir = base_dir / 'models'
    target_dir.mkdir(parents=True, exist_ok=True)
    target_weights = target_dir / 'best.pt'

    if best_weights_source.exists():
        shutil.copy(best_weights_source, target_weights)
        print(f"[+] Successfully generated custom model checkpoint: {target_weights}")
    else:
        # Fallback save active model
        model.save(str(target_weights))
        print(f"[+] Saved model checkpoint: {target_weights}")

if __name__ == '__main__':
    quick_train_lake_model()
