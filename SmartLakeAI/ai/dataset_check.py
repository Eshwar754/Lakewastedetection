"""
SmartLake AI - Dataset Validation and Integrity Checker
------------------------------------------------------
This module validates image/label pairs, checks for corruption or missing files,
counts original and mapped target class instances, and generates YOLO dataset YAML configurations.
"""

import os
import json
import yaml
from pathlib import Path
from PIL import Image
from collections import defaultdict


def load_class_mapping(mapping_path):
    """Load class mapping JSON configuration."""
    with open(mapping_path, 'r') as f:
        return json.load(f)


def check_dataset_integrity(dataset_dir, mapping_config):
    """
    Scans dataset_dir (with train/ and val/ subdirectories) for:
    - Image and label pairings
    - Corrupted image files
    - Unmatched image or label files
    - Class counts (original and mapped)
    """
    print(f"==================================================")
    print(f"  SmartLake AI: Checking Dataset Integrity")
    print(f"  Target Dir: {dataset_dir}")
    print(f"==================================================")

    dataset_path = Path(dataset_dir)
    splits = ['train', 'val']
    
    total_images = 0
    total_labels = 0
    corrupted_images = []
    missing_labels = []
    missing_images = []
    original_class_counts = defaultdict(int)
    mapped_class_counts = defaultdict(int)

    orig_names = mapping_config['original_classes']
    mapping = mapping_config['mapping']

    for split in splits:
        img_dir = dataset_path / split / 'images'
        lbl_dir = dataset_path / split / 'labels'

        if not img_dir.exists():
            print(f"[!] Warning: Image directory does not exist: {img_dir}")
            continue

        images = {f.stem: f for f in img_dir.glob('*.*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']}
        labels = {f.stem: f for f in lbl_dir.glob('*.txt')} if lbl_dir.exists() else {}

        print(f"\n--- Split: {split.upper()} ---")
        print(f"Found {len(images)} images and {len(labels)} label files.")

        total_images += len(images)
        total_labels += len(labels)

        # Check images for corruption and paired labels
        for stem, img_path in images.items():
            # 1. Image integrity check
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                print(f"[ERROR] Corrupted image: {img_path} ({e})")
                corrupted_images.append(str(img_path))
                continue

            # 2. Label pairing check
            if stem not in labels:
                missing_labels.append(str(img_path))
            else:
                lbl_path = labels[stem]
                with open(lbl_path, 'r') as lf:
                    lines = lf.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        cls_id = parts[0]
                        orig_name = orig_names.get(cls_id, f"unknown_{cls_id}")
                        mapped_name = mapping.get(cls_id, "Other_Waste")

                        original_class_counts[orig_name] += 1
                        mapped_class_counts[mapped_name] += 1

        # Check for orphan labels
        for stem, lbl_path in labels.items():
            if stem not in images:
                missing_images.append(str(lbl_path))

    print(f"\n==================================================")
    print(f"  Integrity Summary:")
    print(f"  - Total Images Scanned: {total_images}")
    print(f"  - Total Labels Scanned: {total_labels}")
    print(f"  - Corrupted Images: {len(corrupted_images)}")
    print(f"  - Images Missing Labels: {len(missing_labels)}")
    print(f"  - Labels Missing Images: {len(missing_images)}")

    print(f"\n  Original Class Instance Counts:")
    for cls_name, count in sorted(original_class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {cls_name}: {count}")

    print(f"\n  Mapped Target Class Instance Counts:")
    for cls_name, count in sorted(mapped_class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {cls_name}: {count}")

    unrep = mapping_config.get('unrepresented_classes', [])
    if unrep:
        print(f"\n  [NOTE] Unrepresented Classes (Require Additional Data):")
        for u in unrep:
            print(f"    - {u}: 0 instances in current dataset")

    print(f"==================================================\n")

    return {
        "total_images": total_images,
        "total_labels": total_labels,
        "corrupted": len(corrupted_images),
        "original_counts": dict(original_class_counts),
        "mapped_counts": dict(mapped_class_counts)
    }


def generate_dataset_yaml(dataset_dir, output_yaml_path, mapping_config):
    """Generates the dataset.yaml file required by Ultralytics YOLO training."""
    dataset_path = Path(dataset_dir).resolve()
    
    # Use target classes
    target_classes = mapping_config['target_classes']
    names_dict = {i: name for i, name in enumerate(target_classes)}

    yaml_data = {
        'path': str(dataset_path),
        'train': 'train/images',
        'val': 'val/images',
        'names': names_dict
    }

    output_path = Path(output_yaml_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(f"[+] Dataset YAML configuration generated successfully at: {output_path}")


def convert_annotations_to_mapped(dataset_dir, mapping_config):
    """
    Creates converted target class label files mapping 16 original TrashCan IDs
    to target class indices in target_classes list.
    """
    target_classes = mapping_config['target_classes']
    class_to_target_id = {name: i for i, name in enumerate(target_classes)}
    mapping = mapping_config['mapping']

    dataset_path = Path(dataset_dir)
    splits = ['train', 'val']

    converted_count = 0
    for split in splits:
        lbl_dir = dataset_path / split / 'labels'
        if not lbl_dir.exists():
            continue

        for lbl_file in lbl_dir.glob('*.txt'):
            lines = lbl_file.read_text().strip().split('\n')
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                orig_id = parts[0]
                mapped_name = mapping.get(orig_id, "Other_Waste")
                new_id = class_to_target_id.get(mapped_name, class_to_target_id["Other_Waste"])
                new_line = f"{new_id} " + " ".join(parts[1:])
                new_lines.append(new_line)
            
            # Write mapped label back
            lbl_file.write_text('\n'.join(new_lines))
            converted_count += 1

    print(f"[+] Converted {converted_count} label files to target class schema.")


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent
    mapping_path = base_dir / 'dataset' / 'class_mapping.json'
    
    # Locate dataset
    trash_dataset = Path(r"c:\Users\smesh\Downloads\trash_inst_material\trash_inst_material")
    if not trash_dataset.exists():
        trash_dataset = base_dir / 'dataset'

    if mapping_path.exists():
        mapping_config = load_class_mapping(mapping_path)
        check_dataset_integrity(trash_dataset, mapping_config)
        generate_dataset_yaml(trash_dataset, base_dir / 'dataset' / 'dataset.yaml', mapping_config)
    else:
        print(f"[ERROR] Mapping configuration file not found at: {mapping_path}")
