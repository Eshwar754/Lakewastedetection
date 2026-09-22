"""
SmartLake AI Backend Configuration Module
----------------------------------------
"""

import yaml
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

CONFIG_YAML_PATH = BASE_DIR / 'config.yaml'

with open(CONFIG_YAML_PATH, 'r') as f:
    SYSTEM_CONFIG = yaml.safe_load(f)

DB_PATH = BASE_DIR / SYSTEM_CONFIG['database'].get('sqlite_path', 'backend/database/smartlake.db')
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL_WEIGHTS_PATH = BASE_DIR / SYSTEM_CONFIG['model'].get('custom_weights', 'models/best.pt')
if not MODEL_WEIGHTS_PATH.exists():
    MODEL_WEIGHTS_PATH = BASE_DIR / SYSTEM_CONFIG['model'].get('pretrained_weights', 'yolov8n-seg.pt')

REPORTS_DIR = BASE_DIR / SYSTEM_CONFIG['reports'].get('output_dir', 'reports/')
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
