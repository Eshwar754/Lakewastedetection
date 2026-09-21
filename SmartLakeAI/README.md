# SmartLake AI — Intelligent Lake Waste Detection, Classification & Pollution Monitoring System

SmartLake AI is an end-to-end computer-vision system designed to analyze images, videos, and live camera streams from lakes or water bodies to automatically detect, segment, classify, track, and report visible waste materials.

---

## 🌟 System Architecture & Project Structure

```
SmartLakeAI/
├── backend/
│   ├── main.py                  # FastAPI server launcher & static mounts
│   ├── config.py                # System configuration loader
│   ├── api/
│   │   └── endpoints.py         # REST API routes (/predict, /camera, /statistics, /detections)
│   ├── services/
│   │   ├── ai_service.py        # Inference & camera controller service
│   │   └── report_service.py    # Report generator
│   └── database/
│       ├── db.py                # Database setup
│       └── models.py            # SQLite database schema
│
├── ai/
│   ├── dataset_check.py         # Dataset integrity & YAML generator
│   ├── visualize_dataset.py     # Annotation overlay visualization
│   ├── train.py                 # YOLO segmentation model training pipeline
│   ├── validate.py              # Validation evaluation metrics script
│   ├── test.py                  # Testing script
│   ├── predict.py               # Single image inference engine
│   ├── tracker.py               # Multi-object tracker
│   ├── video.py                 # Video file inference processor
│   ├── camera.py                # Live webcam/RTSP stream processor
│   └── statistics.py            # Waste metrics & pollution indicator calculator
│
├── dataset/
│   ├── class_mapping.json       # Original TrashCan -> Target class mapping
│   └── dataset.yaml             # Generated YOLO dataset configuration
│
├── models/
│   └── best.pt                  # Trained YOLO instance segmentation weights
│
├── frontend/
│   ├── index.html               # Main landing page
│   ├── dashboard.html           # Real-time monitoring dashboard
│   ├── style.css                # Glassmorphic dark design system
│   └── app.js                   # Dashboard UI logic & API fetch controller
│
├── reports/                     # Output directory for generated pollution reports
├── config.yaml                  # System configuration settings
├── requirements.txt             # Dependency manifest
└── README.md                    # Beginner-friendly execution guide
```

---

## 📋 Recommended Target Classes (13 Categories)

| Target Class | TrashCan Source Mapping | Status in Dataset |
| :--- | :--- | :--- |
| **Plastic** | `trash_plastic` | ✅ Present |
| **Paper** | `trash_paper` | ✅ Present |
| **Glass** | *None* | ⚠️ Unrepresented (Requires Secondary Dataset) |
| **Metal** | `trash_metal` | ✅ Present |
| **Rubber** | `trash_rubber` | ✅ Present |
| **Fabric** | `trash_fabric` | ✅ Present |
| **Wood** | `trash_wood` | ✅ Present |
| **Fishing_Gear** | `trash_fishing_gear` | ✅ Present |
| **Thermocol/Foam** | *None* | ⚠️ Unrepresented (Requires Secondary Dataset) |
| **Other_Waste** | `trash_etc`, `rov` | ✅ Present |
| **Plant** | `plant` | ✅ Present |
| **Fish** | `animal_fish` | ✅ Present |
| **Other_Animal** | `animal_starfish`, `animal_shells`, `animal_crab`, `animal_eel`, `animal_etc` | ✅ Present |

> [!NOTE]
> **Important Dataset Rule**: As per guidelines, **Glass** and **Thermocol/Foam** are explicitly documented as requiring secondary lake dataset collection rather than assuming they exist in the original TrashCan dataset.

---

## 🚀 Step-by-Step Execution Guide

### Step 1: Navigate to Project Directory & Install Dependencies
- **What we are doing:** Changing directory into `SmartLakeAI` and installing required Python packages.
- **Exact Commands:**
  ```powershell
  cd SmartLakeAI
  python -m pip install -r requirements.txt
  ```

### Step 2: Validate Dataset Integrity & Generate YAML
- **What we are doing:** Scanning image/label pairs, detecting corrupted files, counting original and mapped class instances, and creating `dataset/dataset.yaml`.
- **Exact Command:**
  ```powershell
  python ai/dataset_check.py
  ```
- **Expected Output:**
  ```
  Integrity Summary:
  - Total Images Scanned: 7212
  - Total Labels Scanned: 7212
  - Corrupted Images: 0
  - Images Missing Labels: 0
  - Dataset YAML configuration generated successfully at: dataset/dataset.yaml
  ```

### Step 3: Visualize Sample Annotations
- **What we are doing:** Drawing segmentation masks and bounding polygons on sample images to verify annotations.
- **Exact Command:**
  ```powershell
  python ai/visualize_dataset.py
  ```
- **Expected Output:** Visualized images saved into `dataset/sample_visualizations/`.

### Step 4: Train YOLO Instance Segmentation Model
- **What we are doing:** Running transfer learning from `yolov8n-seg.pt` on the lake dataset.
- **Exact Command:**
  ```powershell
  python ai/train.py
  ```
- **Expected Output:** Saved model weights at `models/best.pt`, training loss curves, confusion matrix, precision, recall, and mAP metrics.

### Step 5: Validate Model Performance
- **What we are doing:** Calculating mAP50 and mAP50-95 scores on validation dataset.
- **Exact Command:**
  ```powershell
  python ai/validate.py
  ```

### Step 6: Test Single Image Inference
- **What we are doing:** Running instance segmentation on a single image.
- **Exact Command:**
  ```powershell
  python ai/predict.py
  ```

### Step 7: Launch FastAPI Web Backend & Dashboard
- **What we are doing:** Starting the web server and monitoring dashboard.
- **Exact Command:**
  ```powershell
  python backend/main.py
  ```
- **Accessing Dashboard:**
  Open your web browser and navigate to:
  `http://127.0.0.1:8000/dashboard`

---

## 📊 Pollution Density Indicator Scale

The system features an automated **AI-Based Visual Waste Density Indicator**:

- **Low Pollution:** 0 – 5 detected waste objects
- **Moderate Pollution:** 6 – 15 detected waste objects
- **High Pollution:** 16 – 30 detected waste objects
- **Critical Pollution:** 31+ detected waste objects

---

## 🔧 Troubleshooting & Common Errors

1. **`ModuleNotFoundError: No module named 'torch'` or `'ultralytics'`**
   - *Fix:* Ensure dependencies are installed by re-running `python -m pip install -r requirements.txt`.

2. **`CUDA Out of Memory` Error during training**
   - *Fix:* Reduce `batch_size` in `config.yaml` from 16 to 8 or 4.

3. **Camera Stream Not Opening**
   - *Fix:* Verify camera index in `config.yaml` (default is `0` for default webcam). If using RTSP, provide the RTSP stream URL.

---

*SmartLake AI System &copy; 2026 — Designed for Lake Environmental Monitoring and Autonomous Cleaning Vessels.*
