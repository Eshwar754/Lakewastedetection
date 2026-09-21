# SmartLake AI — System Architecture, Implementation & Model Training Documentation

## 1. Executive Summary & Overview

**SmartLake AI** is an intelligent, end-to-end computer-vision and IoT monitoring system designed for real-time detection, instance segmentation, classification, multi-object tracking, and density estimation of visible waste in lakes, rivers, and water bodies.

### Key Objectives
* **Automated Waste Identification**: Detect and delineate precise boundaries (masks) of floating or submerged debris.
* **13-Category Taxonomic Classification**: Categorize detected objects into 10 waste types and 3 natural aquatic categories.
* **Real-time Stream Monitoring**: Process live camera feeds (webcams, RTSP streams from autonomous cleaning boats or stationary lake sensors) and video files.
* **AI-Based Visual Waste Density Indicator**: Automatically evaluate pollution severity (Low, Moderate, High, Critical).
* **Full-Stack Web Interface**: Provide a real-time glassmorphic monitoring dashboard with dynamic charts, historical detection tables, and live streams.
* **Automated Report Generation**: Generate PDF and CSV pollution assessment reports for environmental agencies.

---

## 2. How the System Works (Runtime Workflow)

The SmartLake AI system operates through a multi-stage sequential pipeline:

```
[ Input Stream / Image / Video ]
               │
               ▼
   [ Preprocessing & Resizing ]
               │
               ▼
[ YOLOv8-seg Instance Segmentation Model ]
   ├── Bounding Box Detection (X, Y, W, H)
   ├── Class Classification & Confidence Score
   └── Pixel-level Polygon Mask Generation
               │
               ▼
   [ Multi-Object Tracking Engine ]
   (Assigns persistent IDs to debris across video frames)
               │
               ▼
 [ Waste Statistics & Density Calculation ]
   ├── Count Waste vs. Natural Objects
   ├── Calculate Percentage Distribution per Material
   └── Compute Pollution Density Index (Low/Moderate/High/Critical)
               │
               ▼
      [ FastAPI Backend Server ]
   ├── Store Detections in SQLite Database
   ├── Stream MJPEG Live Feed
   └── Expose REST API Endpoints
               │
               ▼
   [ Frontend Glassmorphic Dashboard ]
   ├── Real-Time Visual Overlays & Bounding Polygons
   ├── Interactive Analytics (Chart.js)
   └── Automated PDF/CSV Pollution Reports
```

### Workflow Steps Explained:
1. **Data Ingestion**: The system accepts single images (JPEG/PNG), video files (MP4/AVI/MOV), or live video streams (Webcam index or RTSP IP camera feed).
2. **AI Inference**: The frame passes through the trained **YOLOv8 Nano Instance Segmentation** model (`models/best.pt`).
3. **Mask & Box Generation**: For every detected item, the model outputs bounding coordinates, class confidence, and polygonal coordinates outlining the exact mask shape.
4. **Multi-Object Tracking**: For video and live camera feeds, a multi-object tracker maintains persistent IDs across consecutive frames to avoid double-counting drifting debris.
5. **Pollution Indicator Evaluation**:
   - **Low Pollution**: 0 – 5 detected waste objects.
   - **Moderate Pollution**: 6 – 15 detected waste objects.
   - **High Pollution**: 16 – 30 detected waste objects.
   - **Critical Pollution**: 31+ detected waste objects.
6. **Database Persistence & Streaming**: Results are saved to an SQLite database (`smartlake.db`), and processed frames are rendered with semi-transparent mask overlays and broadcasted over HTTP MJPEG streams.
7. **Dashboard & Reporting**: The frontend consumes FastAPI endpoints to display real-time telemetry and generate downloadable PDF/CSV reports.

---

## 3. How the System is Built (Architecture & Technology Stack)

### Technology Stack Overview

| Layer | Technologies & Tools | Description |
| :--- | :--- | :--- |
| **AI / Machine Learning** | Python 3.10+, PyTorch, Ultralytics YOLOv8-seg, OpenCV, NumPy, PIL | Computer vision, instance segmentation, image processing |
| **Backend Web Server** | FastAPI, Uvicorn, Pydantic, SQLite3, YAML | Asynchronous REST API, static file hosting, database management |
| **Frontend UI** | HTML5, Vanilla CSS3 (Glassmorphic dark design system), JavaScript (ES6+), Chart.js | Dynamic real-time monitoring interface with live graphs |
| **Reporting & Export** | FPDF / ReportLab engine, CSV engine | Automated PDF and spreadsheet report generation |
| **Configuration** | `config.yaml`, `class_mapping.json` | System settings, model thresholds, class mapping dictionary |

---

### Project Directory & Modular Architecture

```
SmartLakeAI/
├── backend/
│   ├── main.py                  # FastAPI application entrypoint & static mounting
│   ├── config.py                # System configuration loader
│   ├── api/
│   │   └── endpoints.py         # REST API routes (/predict, /camera, /statistics, /reports)
│   ├── services/
│   │   ├── ai_service.py        # Centralized inference & camera controller service
│   │   └── report_service.py    # PDF/CSV pollution report generator
│   └── database/
│       ├── db.py                # SQLite database connection manager
│       └── models.py            # Database schema definition & query handlers
│
├── ai/
│   ├── dataset_check.py         # Dataset integrity verification & dataset.yaml generator
│   ├── visualize_dataset.py     # Ground truth mask & polygon visualization script
│   ├── train.py                 # YOLO instance segmentation training pipeline
│   ├── validate.py              # Validation mAP evaluation script
│   ├── test.py                  # Test set evaluation script
│   ├── predict.py               # Single image inference engine with mask overlays
│   ├── tracker.py               # Multi-object tracking processor for streams
│   ├── video.py                 # Offline video file inference processor
│   ├── camera.py                # Live webcam & RTSP IP stream processor
│   └── statistics.py            # Waste metrics & pollution density indicator calculator
│
├── dataset/
│   ├── class_mapping.json       # Original TrashCan class ID to 13 target classes mapping
│   └── dataset.yaml             # Ultralytics dataset configuration file
│
├── models/
│   └── best.pt                  # Trained YOLO instance segmentation PyTorch weights
│
├── frontend/
│   ├── index.html               # System landing page
│   ├── dashboard.html           # Real-time monitoring dashboard
│   ├── style.css                # Custom glassmorphic dark CSS design system
│   └── app.js                   # Client-side API fetch controller & dynamic UI renderer
│
├── reports/                     # Output directory for generated PDF pollution reports
├── config.yaml                  # System-wide configuration file
├── requirements.txt             # Python dependency manifest
├── README.md                    # Quick execution guide
└── PROJECT_DOCUMENTATION.md     # Detailed architecture & technical reference guide
```

---

## 4. Model Training & AI Pipeline

### Dataset Source & Mapping Schema
The model is trained using underwater and lake surface waste imagery derived from the **TrashCan** instance segmentation dataset, mapped into **13 target categories**:

#### 13 Target Classes:
1. **Plastic** (`trash_plastic`)
2. **Paper** (`trash_paper`)
3. **Glass** (*Marked as unrepresented; requires secondary lake dataset*)
4. **Metal** (`trash_metal`)
5. **Rubber** (`trash_rubber`)
6. **Fabric** (`trash_fabric`)
7. **Wood** (`trash_wood`)
8. **Fishing_Gear** (`trash_fishing_gear`)
9. **Thermocol/Foam** (*Marked as unrepresented; requires secondary lake dataset*)
10. **Other_Waste** (`trash_etc`, `rov`)
11. **Plant** (`plant`)
12. **Fish** (`animal_fish`)
13. **Other_Animal** (`animal_starfish`, `animal_shells`, `animal_crab`, `animal_eel`, `animal_etc`)

---

### Data Preprocessing & Validation Pipeline (`ai/dataset_check.py`)
Before training, the dataset undergoes strict integrity validation:
* **Corrupted Image Filtering**: Images are opened and verified using PIL to prevent runtime crashes during training.
* **Label-to-Image Pairing**: Ensures every bounding polygon label file matches an image file.
* **Class ID Remapping**: Remaps original 16 TrashCan class IDs into the 13 target index schema.
* **`dataset.yaml` Generation**: Automatically outputs the standard Ultralytics configuration containing train/val paths and class index definitions.

---

### Model Architecture Selection
* **Architecture**: **YOLOv8 Nano Instance Segmentation (`yolov8n-seg.pt`)**
* **Backbone**: CSPDarknet feature extractor with C2f module blocks.
* **Neck**: Feature Pyramid Network (FPN) + Path Aggregation Network (PANet).
* **Head**: Dual-head architecture outputting box regression, class probabilities, and 32 prototype mask coefficients per detection.

---

### Training Configurations & Hyperparameters (`config.yaml` / `ai/train.py`)

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Base Model** | `yolov8n-seg.pt` | Transfer learning from COCO pretrained instance segmentation weights |
| **Image Resolution** | 640 x 640 | Optimal balance between spatial mask accuracy and real-time processing speed |
| **Epochs** | 30 | Sufficient for transfer learning convergence on lake waste features |
| **Batch Size** | 16 | Optimized for standard GPU VRAM memory constraints |
| **Learning Rate (`lr0`)** | 0.01 | Initial learning rate with cosine learning rate schedule decay |
| **Patience** | 10 | Early stopping criteria to prevent overfitting |
| **Optimizer** | AdamW / SGD | Adaptive weight decay optimizer |
| **Augmentation** | Mosaic, Flip, Color Jitter | Enhances model robustness against water reflections, ripples, and lighting shifts |

---

### Loss Functions
Training optimizes a combined multi-task loss function:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{box}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{dfl}} + \lambda_{\text{mask}} \mathcal{L}_{\text{mask}}$$

* **Box Loss ($\mathcal{L}_{\text{box}}$)**: Complete Intersection over Union (CIoU) loss for precise bounding box localization.
* **Class Loss ($\mathcal{L}_{\text{cls}}$)**: Binary Cross-Entropy (BCE) loss for 13-class classification.
* **Distribution Focal Loss ($\mathcal{L}_{\text{dfl}}$)**: Bounding box boundary regression.
* **Mask Loss ($\mathcal{L}_{\text{mask}}$)**: Binary Cross-Entropy over prototype mask overlays to ensure accurate object boundaries.

---

### Validation & Performance Metrics (`ai/validate.py`)
Upon completion of training, the model is evaluated on the validation split:
* **mAP@50 (Box & Mask)**: Mean Average Precision at 50% Intersection over Union threshold.
* **mAP@50-95 (Box & Mask)**: Overall precision metric across IoU thresholds from 0.50 to 0.95.
* **Precision & Recall**: Evaluates false positive and false negative trade-offs.
* **Confusion Matrix**: Identifies cross-class confusion between waste and natural aquatic species.
* **Best Model Export**: The highest-performing checkpoint is automatically copied to `models/best.pt`.

---

## 5. API Endpoints Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/health` | `GET` | System health check, GPU status, and model readiness |
| `/api/predict/image` | `POST` | Processes uploaded image file and returns base64 annotated image & stats JSON |
| `/api/predict/video` | `POST` | Processes uploaded MP4/AVI video with multi-object tracking |
| `/api/predict/camera/start` | `POST` | Starts live camera/RTSP stream capture |
| `/api/predict/camera/stop` | `POST` | Stops live camera capture |
| `/api/predict/camera/stream` | `GET` | Delivers live MJPEG video stream for dashboard rendering |
| `/api/statistics` | `GET` | Fetches aggregated historical waste counts & pollution indicator status |
| `/api/detections` | `GET` | Fetches recent detection entries from SQLite database |
| `/api/model/info` | `GET` | Returns active model metadata, supported classes, and hardware parameters |
| `/api/reports/generate` | `GET` | Triggers creation of downloadable PDF pollution report |

---

## 6. How to Run the Project

```powershell
# Step 1: Install Dependencies
python -m pip install -r requirements.txt

# Step 2: Validate Dataset & Generate dataset.yaml
python ai/dataset_check.py

# Step 3: Train YOLO Segmentation Model
python ai/train.py

# Step 4: Validate Model
python ai/validate.py

# Step 5: Test Single Image Inference
python ai/predict.py

# Step 6: Launch FastAPI Server & Monitoring Dashboard
python backend/main.py
```
*Access Dashboard in Web Browser at:* `http://127.0.0.1:8000/dashboard`
