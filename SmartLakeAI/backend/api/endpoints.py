"""
SmartLake AI FastAPI REST API Router & Endpoints
-----------------------------------------------
Defines endpoints for image/video upload inference, camera streaming, statistics queries,
model information, and report generation.
"""

import os
import shutil
import tempfile
import torch
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, Query, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse

from backend.config import DB_PATH, MODEL_WEIGHTS_PATH, SYSTEM_CONFIG
from backend.services.ai_service import AIService
from backend.services.report_service import generate_pollution_report
from backend.database.models import init_db, fetch_statistics_summary, fetch_all_detections

# Initialize database tables
init_db(DB_PATH)

# Initialize AI Service
ai_service = AIService()

router = APIRouter()


@router.get("/")
def read_root():
    return {
        "title": SYSTEM_CONFIG['project']['name'],
        "subtitle": SYSTEM_CONFIG['project']['subtitle'],
        "status": "Online",
        "documentation": "/docs"
    }


@router.get("/health")
def health_check():
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    return {
        "status": "healthy",
        "cuda_available": cuda_avail,
        "device": gpu_name,
        "model_loaded": True
    }


@router.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Accepts JPG/PNG image upload and returns detection results & statistics."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image (JPG/PNG).")

    contents = await file.read()
    try:
        res = ai_service.process_image(contents, filename=file.filename)
        # Return JSON with base64 encoded image or metadata
        import base64
        b64_img = base64.b64encode(res['annotated_image_bytes']).decode('utf-8')
        
        return {
            "session_id": res["session_id"],
            "filename": res["filename"],
            "total_objects": res["total_objects"],
            "detections": res["detections"],
            "statistics": res["statistics"],
            "annotated_image_base64": f"data:image/jpeg;base64,{b64_img}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image inference error: {str(e)}")


@router.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    """Accepts MP4/AVI video upload, processes frame-by-frame with tracking, returns stats."""
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid video (MP4/AVI/MOV).")

    temp_dir = Path(tempfile.mkdtemp())
    input_path = temp_dir / file.filename
    output_path = temp_dir / f"processed_{file.filename}"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        res = ai_service.process_video(input_path, output_path)
        return {
            "status": "success",
            "session_id": res["session_id"],
            "frames_processed": res["frames_processed"],
            "processing_time_sec": res["processing_time_sec"],
            "statistics": res["statistics"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")


@router.post("/predict/camera/start")
def start_camera(camera_source: str = Query("0")):
    """Starts live camera / RTSP stream capture."""
    success = ai_service.start_camera(camera_source=camera_source)
    if not success:
        raise HTTPException(status_code=500, detail="Could not open camera stream source.")
    return {"status": "started", "camera_source": camera_source}


@router.post("/predict/camera/stop")
def stop_camera():
    """Stops live camera stream."""
    ai_service.stop_camera()
    return {"status": "stopped"}


@router.get("/predict/camera/stream")
def camera_stream():
    """Yields real-time MJPEG video stream for dashboard image tag."""
    return StreamingResponse(
        ai_service.camera_processor.generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/statistics")
def get_statistics():
    """Fetches aggregated statistics across all historical sessions."""
    summary = fetch_statistics_summary(DB_PATH)
    from ai.statistics import calculate_waste_statistics
    # Re-evaluate pollution level indicator
    mock_dets = []
    for cls_name, count in summary['category_counts'].items():
        for _ in range(count):
            mock_dets.append({'class': cls_name, 'confidence': 0.90})
    
    evaluated = calculate_waste_statistics(mock_dets)
    summary['pollution_indicator'] = evaluated['pollution_indicator']
    summary['pollution_level'] = evaluated['pollution_level']
    summary['most_common_waste'] = evaluated['most_common_waste']
    return summary


@router.get("/detections")
def get_detections(limit: int = 50):
    """Fetches recent detection logs from SQLite database."""
    return fetch_all_detections(DB_PATH, limit=limit)


@router.get("/model/info")
def get_model_info():
    """Returns active model details, supported class list, and hardware parameters."""
    return {
        "model_name": SYSTEM_CONFIG['project']['name'],
        "weights_path": str(MODEL_WEIGHTS_PATH),
        "classes": ai_service.predictor.class_names,
        "conf_threshold": ai_service.predictor.conf_threshold,
        "iou_threshold": ai_service.predictor.iou_threshold,
        "device": ai_service.predictor.device,
        "cuda_available": torch.cuda.is_available(),
        "unrepresented_classes_note": "Glass and Thermocol/Foam require secondary lake dataset collection."
    }


@router.get("/reports/generate")
def generate_report():
    """Generates pollution report based on historical statistics."""
    summary = fetch_statistics_summary(DB_PATH)
    from ai.statistics import calculate_waste_statistics
    
    mock_dets = []
    for cls_name, count in summary.get('category_counts', {}).items():
        for _ in range(count):
            mock_dets.append({'class': cls_name, 'confidence': 0.90})

    evaluated = calculate_waste_statistics(mock_dets)
    evaluated['category_counts'] = summary['category_counts']
    evaluated['total_objects'] = summary['total_objects']
    evaluated['total_waste'] = summary['total_waste']

    report_res = generate_pollution_report(evaluated, session_id="historical_summary", source_info="SmartLake System Aggregated Log")
    return report_res
