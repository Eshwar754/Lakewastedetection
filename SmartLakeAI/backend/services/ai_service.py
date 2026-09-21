"""
SmartLake AI Backend Service Wrapper
------------------------------------
Integrates YOLO segmentation model inference, live camera streaming controller,
and SQLite database saving.
"""

import uuid
import cv2
import numpy as np
from pathlib import Path

from ai.predict import SmartLakePredictor
from ai.camera import CameraStreamProcessor
from ai.video import process_video_file
from backend.config import DB_PATH, MODEL_WEIGHTS_PATH
from backend.database.models import save_detection_record


class AIService:
    def __init__(self):
        weights_str = str(MODEL_WEIGHTS_PATH) if MODEL_WEIGHTS_PATH.exists() else None
        self.predictor = SmartLakePredictor(weights_path=weights_str)
        self.camera_processor = CameraStreamProcessor(camera_source=0, weights_path=weights_str)

    def process_image(self, image_bytes, filename="uploaded.jpg"):
        """Decodes image bytes, runs prediction, logs to DB, and returns result."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file data.")

        annotated_img, meta = self.predictor.predict_image(img)

        # Encode annotated image to base64 or JPEG bytes
        ret, buf = cv2.imencode('.jpg', annotated_img)
        img_bytes = buf.tobytes()

        # Log to database
        session_id = f"img_{uuid.uuid4().hex[:8]}"
        save_detection_record(DB_PATH, session_id, "image", meta['detections'], meta['statistics'])

        return {
            "session_id": session_id,
            "filename": filename,
            "detections": meta['detections'],
            "total_objects": meta['statistics']['total_objects'],
            "statistics": meta['statistics'],
            "annotated_image_bytes": img_bytes
        }

    def process_video(self, input_video_path, output_video_path):
        """Processes video file asynchronously or synchronously and saves to DB."""
        session_id = f"vid_{uuid.uuid4().hex[:8]}"
        res = process_video_file(input_video_path, output_path=output_video_path)

        save_detection_record(DB_PATH, session_id, "video", [], res['statistics'])
        res['session_id'] = session_id
        return res

    def start_camera(self, camera_source=0):
        """Starts real-time live camera stream."""
        self.camera_processor.camera_source = camera_source
        return self.camera_processor.start()

    def stop_camera(self):
        """Stops live camera stream."""
        self.camera_processor.stop()

    def get_camera_status(self):
        """Gets active camera status and stats."""
        return self.camera_processor.get_status()
