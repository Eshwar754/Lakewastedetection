"""
SmartLake AI - Real-Time Camera Stream Processor
-------------------------------------------------
Captures live frames from a webcam or RTSP camera, applies instance segmentation
and multi-object tracking, calculates FPS and inference latency, and yields JPEG frames for FastAPI streaming.
"""

import cv2
import time
import threading
import sys
from pathlib import Path

# Add project root to sys.path
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from ai.predict import SmartLakePredictor
from ai.tracker import WasteTracker
from ai.statistics import calculate_waste_statistics


class CameraStreamProcessor:
    def __init__(self, camera_source=0, weights_path=None):
        self.camera_source = camera_source
        self.predictor = SmartLakePredictor(weights_path=weights_path)
        self.tracker = WasteTracker()
        self.cap = None
        self.is_running = False
        self.lock = threading.Lock()
        self.latest_stats = calculate_waste_statistics([])
        self.current_fps = 0.0
        self.inference_time_ms = 0.0

    def start(self):
        """Starts camera stream capture loop."""
        with self.lock:
            if self.is_running:
                return
            # Convert integer camera index if numeric string
            src = int(self.camera_source) if str(self.camera_source).isdigit() else self.camera_source
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                print(f"[ERROR] Could not open camera source: {self.camera_source}")
                return False
            self.is_running = True
            print(f"[+] Camera stream started on source: {self.camera_source}")
            return True

    def stop(self):
        """Stops camera stream capture."""
        with self.lock:
            self.is_running = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            print(f"[+] Camera stream stopped.")

    def generate_mjpeg_stream(self):
        """Yields MJPEG encoded video frames for FastAPI StreamingResponse."""
        if not self.is_running:
            if not self.start():
                return

        frame_idx = 0
        while self.is_running and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print(f"[!] Warning: Camera frame read failed.")
                time.sleep(0.05)
                continue

            frame_idx += 1
            t0 = time.time()

            # Perform AI inference
            annotated_frame, meta = self.predictor.predict_image(frame)
            raw_dets = meta['detections']

            # Apply tracking
            tracked_dets = self.tracker.update(raw_dets, frame_idx)

            t1 = time.time()
            self.inference_time_ms = round((t1 - t0) * 1000.0, 1)
            self.current_fps = round(1.0 / (t1 - t0 + 1e-6), 1)

            # Update latest statistics
            self.latest_stats = calculate_waste_statistics(tracked_dets)

            # Draw tracked IDs & waste names on frame
            for det in tracked_dets:
                tid = det.get('track_id')
                cls_name = det.get('class', 'Waste').upper()
                x1, y1, x2, y2 = det['bbox']
                label_text = f" ID #{tid} | {cls_name} "
                cv2.putText(annotated_frame, label_text, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(annotated_frame, label_text, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 242, 254), 2, cv2.LINE_AA)

            # HUD overlay
            hud_text = f"SmartLake AI | FPS: {self.current_fps} | Latency: {self.inference_time_ms}ms | Waste: {self.latest_stats['total_waste']}"
            cv2.putText(annotated_frame, hud_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(annotated_frame, hud_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)


            # Encode frame to JPEG
            ret_encode, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret_encode:
                continue

            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def get_status(self):
        """Returns active stream performance and waste metrics."""
        return {
            "is_running": self.is_running,
            "fps": self.current_fps,
            "inference_time_ms": self.inference_time_ms,
            "statistics": self.latest_stats
        }


if __name__ == '__main__':
    cam_proc = CameraStreamProcessor(camera_source=0)
    print("Testing camera stream setup...")
    if cam_proc.start():
        print("Camera opened successfully. Press Ctrl+C to stop.")
        try:
            for _ in cam_proc.generate_mjpeg_stream():
                pass
        except KeyboardInterrupt:
            cam_proc.stop()
