"""
SmartLake AI - Video File Inference Pipeline
---------------------------------------------
Processes uploaded MP4/AVI videos frame-by-frame, applying instance segmentation,
multi-object tracking, real-time FPS overlay, and generating consolidated statistics.
"""

import cv2
import time
import json
import sys
from pathlib import Path

# Add project root to sys.path
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from ai.predict import SmartLakePredictor
from ai.tracker import WasteTracker
from ai.statistics import calculate_waste_statistics


def process_video_file(video_path, output_path=None, weights_path=None):
    """Processes video file, applying segmentation, tracking, and saving result."""
    print(f"==================================================")
    print(f"  SmartLake AI: Video File Inference")
    print(f"  Input Video: {video_path}")
    print(f"==================================================")

    predictor = SmartLakePredictor(weights_path=weights_path)
    tracker = WasteTracker()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_writer = None
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    all_tracked_detections = []
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        t0 = time.time()

        # Run model inference on frame
        annotated_frame, meta = predictor.predict_image(frame)
        raw_detections = meta['detections']

        # Run multi-object tracking
        tracked_dets = tracker.update(raw_detections, frame_idx)
        all_tracked_detections.extend(tracked_dets)

        # Draw persistent object IDs on frame
        for det in tracked_dets:
            track_id = det.get('track_id')
            x1, y1, x2, y2 = det['bbox']
            cv2.putText(annotated_frame, f"ID #{track_id}", (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        t1 = time.time()
        curr_fps = round(1.0 / (t1 - t0 + 1e-6), 1)

        # Draw status overlay
        cv2.putText(annotated_frame, f"SmartLake AI | FPS: {curr_fps} | Waste Count: {len(tracked_dets)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if out_writer:
            out_writer.write(annotated_frame)

    cap.release()
    if out_writer:
        out_writer.release()

    total_time = round(time.time() - start_time, 2)
    summary_stats = calculate_waste_statistics(all_tracked_detections)

    print(f"\n==================================================")
    print(f"  Video Processing Complete!")
    print(f"  - Total Frames Processed: {frame_idx}")
    print(f"  - Total Processing Time: {total_time}s")
    print(f"  - Total Unique Tracked Waste: {summary_stats['total_waste']}")
    print(f"  - Pollution Indicator: {summary_stats['pollution_indicator']}")
    print(f"==================================================\n")

    return {
        "output_path": str(output_path) if output_path else None,
        "frames_processed": frame_idx,
        "processing_time_sec": total_time,
        "statistics": summary_stats,
        "tracked_objects": tracker.get_tracked_summary()
    }


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent
    sample_vid = base_dir / 'samples' / 'test_lake.mp4'
    if sample_vid.exists():
        process_video_file(sample_vid, output_path='output_processed.mp4')
    else:
        print(f"[!] Sample video file not found at {sample_vid}. Provide a valid video path.")
