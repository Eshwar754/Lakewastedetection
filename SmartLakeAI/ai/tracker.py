"""
SmartLake AI - Multi-Object Tracker for Video Streams
------------------------------------------------------
Maintains persistent unique IDs across video frames to avoid counting the same
floating waste object multiple times.
"""

import numpy as np


class WasteTracker:
    def __init__(self, max_disappeared=15, iou_threshold=0.3):
        self.next_object_id = 1
        self.objects = {}       # {id: {'bbox': [...], 'class': str, 'confidence': float, 'first_frame': int, 'last_frame': int}}
        self.disappeared = {}   # {id: count}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold

    def _compute_iou(self, boxA, boxB):
        """Calculates Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[0])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[0])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def update(self, new_detections, frame_idx):
        """
        Updates tracked objects with new detections in the current frame.
        new_detections: list of dicts [{'bbox': [x1,y1,x2,y2], 'class': str, 'confidence': float, 'mask': [...]}]
        Returns list of updated detections with added 'track_id' field.
        """
        if len(new_detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]
            return []

        # If no tracked objects yet, register all new detections
        if len(self.objects) == 0:
            tracked_results = []
            for det in new_detections:
                obj_id = self.next_object_id
                self.next_object_id += 1
                self.objects[obj_id] = {
                    'bbox': det['bbox'],
                    'class': det['class'],
                    'confidence': det['confidence'],
                    'first_frame': frame_idx,
                    'last_frame': frame_idx,
                    'mask': det.get('mask')
                }
                self.disappeared[obj_id] = 0
                det_copy = dict(det)
                det_copy['track_id'] = obj_id
                tracked_results.append(det_copy)
            return tracked_results

        # Match existing tracked objects with new detections via IoU
        object_ids = list(self.objects.keys())
        existing_boxes = [self.objects[oid]['bbox'] for oid in object_ids]
        new_boxes = [d['bbox'] for d in new_detections]

        iou_matrix = np.zeros((len(existing_boxes), len(new_boxes)), dtype=np.float32)
        for i, ebox in enumerate(existing_boxes):
            for j, nbox in enumerate(new_boxes):
                iou_matrix[i, j] = self._compute_iou(ebox, nbox)

        matched_existing = set()
        matched_new = set()

        if iou_matrix.size > 0:
            # Greedy matching by highest IoU
            while True:
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_iou = iou_matrix[max_idx]

                if max_iou < self.iou_threshold:
                    break

                i, j = max_idx
                if i not in matched_existing and j not in matched_new:
                    matched_existing.add(i)
                    matched_new.add(j)

                    obj_id = object_ids[i]
                    det = new_detections[j]

                    # Update tracked object
                    self.objects[obj_id]['bbox'] = det['bbox']
                    self.objects[obj_id]['class'] = det['class']
                    self.objects[obj_id]['confidence'] = det['confidence']
                    self.objects[obj_id]['last_frame'] = frame_idx
                    self.objects[obj_id]['mask'] = det.get('mask')
                    self.disappeared[obj_id] = 0

                iou_matrix[i, :] = -1
                iou_matrix[:, j] = -1

        # Handle unmatched existing objects (increment disappeared count)
        for i, obj_id in enumerate(object_ids):
            if i not in matched_existing:
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]

        # Register new unmatched detections
        tracked_results = []
        for j, det in enumerate(new_detections):
            if j in matched_new:
                # Find matching object_id
                for i in matched_existing:
                    # Corresponding j was matched
                    pass
                # Get ID from objects
                for oid, data in self.objects.items():
                    if data['bbox'] == det['bbox']:
                        det_copy = dict(det)
                        det_copy['track_id'] = oid
                        tracked_results.append(det_copy)
                        break
            else:
                # Register new object
                obj_id = self.next_object_id
                self.next_object_id += 1
                self.objects[obj_id] = {
                    'bbox': det['bbox'],
                    'class': det['class'],
                    'confidence': det['confidence'],
                    'first_frame': frame_idx,
                    'last_frame': frame_idx,
                    'mask': det.get('mask')
                }
                self.disappeared[obj_id] = 0
                det_copy = dict(det)
                det_copy['track_id'] = obj_id
                tracked_results.append(det_copy)

        return tracked_results

    def get_tracked_summary(self):
        """Returns summary of all unique tracked objects over stream lifetime."""
        return {
            'total_unique_objects': len(self.objects),
            'objects': self.objects
        }
