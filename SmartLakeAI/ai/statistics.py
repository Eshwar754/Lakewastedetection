"""
SmartLake AI - Waste Statistics & Pollution Level Calculator
------------------------------------------------------------
Processes raw detection results to compute total waste, individual material counts,
percentages, dominant waste types, and the AI-based visual waste density indicator.
"""

from collections import defaultdict

# Category classifications
WASTE_CATEGORIES = {
    'Plastic', 'Paper', 'Glass', 'Metal', 'Rubber',
    'Fabric', 'Wood', 'Fishing_Gear', 'Thermocol/Foam', 'Other_Waste'
}

NATURAL_CATEGORIES = {'Plant', 'Fish', 'Other_Animal'}


def calculate_waste_statistics(detections, thresholds=None):
    """
    Given a list of detection dicts:
    [{'class': 'Plastic', 'confidence': 0.92, 'bbox': [x1,y1,x2,y2]}, ...]
    
    Returns structured waste statistics and pollution indicator.
    """
    if thresholds is None:
        thresholds = {
            'low': (0, 5),
            'moderate': (6, 15),
            'high': (16, 30),
            'critical': (31, 999999)
        }

    category_counts = defaultdict(int)
    confidence_sum = defaultdict(float)

    total_objects = len(detections)
    total_waste_count = 0
    total_natural_count = 0

    for det in detections:
        cls_name = det.get('class', 'Other_Waste')
        conf = det.get('confidence', 0.0)

        category_counts[cls_name] += 1
        confidence_sum[cls_name] += conf

        if cls_name in WASTE_CATEGORIES or (cls_name not in NATURAL_CATEGORIES):
            total_waste_count += 1
        else:
            total_natural_count += 1

    # Percentages of waste categories
    waste_percentages = {}
    for cls_name, count in category_counts.items():
        if cls_name in WASTE_CATEGORIES or (cls_name not in NATURAL_CATEGORIES):
            perc = (count / total_waste_count * 100.0) if total_waste_count > 0 else 0.0
            waste_percentages[cls_name] = round(perc, 1)

    # Average confidence per class
    avg_confidence = {
        cls_name: round(confidence_sum[cls_name] / count, 2)
        for cls_name, count in category_counts.items()
    }

    # Most common waste type
    most_common = "None"
    max_count = 0
    for cls_name, count in category_counts.items():
        if (cls_name in WASTE_CATEGORIES or cls_name not in NATURAL_CATEGORIES) and count > max_count:
            max_count = count
            most_common = cls_name

    # Determine AI-based visual waste density indicator level
    pollution_level = "Low"
    for level, (min_val, max_val) in thresholds.items():
        if min_val <= total_waste_count <= max_val:
            pollution_level = level.capitalize()
            break

    return {
        "total_objects": total_objects,
        "total_waste": total_waste_count,
        "total_natural": total_natural_count,
        "category_counts": dict(category_counts),
        "waste_percentages": waste_percentages,
        "avg_confidence": avg_confidence,
        "most_common_waste": most_common,
        "pollution_indicator": f"{pollution_level} Pollution",
        "pollution_level": pollution_level,
        "pollution_label": "AI-based visual waste density indicator"
    }
