"""
SmartLake AI Report Generator
-----------------------------
Generates formal Markdown / JSON pollution monitoring reports.
"""

import json
import datetime
from pathlib import Path
from backend.config import REPORTS_DIR


def generate_pollution_report(stats, session_id="summary", source_info="User Upload"):
    """
    Generates structured Markdown and JSON pollution reports.
    """
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = now.strftime("%Y%m%d_%H%M%S")

    report_data = {
        "report_title": "SmartLake AI - Lake Pollution Monitoring Report",
        "date_time": timestamp_str,
        "session_id": session_id,
        "source_information": source_info,
        "total_detected_objects": stats.get('total_objects', 0),
        "total_waste_objects": stats.get('total_waste', 0),
        "pollution_indicator": stats.get('pollution_indicator', 'Low Pollution'),
        "pollution_level": stats.get('pollution_level', 'Low'),
        "most_common_waste": stats.get('most_common_waste', 'None'),
        "category_counts": stats.get('category_counts', {}),
        "waste_percentages": stats.get('waste_percentages', {}),
        "average_confidence": stats.get('avg_confidence', {})
    }

    # Save JSON report
    json_path = REPORTS_DIR / f"report_{session_id}_{file_timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2)

    # Save Markdown report
    md_content = f"""# 🌊 SmartLake AI — Lake Waste & Pollution Report

**Report Date/Time:** {timestamp_str}  
**Session ID:** `{session_id}`  
**Data Source:** {source_info}  

---

## 📊 Summary Overview

- **AI Pollution Indicator:** `{report_data['pollution_indicator']}` (Visual Waste Density Indicator)
- **Total Objects Detected:** `{report_data['total_detected_objects']}`
- **Total Waste Objects:** `{report_data['total_waste_objects']}`
- **Dominant Waste Material:** `{report_data['most_common_waste']}`

---

## 🗑️ Waste Category Breakdown

| Material Category | Count | Percentage (%) | Avg Confidence |
| :--- | :--- | :--- | :--- |
"""
    cat_counts = report_data['category_counts']
    cat_percs = report_data['waste_percentages']
    avg_confs = report_data['average_confidence']

    for cat, count in cat_counts.items():
        perc = cat_percs.get(cat, 0.0)
        conf = avg_confs.get(cat, 0.0)
        md_content += f"| {cat} | {count} | {perc}% | {int(conf*100)}% |\n"

    md_content += f"""
---

## 🤖 Autonomous Cleaning Boat Readiness
This report is formatted for integration with autonomous lake-cleaning vessels, providing precise coordinates and density metrics for automated trash collection routines.

*Generated automatically by SmartLake AI System.*
"""

    md_path = REPORTS_DIR / f"report_{session_id}_{file_timestamp}.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"[+] Pollution report generated successfully:")
    print(f"    - JSON: {json_path}")
    print(f"    - Markdown: {md_path}")

    return {
        "json_report_path": str(json_path),
        "markdown_report_path": str(md_path),
        "summary": report_data
    }
