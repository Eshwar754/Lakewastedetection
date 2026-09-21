"""
SmartLake AI Database Schema Definition (SQLite)
-----------------------------------------------
Defines tables for session tracking, detection logs, and waste statistics history.
"""

import sqlite3
import datetime
from pathlib import Path


def init_db(db_path):
    """Initializes SQLite database tables if they do not exist."""
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()

    # 1. Sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            total_waste_detected INTEGER DEFAULT 0,
            pollution_indicator TEXT
        )
    ''')

    # 2. Detections table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp TEXT NOT NULL,
            source_type TEXT NOT NULL,
            class TEXT NOT NULL,
            confidence REAL NOT NULL,
            bbox TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )
    ''')

    # 3. Waste Statistics table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS waste_statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp TEXT NOT NULL,
            total_objects INTEGER DEFAULT 0,
            total_waste INTEGER DEFAULT 0,
            total_natural INTEGER DEFAULT 0,
            plastic_count INTEGER DEFAULT 0,
            paper_count INTEGER DEFAULT 0,
            glass_count INTEGER DEFAULT 0,
            metal_count INTEGER DEFAULT 0,
            rubber_count INTEGER DEFAULT 0,
            fabric_count INTEGER DEFAULT 0,
            wood_count INTEGER DEFAULT 0,
            fishing_gear_count INTEGER DEFAULT 0,
            other_waste_count INTEGER DEFAULT 0,
            pollution_level TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )
    ''')

    conn.commit()
    conn.close()
    print(f"[+] SQLite database initialized at: {db_file}")


def save_detection_record(db_path, session_id, source_type, detections, stats):
    """Saves detection batch and summary statistics into SQLite."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    now_str = datetime.datetime.now().isoformat()

    # 1. Ensure session exists
    cursor.execute('SELECT session_id FROM sessions WHERE session_id = ?', (session_id,))
    if not cursor.fetchone():
        cursor.execute('''
            INSERT INTO sessions (session_id, source_type, started_at, total_waste_detected, pollution_indicator)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, source_type, now_str, stats.get('total_waste', 0), stats.get('pollution_indicator', 'Low Pollution')))

    # 2. Insert detection items
    for det in detections:
        cursor.execute('''
            INSERT INTO detections (session_id, timestamp, source_type, class, confidence, bbox)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, now_str, source_type, det['class'], det['confidence'], str(det['bbox'])))

    # 3. Insert summary statistics record
    cat_counts = stats.get('category_counts', {})
    cursor.execute('''
        INSERT INTO waste_statistics (
            session_id, timestamp, total_objects, total_waste, total_natural,
            plastic_count, paper_count, glass_count, metal_count, rubber_count,
            fabric_count, wood_count, fishing_gear_count, other_waste_count, pollution_level
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        session_id, now_str, stats.get('total_objects', 0), stats.get('total_waste', 0), stats.get('total_natural', 0),
        cat_counts.get('Plastic', 0), cat_counts.get('Paper', 0), cat_counts.get('Glass', 0),
        cat_counts.get('Metal', 0), cat_counts.get('Rubber', 0), cat_counts.get('Fabric', 0),
        cat_counts.get('Wood', 0), cat_counts.get('Fishing_Gear', 0), cat_counts.get('Other_Waste', 0),
        stats.get('pollution_indicator', 'Low Pollution')
    ))

    # Update session total
    cursor.execute('''
        UPDATE sessions SET total_waste_detected = total_waste_detected + ?, pollution_indicator = ?
        WHERE session_id = ?
    ''', (stats.get('total_waste', 0), stats.get('pollution_indicator', 'Low Pollution'), session_id))

    conn.commit()
    conn.close()


def fetch_statistics_summary(db_path):
    """Retrieves aggregated statistics across all historical sessions."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute('''
        SELECT 
            SUM(total_objects), SUM(total_waste), SUM(total_natural),
            SUM(plastic_count), SUM(paper_count), SUM(glass_count),
            SUM(metal_count), SUM(rubber_count), SUM(fabric_count),
            SUM(wood_count), SUM(fishing_gear_count), SUM(other_waste_count)
        FROM waste_statistics
    ''')
    row = cursor.fetchone()
    conn.close()

    if not row or row[0] is None:
        return {
            "total_objects": 0, "total_waste": 0, "total_natural": 0,
            "category_counts": {
                "Plastic": 0, "Paper": 0, "Glass": 0, "Metal": 0, "Rubber": 0,
                "Fabric": 0, "Wood": 0, "Fishing_Gear": 0, "Other_Waste": 0
            }
        }

    return {
        "total_objects": row[0] or 0,
        "total_waste": row[1] or 0,
        "total_natural": row[2] or 0,
        "category_counts": {
            "Plastic": row[3] or 0, "Paper": row[4] or 0, "Glass": row[5] or 0,
            "Metal": row[6] or 0, "Rubber": row[7] or 0, "Fabric": row[8] or 0,
            "Wood": row[9] or 0, "Fishing_Gear": row[10] or 0, "Other_Waste": row[11] or 0
        }
    }


def fetch_all_detections(db_path, limit=50):
    """Retrieves recent detection logs."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, session_id, timestamp, source_type, class, confidence, bbox
        FROM detections ORDER BY id DESC LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()

    detections = []
    for r in rows:
        detections.append({
            "id": r[0],
            "session_id": r[1],
            "timestamp": r[2],
            "source_type": r[3],
            "class": r[4],
            "confidence": r[5],
            "bbox": r[6]
        })
    return detections
