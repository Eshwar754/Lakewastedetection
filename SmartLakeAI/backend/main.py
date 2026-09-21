"""
SmartLake AI - Main FastAPI Web Application
--------------------------------------------
Launches uvicorn server, handles CORS, mounts static frontend directory, and registers API routers.
"""

import uvicorn
import sys
from pathlib import Path

# Add SmartLakeAI project root to sys.path
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.config import SYSTEM_CONFIG
from backend.api.endpoints import router as api_router

app = FastAPI(
    title=SYSTEM_CONFIG['project']['name'],
    description=SYSTEM_CONFIG['project']['subtitle'],
    version=SYSTEM_CONFIG['project']['version']
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include REST API endpoints
app.include_router(api_router, prefix="/api")

# Serve static frontend files
FRONTEND_DIR = Path(__file__).resolve().parent.parent / 'frontend'
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    @app.get("/dashboard")
    async def serve_dashboard():
        return FileResponse(str(FRONTEND_DIR / "dashboard.html"))


if __name__ == "__main__":
    print(f"==================================================")
    print(f"  Starting SmartLake AI Web Server")
    print(f"  URL: http://127.0.0.1:8000")
    print(f"  Dashboard: http://127.0.0.1:8000/dashboard")
    print(f"==================================================")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
