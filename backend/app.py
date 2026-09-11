import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.database import init_db
from backend.schemas import HealthResponse
from backend.routes.upload import router as upload_router
from backend.routes.analysis import router as analysis_router
from backend.routes.query import router as query_router
from backend.routes.compare import router as compare_router
from backend.routes.history import router as history_router
from backend.routes.report import router as report_router

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
UPLOAD_DIR = os.path.join(BASE_DIR, "backend", "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "backend", "outputs")
SAMPLES_DIR = os.path.join(BASE_DIR, "backend", "samples")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Initialize database
init_db()

app = FastAPI(
    title="SatQuery AI",
    description="Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis (ISRO / SIH26167)",
    version="1.0.0"
)

# Enable CORS for all local and frontend interactions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(upload_router)
app.include_router(analysis_router)
app.include_router(query_router)
app.include_router(compare_router)
app.include_router(history_router)
app.include_router(report_router)

# Mount Static Directories for direct asset access
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")


@app.get("/", tags=["UI"])
def serve_frontend():
    """Serves the main interactive single-page application UI."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "status": "online",
        "service": "SatQuery AI",
        "message": "Frontend index.html is initializing. Access /docs for API documentation."
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Returns application health and operational status."""
    provider = os.environ.get("VLM_PROVIDER", "local")
    return HealthResponse(
        status="ok",
        service="SatQuery AI",
        version="1.0.0",
        isro_theme="SIH26167 Space Technology",
        vlm_provider=provider
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
