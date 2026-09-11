import os
import shutil
import uuid
import json
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional, List

from backend.database import get_db
from backend.models import SessionRecord, ImageRecord
from backend.schemas import ImageUploadResponse, ImageMetadataSchema, SampleItem
from backend.services.image_processor import (
    is_valid_extension, get_safe_filename, extract_image_metadata, MAX_FILE_SIZE
)

router = APIRouter(prefix="/api", tags=["Upload & Samples"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload", response_model=ImageUploadResponse)
async def upload_satellite_image(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Uploads a satellite image (JPG, PNG, TIFF), extracts spatial/EXIF metadata,
    and registers image and session in the SQLite database.
    """
    if not is_valid_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Unsupported format. Please upload JPG, JPEG, PNG, or TIFF/TIF satellite imagery."
        )

    # Validate or create session
    if not session_id or session_id.strip() == "":
        session_id = f"sess_{uuid.uuid4().hex[:10]}"
        new_session = SessionRecord(id=session_id, title=f"Analysis of {file.filename}")
        db.add(new_session)
        db.commit()
    else:
        existing_session = db.query(SessionRecord).filter(SessionRecord.id == session_id).first()
        if not existing_session:
            new_session = SessionRecord(id=session_id, title=f"Analysis of {file.filename}")
            db.add(new_session)
            db.commit()

    safe_name = get_safe_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, safe_name)

    # Save uploaded file safely
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write image file: {str(e)}")

    # Extract metadata
    metadata = extract_image_metadata(save_path)

    # Register in DB
    img_record = ImageRecord(
        session_id=session_id,
        filename=safe_name,
        original_name=file.filename,
        path=save_path,
        image_type=metadata.get("format", "RGB"),
        width=metadata.get("width", 0),
        height=metadata.get("height", 0),
        file_size=metadata.get("file_size", 0),
        metadata_json=json.dumps(metadata)
    )
    db.add(img_record)
    db.commit()
    db.refresh(img_record)

    meta_schema = ImageMetadataSchema(
        width=metadata["width"],
        height=metadata["height"],
        channels=metadata["channels"],
        color_mode=metadata["color_mode"],
        file_size=metadata["file_size"],
        format=metadata["format"],
        has_georeference=metadata["has_georeference"],
        latitude=metadata["latitude"],
        longitude=metadata["longitude"],
        crs=metadata["crs"],
        bounding_box=metadata["bounding_box"],
        extra_tags=metadata.get("extra_tags")
    )

    return ImageUploadResponse(
        success=True,
        image_id=img_record.id,
        session_id=session_id,
        filename=safe_name,
        original_name=file.filename,
        url=f"/uploads/{safe_name}",
        metadata=meta_schema,
        message="Satellite image uploaded and cataloged successfully."
    )


@router.get("/samples", response_model=List[SampleItem])
def get_sample_satellite_images():
    """Returns catalog of built-in sample satellite imagery for instant demonstration."""
    samples = [
        {
            "id": "sample_agriculture",
            "name": "Agricultural Delta",
            "filename": "agriculture_delta.jpg",
            "category": "Agriculture & Water",
            "description": "Intensive agricultural crop fields, irrigation canals, and rural settlements.",
            "url": "/samples/agriculture_delta.jpg",
            "has_georeference": True,
            "latitude": 16.5062,
            "longitude": 80.6480  # Krishna River Delta, Andhra Pradesh
        },
        {
            "id": "sample_coastal",
            "name": "Coastal Harbor & Port",
            "filename": "coastal_port.jpg",
            "category": "Maritime & Coastal",
            "description": "Deep-water maritime coastline, cargo shipping docks, and coastal greenery.",
            "url": "/samples/coastal_port.jpg",
            "has_georeference": True,
            "latitude": 17.6868,
            "longitude": 83.2185  # Visakhapatnam Port
        },
        {
            "id": "sample_urban",
            "name": "Urban Metropolis Grid",
            "filename": "urban_metropolis.jpg",
            "category": "Urban & Infrastructure",
            "description": "High-density concrete city blocks, arterial road grid, and central park lake.",
            "url": "/samples/urban_metropolis.jpg",
            "has_georeference": True,
            "latitude": 12.9716,
            "longitude": 77.5946  # Bengaluru
        },
        {
            "id": "sample_geotiff",
            "name": "ISRO SDSC Sriharikota (GeoTIFF)",
            "filename": "isro_sriharikota_geotiff.tif",
            "category": "Spaceport / GeoTIFF",
            "description": "Satish Dhawan Space Centre barrier island with embedded WGS84 GeoTIFF tags.",
            "url": "/samples/isro_sriharikota_geotiff.tif",
            "has_georeference": True,
            "latitude": 13.7200,
            "longitude": 80.2300  # Sriharikota, ISRO
        },
        {
            "id": "sample_temporal_t1",
            "name": "Temporal T1 (Pre-Development)",
            "filename": "temporal_t1.jpg",
            "category": "Temporal Pair",
            "description": "Baseline forest and natural river estuary before industrial expansion.",
            "url": "/samples/temporal_t1.jpg",
            "has_georeference": True,
            "latitude": 21.1458,
            "longitude": 79.0882
        },
        {
            "id": "sample_temporal_t2",
            "name": "Temporal T2 (Post-Development)",
            "filename": "temporal_t2.jpg",
            "category": "Temporal Pair",
            "description": "Follow-up satellite pass showing highway construction and industrial clearing.",
            "url": "/samples/temporal_t2.jpg",
            "has_georeference": True,
            "latitude": 21.1458,
            "longitude": 79.0882
        }
    ]
    return samples


@router.post("/sample/load", response_model=ImageUploadResponse)
def load_sample_image(
    sample_filename: str = Form(...),
    session_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Loads a pre-generated sample satellite image directly into active session."""
    src_path = os.path.join(SAMPLES_DIR, sample_filename)
    if not os.path.exists(src_path):
        raise HTTPException(status_code=404, detail="Sample image not found on server.")

    if not session_id or session_id.strip() == "":
        session_id = f"sess_{uuid.uuid4().hex[:10]}"
        new_session = SessionRecord(id=session_id, title=f"Sample: {sample_filename}")
        db.add(new_session)
        db.commit()

    safe_name = get_safe_filename(sample_filename)
    dst_path = os.path.join(UPLOAD_DIR, safe_name)
    shutil.copyfile(src_path, dst_path)

    metadata = extract_image_metadata(dst_path)

    # Attach coordinates for known samples if not embedded in metadata
    if sample_filename == "agriculture_delta.jpg":
        metadata.update({"has_georeference": True, "latitude": 16.5062, "longitude": 80.6480, "crs": "EPSG:4326"})
    elif sample_filename == "coastal_port.jpg":
        metadata.update({"has_georeference": True, "latitude": 17.6868, "longitude": 83.2185, "crs": "EPSG:4326"})
    elif sample_filename == "urban_metropolis.jpg":
        metadata.update({"has_georeference": True, "latitude": 12.9716, "longitude": 77.5946, "crs": "EPSG:4326"})
    elif "temporal" in sample_filename:
        metadata.update({"has_georeference": True, "latitude": 21.1458, "longitude": 79.0882, "crs": "EPSG:4326"})

    img_record = ImageRecord(
        session_id=session_id,
        filename=safe_name,
        original_name=sample_filename,
        path=dst_path,
        image_type=metadata.get("format", "RGB"),
        width=metadata.get("width", 0),
        height=metadata.get("height", 0),
        file_size=metadata.get("file_size", 0),
        metadata_json=json.dumps(metadata)
    )
    db.add(img_record)
    db.commit()
    db.refresh(img_record)

    meta_schema = ImageMetadataSchema(
        width=metadata["width"],
        height=metadata["height"],
        channels=metadata["channels"],
        color_mode=metadata["color_mode"],
        file_size=metadata["file_size"],
        format=metadata["format"],
        has_georeference=metadata["has_georeference"],
        latitude=metadata["latitude"],
        longitude=metadata["longitude"],
        crs=metadata["crs"],
        bounding_box=metadata["bounding_box"],
        extra_tags=metadata.get("extra_tags")
    )

    return ImageUploadResponse(
        success=True,
        image_id=img_record.id,
        session_id=session_id,
        filename=safe_name,
        original_name=sample_filename,
        url=f"/uploads/{safe_name}",
        metadata=meta_schema,
        message=f"Sample '{sample_filename}' loaded successfully into session."
    )
