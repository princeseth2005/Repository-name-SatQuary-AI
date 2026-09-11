import os
import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import ImageRecord, QueryRecord, AnalysisResultRecord, DetectedRegionRecord
from backend.schemas import QueryRequest, AnalysisResponse, LandCoverStatistics, MaskUrls, DetectedRegionSchema
from backend.services.query_engine import classify_query_intent
from backend.services.vision_engine import run_full_vision_pipeline
from backend.services.vlm_engine import get_vlm_engine

router = APIRouter(prefix="/api", tags=["Natural Language Query"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


@router.post("/query", response_model=AnalysisResponse)
def query_satellite_image(
    req: QueryRequest,
    db: Session = Depends(get_db)
):
    """
    Processes natural-language questions about satellite imagery.
    Classifies intent, executes relevant computer vision modules,
    and returns rich multimodal reasoning with statistics and masks.
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    img_record = db.query(ImageRecord).filter(ImageRecord.id == req.image_id).first()
    if not img_record:
        raise HTTPException(status_code=404, detail=f"Satellite image with ID {req.image_id} not found.")

    if not os.path.exists(img_record.path):
        raise HTTPException(status_code=404, detail="Underlying image file could not be accessed.")

    session_id = req.session_id or img_record.session_id or f"sess_q_{req.image_id}"

    # 1. Classify Intent
    intent, intent_conf = classify_query_intent(req.query)

    # 2. Run or retrieve existing CV analysis
    latest_analysis = (
        db.query(AnalysisResultRecord)
        .filter(AnalysisResultRecord.image_id == img_record.id)
        .order_by(AnalysisResultRecord.created_at.desc())
        .first()
    )

    if latest_analysis and latest_analysis.analysis_json:
        try:
            cv_results = json.loads(latest_analysis.analysis_json)
        except Exception:
            cv_results = run_full_vision_pipeline(img_record.path, OUTPUT_DIR, session_id)
    else:
        cv_results = run_full_vision_pipeline(img_record.path, OUTPUT_DIR, session_id)

    # 3. Vision-Language Reasoning Generation
    vlm = get_vlm_engine()
    ai_answer = vlm.generate_answer(cv_results, req.query, intent)

    # 4. Save Query & Result in Database
    query_record = QueryRecord(
        session_id=session_id,
        image_id=img_record.id,
        query_text=req.query,
        intent=intent
    )
    db.add(query_record)
    db.commit()
    db.refresh(query_record)

    stats = cv_results.get("statistics", {})
    mask_paths = cv_results.get("mask_paths", {})

    result_record = AnalysisResultRecord(
        query_id=query_record.id,
        image_id=img_record.id,
        answer=ai_answer,
        vegetation_percentage=stats.get("vegetation", 0.0),
        water_percentage=stats.get("water", 0.0),
        builtup_percentage=stats.get("built_up", 0.0),
        barren_percentage=stats.get("barren", 0.0),
        masks_json=json.dumps({k: f"/outputs/{os.path.basename(v)}" for k, v in mask_paths.items()}),
        analysis_json=json.dumps(cv_results)
    )
    db.add(result_record)
    db.commit()
    db.refresh(result_record)

    mask_urls = MaskUrls(
        vegetation=f"/outputs/{os.path.basename(mask_paths['vegetation'])}" if 'vegetation' in mask_paths else None,
        water=f"/outputs/{os.path.basename(mask_paths['water'])}" if 'water' in mask_paths else None,
        built_up=f"/outputs/{os.path.basename(mask_paths['built_up'])}" if 'built_up' in mask_paths else None,
        roads=f"/outputs/{os.path.basename(mask_paths['roads'])}" if 'roads' in mask_paths else None,
        segmentation=f"/outputs/{os.path.basename(mask_paths['segmentation'])}" if 'segmentation' in mask_paths else None
    )

    detected_schemas = []
    for r in cv_results.get("detected_regions", []):
        detected_schemas.append(DetectedRegionSchema(
            class_name=r["class_name"],
            confidence=r["confidence"],
            bbox=r.get("bbox"),
            area_percentage=r["area_percentage"]
        ))

    return AnalysisResponse(
        success=True,
        query_id=query_record.id,
        image_id=img_record.id,
        intent=intent,
        answer=ai_answer,
        statistics=LandCoverStatistics(**stats),
        masks=mask_urls,
        detected_regions=detected_schemas,
        spatial_summary=cv_results.get("spatial_summary", {})
    )
