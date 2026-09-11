import os
import json
from fastapi import APIRouter, Depends, HTTPException, Form
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import ImageRecord, AnalysisResultRecord, DetectedRegionRecord
from backend.schemas import AnalysisResponse, LandCoverStatistics, MaskUrls, DetectedRegionSchema
from backend.services.vision_engine import run_full_vision_pipeline

router = APIRouter(prefix="/api", tags=["Analysis"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@router.post("/analyze", response_model=AnalysisResponse)
def analyze_satellite_image(
    image_id: int = Form(...),
    db: Session = Depends(get_db)
):
    """
    Executes comprehensive optical remote sensing pipeline:
    extracts vegetation, water, built-up, linear roads, and barren matrices,
    generating transparent overlay masks and classification statistics.
    """
    img_record = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    if not img_record:
        raise HTTPException(status_code=404, detail=f"Image with id {image_id} not found.")

    if not os.path.exists(img_record.path):
        raise HTTPException(status_code=404, detail="Image file is missing from server storage.")

    session_id = img_record.session_id or f"sess_img_{image_id}"

    # Run CV pipeline
    try:
        results = run_full_vision_pipeline(img_record.path, OUTPUT_DIR, session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision pipeline processing error: {str(e)}")

    stats = results["statistics"]
    mask_paths = results["mask_paths"]

    # Generate public URLs for masks
    mask_urls = MaskUrls(
        vegetation=f"/outputs/{os.path.basename(mask_paths['vegetation'])}",
        water=f"/outputs/{os.path.basename(mask_paths['water'])}",
        built_up=f"/outputs/{os.path.basename(mask_paths['built_up'])}",
        roads=f"/outputs/{os.path.basename(mask_paths['roads'])}",
        segmentation=f"/outputs/{os.path.basename(mask_paths['segmentation'])}"
    )

    # Executive answer
    dominant = results["dominant_class"]
    dom_pct = results["dominant_percentage"]
    exec_answer = (
        f"Automated scene analysis complete. The scene is dominated by **{dominant.lower()}** ({dom_pct}%). "
        f"Vegetation canopy constitutes **{stats['vegetation']}%**, surface water bodies cover **{stats['water']}%**, "
        f"and urban built-up infrastructure spans **{stats['built_up']}%**. "
        f"Linear transport corridors comprise **{stats['roads_linear']}%**."
    )

    # Save to database
    analysis_record = AnalysisResultRecord(
        image_id=img_record.id,
        answer=exec_answer,
        vegetation_percentage=stats["vegetation"],
        water_percentage=stats["water"],
        builtup_percentage=stats["built_up"],
        barren_percentage=stats["barren"],
        masks_json=json.dumps({k: f"/outputs/{os.path.basename(v)}" for k, v in mask_paths.items()}),
        analysis_json=json.dumps(results)
    )
    db.add(analysis_record)
    db.commit()
    db.refresh(analysis_record)

    # Save detected regions
    detected_schemas = []
    for r in results.get("detected_regions", []):
        d_rec = DetectedRegionRecord(
            analysis_id=analysis_record.id,
            class_name=r["class_name"],
            confidence=r["confidence"],
            bbox_json=json.dumps(r.get("bbox")),
            area_percentage=r["area_percentage"]
        )
        db.add(d_rec)
        detected_schemas.append(DetectedRegionSchema(
            class_name=r["class_name"],
            confidence=r["confidence"],
            bbox=r.get("bbox"),
            area_percentage=r["area_percentage"]
        ))
    db.commit()

    return AnalysisResponse(
        success=True,
        query_id=None,
        image_id=img_record.id,
        intent="GENERAL_DESCRIPTION",
        answer=exec_answer,
        statistics=LandCoverStatistics(**stats),
        masks=mask_urls,
        detected_regions=detected_schemas,
        spatial_summary=results.get("spatial_summary", {})
    )
