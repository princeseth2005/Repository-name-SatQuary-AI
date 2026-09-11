import os
import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import ImageRecord, AnalysisResultRecord, QueryRecord
from backend.schemas import CompareRequest, CompareResponse
from backend.services.change_detection import analyze_change

router = APIRouter(prefix="/api", tags=["Temporal Change Detection"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


@router.post("/compare", response_model=CompareResponse)
def compare_satellite_images(
    req: CompareRequest,
    db: Session = Depends(get_db)
):
    """
    Executes automated dual-image change detection pipeline:
    aligns images, calculates perceptual CIE-Lab and structural differences,
    isolates changed sectors, and generates natural-language explanation.
    """
    img1 = db.query(ImageRecord).filter(ImageRecord.id == req.image_id_1).first()
    img2 = db.query(ImageRecord).filter(ImageRecord.id == req.image_id_2).first()

    if not img1 or not img2:
        raise HTTPException(status_code=404, detail="One or both comparison images were not found.")

    if not os.path.exists(img1.path) or not os.path.exists(img2.path):
        raise HTTPException(status_code=404, detail="Source image files for comparison are missing from server.")

    session_id = req.session_id or img1.session_id or f"sess_comp_{img1.id}_{img2.id}"
    prefix = f"change_{img1.id}_{img2.id}"

    try:
        ch_pct, mask_path, side_path, details, features, summary_nl = analyze_change(
            img1.path, img2.path, OUTPUT_DIR, prefix=prefix
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Change detection error: {str(e)}")

    # Store query record
    q_rec = QueryRecord(
        session_id=session_id,
        image_id=img1.id,
        query_text=req.query or "Compare images and detect changes",
        intent="CHANGE_DETECTION"
    )
    db.add(q_rec)
    db.commit()

    # Store analysis record
    ar_rec = AnalysisResultRecord(
        query_id=q_rec.id,
        image_id=img1.id,
        answer=summary_nl,
        change_percentage=ch_pct,
        masks_json=json.dumps({
            "change": f"/outputs/{os.path.basename(mask_path)}",
            "side_by_side": f"/outputs/{os.path.basename(side_path)}"
        }),
        analysis_json=json.dumps(details)
    )
    db.add(ar_rec)
    db.commit()

    return CompareResponse(
        success=True,
        image_id_1=img1.id,
        image_id_2=img2.id,
        change_percentage=ch_pct,
        change_mask_url=f"/outputs/{os.path.basename(mask_path)}",
        answer=summary_nl,
        quadrant_changes=details.get("quadrant_changes", {}),
        summary=summary_nl
    )
