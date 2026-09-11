import os
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from backend.database import get_db
from backend.models import SessionRecord, ImageRecord, QueryRecord, AnalysisResultRecord
from backend.schemas import SessionDetailResponse

router = APIRouter(prefix="/api", tags=["History & Sessions"])


@router.get("/history")
def get_analysis_history(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """Retrieves list of past analysis sessions with summary metadata."""
    sessions = db.query(SessionRecord).order_by(SessionRecord.updated_at.desc()).all()
    history = []

    for s in sessions:
        first_img = db.query(ImageRecord).filter(ImageRecord.session_id == s.id).first()
        query_count = db.query(QueryRecord).filter(QueryRecord.session_id == s.id).count()
        img_count = db.query(ImageRecord).filter(ImageRecord.session_id == s.id).count()

        history.append({
            "session_id": s.id,
            "title": s.title or f"Session {s.id[:8]}",
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "updated_at": s.updated_at.isoformat() if s.updated_at else None,
            "image_count": img_count,
            "query_count": query_count,
            "thumbnail_url": f"/uploads/{first_img.filename}" if first_img else None,
            "primary_image_id": first_img.id if first_img else None
        })

    return history


@router.get("/session/{session_id}", response_model=SessionDetailResponse)
def get_session_detail(session_id: str, db: Session = Depends(get_db)):
    """Fetches full details, images, and query transcript of a specific session."""
    session = db.query(SessionRecord).filter(SessionRecord.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    images = db.query(ImageRecord).filter(ImageRecord.session_id == session_id).all()
    queries = db.query(QueryRecord).filter(QueryRecord.session_id == session_id).order_by(QueryRecord.created_at.asc()).all()

    img_list = []
    for img in images:
        img_list.append({
            "id": img.id,
            "filename": img.filename,
            "original_name": img.original_name,
            "url": f"/uploads/{img.filename}",
            "width": img.width,
            "height": img.height,
            "file_size": img.file_size
        })

    query_list = []
    for q in queries:
        ans = db.query(AnalysisResultRecord).filter(AnalysisResultRecord.query_id == q.id).first()
        query_list.append({
            "id": q.id,
            "query": q.query_text,
            "intent": q.intent,
            "answer": ans.answer if ans else "Processing...",
            "created_at": q.created_at.isoformat() if q.created_at else None
        })

    return SessionDetailResponse(
        session_id=session.id,
        title=session.title,
        created_at=session.created_at,
        image_count=len(images),
        query_count=len(queries),
        images=img_list,
        recent_queries=query_list
    )


@router.delete("/session/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db)):
    """Deletes an analysis session and associated records."""
    session = db.query(SessionRecord).filter(SessionRecord.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    db.delete(session)
    db.commit()
    return {"success": True, "message": f"Session {session_id} removed successfully."}


@router.get("/image/{image_id}")
def get_image_file(image_id: int, db: Session = Depends(get_db)):
    """Retrieves raw or uploaded image file by ID."""
    img = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    if not img or not os.path.exists(img.path):
        raise HTTPException(status_code=404, detail="Image not found on server.")

    return FileResponse(img.path)
