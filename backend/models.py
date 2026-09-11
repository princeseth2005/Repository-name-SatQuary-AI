import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from backend.database import Base


class SessionRecord(Base):
    __tablename__ = "sessions"

    id = Column(String(64), primary_key=True, index=True)
    title = Column(String(255), default="Remote Sensing Session")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    images = relationship("ImageRecord", back_populates="session", cascade="all, delete-orphan")
    queries = relationship("QueryRecord", back_populates="session", cascade="all, delete-orphan")


class ImageRecord(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("sessions.id"), nullable=True)
    filename = Column(String(255), nullable=False)
    original_name = Column(String(255), nullable=False)
    path = Column(String(512), nullable=False)
    image_type = Column(String(50), default="RGB")  # RGB, Multispectral, GeoTIFF
    width = Column(Integer, default=0)
    height = Column(Integer, default=0)
    file_size = Column(Integer, default=0)  # bytes
    upload_time = Column(DateTime, default=datetime.datetime.utcnow)
    metadata_json = Column(Text, nullable=True)  # Coordinates, CRS, EXIF, sensor info

    session = relationship("SessionRecord", back_populates="images")
    queries = relationship("QueryRecord", back_populates="image")
    analysis_results = relationship("AnalysisResultRecord", back_populates="image")


class QueryRecord(Base):
    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("sessions.id"), nullable=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=True)
    query_text = Column(Text, nullable=False)
    intent = Column(String(64), default="GENERAL_DESCRIPTION")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    session = relationship("SessionRecord", back_populates="queries")
    image = relationship("ImageRecord", back_populates="queries")
    analysis_result = relationship("AnalysisResultRecord", back_populates="query", uselist=False)


class AnalysisResultRecord(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    query_id = Column(Integer, ForeignKey("queries.id"), nullable=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=True)
    answer = Column(Text, nullable=False)
    vegetation_percentage = Column(Float, default=0.0)
    water_percentage = Column(Float, default=0.0)
    builtup_percentage = Column(Float, default=0.0)
    barren_percentage = Column(Float, default=0.0)
    change_percentage = Column(Float, nullable=True)
    masks_json = Column(Text, nullable=True)     # Paths to generated transparent masks
    analysis_json = Column(Text, nullable=True)  # Complete detailed statistical breakdown
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    query = relationship("QueryRecord", back_populates="analysis_result")
    image = relationship("ImageRecord", back_populates="analysis_results")
    detected_regions = relationship("DetectedRegionRecord", back_populates="analysis", cascade="all, delete-orphan")


class DetectedRegionRecord(Base):
    __tablename__ = "detected_regions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    analysis_id = Column(Integer, ForeignKey("analysis_results.id"), nullable=False)
    class_name = Column(String(100), nullable=False)
    confidence = Column(Float, default=0.85)
    bbox_json = Column(String(255), nullable=True)  # [ymin, xmin, ymax, xmax] normalized
    area_percentage = Column(Float, default=0.0)

    analysis = relationship("AnalysisResultRecord", back_populates="detected_regions")
