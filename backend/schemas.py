from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import datetime


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "SatQuery AI"
    version: str = "1.0.0"
    isro_theme: str = "SIH26167 Space Technology"
    vlm_provider: str = "local"


class ImageMetadataSchema(BaseModel):
    width: int
    height: int
    channels: int = 3
    color_mode: str = "RGB"
    file_size: int
    format: str
    has_georeference: bool = False
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    crs: Optional[str] = None
    bounding_box: Optional[List[float]] = None  # [min_lat, min_lon, max_lat, max_lon]
    extra_tags: Optional[Dict[str, Any]] = None


class ImageUploadResponse(BaseModel):
    success: bool
    image_id: int
    session_id: str
    filename: str
    original_name: str
    url: str
    metadata: ImageMetadataSchema
    message: str = "Image uploaded successfully"


class LandCoverStatistics(BaseModel):
    vegetation: float = Field(..., description="Estimated percentage of vegetation")
    water: float = Field(..., description="Estimated percentage of water bodies")
    built_up: float = Field(..., description="Estimated percentage of built-up / urban")
    barren: float = Field(..., description="Estimated percentage of barren/soil/other")
    roads_linear: Optional[float] = Field(0.0, description="Estimated linear feature density")
    confidence: float = Field(0.88, description="Model confidence score")


class MaskUrls(BaseModel):
    vegetation: Optional[str] = None
    water: Optional[str] = None
    built_up: Optional[str] = None
    segmentation: Optional[str] = None
    roads: Optional[str] = None
    change: Optional[str] = None


class DetectedRegionSchema(BaseModel):
    class_name: str
    confidence: float
    bbox: Optional[List[float]] = None
    area_percentage: float


class AnalysisResponse(BaseModel):
    success: bool
    query_id: Optional[int] = None
    image_id: int
    intent: str
    answer: str
    statistics: LandCoverStatistics
    masks: MaskUrls
    detected_regions: List[DetectedRegionSchema] = []
    spatial_summary: Dict[str, Any] = {}
    is_scientific_grade: bool = False
    disclaimer: str = "Estimates generated from optical satellite analysis. Not a certified remote sensing product."


class QueryRequest(BaseModel):
    image_id: int
    query: str
    session_id: Optional[str] = None


class CompareRequest(BaseModel):
    image_id_1: int
    image_id_2: int
    session_id: Optional[str] = None
    query: Optional[str] = "Compare these two satellite images and find changes."


class CompareResponse(BaseModel):
    success: bool
    image_id_1: int
    image_id_2: int
    change_percentage: float
    change_mask_url: str
    answer: str
    quadrant_changes: Dict[str, float]
    summary: str
    disclaimer: str = "Image-based visual change detection. Cloud cover, seasonal variations and angle of incidence can affect estimates."


class SessionDetailResponse(BaseModel):
    session_id: str
    title: str
    created_at: datetime.datetime
    image_count: int
    query_count: int
    images: List[Dict[str, Any]] = []
    recent_queries: List[Dict[str, Any]] = []


class SampleItem(BaseModel):
    id: str
    name: str
    filename: str
    category: str
    description: str
    url: str
    has_georeference: bool = False
    latitude: Optional[float] = None
    longitude: Optional[float] = None
