from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class DetectionBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

class DetectionResult(BaseModel):
    face_id: int
    bbox: DetectionBox
    landmarks: Optional[List[List[float]]] = None
    quality_score: float

class RecognitionMatch(BaseModel):
    name: str
    confidence: float
    identity_id: int

class RecognitionResult(BaseModel):
    face_id: int
    bbox: DetectionBox
    matched_identity: Optional[RecognitionMatch] = None
    top_5_matches: List[RecognitionMatch]
    processing_time_ms: float

class Identity(BaseModel):
    id: int
    name: str
    num_embeddings: int
    created_at: datetime
    is_active: bool

class AddIdentityRequest(BaseModel):
    name: str

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime

class StatsResponse(BaseModel):
    total_identities: int
    total_embeddings: int
    total_recognitions: int
    avg_latency_ms: float