from sqlalchemy import create_engine, Column, Integer, String, Float, LargeBinary, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from src.config import DATABASE_URL

# Database setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Models
class IdentityModel(Base):
    __tablename__ = "identities"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class EmbeddingModel(Base):
    __tablename__ = "embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    identity_id = Column(Integer, ForeignKey("identities.id"), nullable=False)
    embedding_vector = Column(LargeBinary, nullable=False)
    image_path = Column(String(500))
    quality_score = Column(Float)
    detection_confidence = Column(Float)
    alignment_quality = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class RecognitionLogModel(Base):
    __tablename__ = "recognition_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    input_image_path = Column(String(500))
    detected_face_count = Column(Integer)
    matched_identity_id = Column(Integer, ForeignKey("identities.id"))
    matched_confidence = Column(Float)
    top_5_matches = Column(JSON)
    processing_time_ms = Column(Integer)

class AuditLogModel(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    action = Column(String(50))
    identity_id = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    details = Column(JSON)

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()