from src.database import SessionLocal, IdentityModel, EmbeddingModel, RecognitionLogModel
from sqlalchemy import func
from typing import List, Dict

def get_identity_by_name(name: str) -> IdentityModel:
    """Get identity by name."""
    db = SessionLocal()
    try:
        return db.query(IdentityModel).filter(IdentityModel.name == name).first()
    finally:
        db.close()

def get_identity_by_id(identity_id: int) -> IdentityModel:
    """Get identity by ID."""
    db = SessionLocal()
    try:
        return db.query(IdentityModel).filter(IdentityModel.id == identity_id).first()
    finally:
        db.close()

def get_all_identities(active_only: bool = True) -> List[IdentityModel]:
    """Get all identities."""
    db = SessionLocal()
    try:
        query = db.query(IdentityModel)
        if active_only:
            query = query.filter(IdentityModel.is_active == True)
        return query.all()
    finally:
        db.close()

def get_embeddings_by_identity(identity_id: int) -> List[EmbeddingModel]:
    """Get all embeddings for an identity."""
    db = SessionLocal()
    try:
        return db.query(EmbeddingModel).filter(
            EmbeddingModel.identity_id == identity_id
        ).all()
    finally:
        db.close()

def get_recognition_stats() -> Dict:
    """Get recognition statistics."""
    db = SessionLocal()
    try:
        total_recognitions = db.query(RecognitionLogModel).count()
        total_identities = db.query(IdentityModel).count()
        total_embeddings = db.query(EmbeddingModel).count()
        
        avg_latency = db.query(func.avg(RecognitionLogModel.processing_time_ms)).scalar() or 0
        
        return {
            'total_recognitions': total_recognitions,
            'total_identities': total_identities,
            'total_embeddings': total_embeddings,
            'avg_latency_ms': float(avg_latency)
        }
    finally:
        db.close()

def get_hard_to_recognize_identities() -> List[Dict]:
    """
    Get identities with lowest average match confidence.
    These are harder to recognize.
    """
    db = SessionLocal()
    try:
        logs = db.query(RecognitionLogModel).filter(
            RecognitionLogModel.matched_identity_id != None
        ).all()
        
        identity_scores = {}
        for log in logs:
            if log.matched_identity_id not in identity_scores:
                identity_scores[log.matched_identity_id] = []
            if log.matched_confidence:
                identity_scores[log.matched_identity_id].append(log.matched_confidence)
        
        # Calculate average confidence per identity
        results = []
        for identity_id, scores in identity_scores.items():
            avg_confidence = sum(scores) / len(scores)
            identity = db.query(IdentityModel).filter(IdentityModel.id == identity_id).first()
            results.append({
                'identity_id': identity_id,
                'identity_name': identity.name if identity else 'Unknown',
                'avg_confidence': avg_confidence,
                'num_recognitions': len(scores)
            })
        
        # Sort by average confidence (lowest first)
        results.sort(key=lambda x: x['avg_confidence'])
        return results
    finally:
        db.close()

def get_false_positive_rate() -> float:
    """
    Calculate percentage of detections that didn't match anything.
    """
    db = SessionLocal()
    try:
        total_detections = db.query(RecognitionLogModel).count()
        if total_detections == 0:
            return 0.0
        
        unmatched = db.query(RecognitionLogModel).filter(
            RecognitionLogModel.matched_identity_id == None
        ).count()
        
        return (unmatched / total_detections) * 100
    finally:
        db.close()