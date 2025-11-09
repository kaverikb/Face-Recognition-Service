import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import time
from datetime import datetime

from src.schemas import (
    DetectionResult, RecognitionResult, Identity, 
    HealthResponse, StatsResponse, RecognitionMatch
)
from src.detector import get_detector
from src.embedder import get_embedder
from src.matcher import get_matcher
from src.postprocessing import (
    apply_nms, filter_by_blur, filter_by_confidence, filter_by_size
)
from src.database import SessionLocal, IdentityModel, EmbeddingModel, RecognitionLogModel, AuditLogModel
from src.config import (
    DETECTION_CONFIDENCE_THRESHOLD, NMS_THRESHOLD, MIN_FACE_SIZE,
    BLUR_THRESHOLD, RECOGNITION_THRESHOLD, TOP_K_MATCHES
)

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition Service",
    description="End-to-end face detection and recognition system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for statistics
stats = {
    'total_detections': 0,
    'total_recognitions': 0,
    'total_latency': 0
}

# ====================== UTILITY FUNCTIONS ======================

def read_image_from_upload(file: UploadFile) -> np.ndarray:
    """Convert uploaded file to BGR numpy array."""
    try:
        contents = file.file.read()
        pil_image = Image.open(BytesIO(contents))
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return cv_image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

def log_recognition(detected_count: int, matched_identity_id: int = None, 
                   matched_confidence: float = None, top_5_matches: list = None,
                   processing_time: float = 0):
    """Log recognition event to database."""
    db = SessionLocal()
    try:
        log = RecognitionLogModel(
            detected_face_count=detected_count,
            matched_identity_id=matched_identity_id,
            matched_confidence=matched_confidence,
            top_5_matches=top_5_matches,
            processing_time_ms=int(processing_time)
        )
        db.add(log)
        db.commit()
    except Exception as e:
        print(f"Error logging recognition: {e}")
    finally:
        db.close()

def log_audit(action: str, identity_id: int = None, details: dict = None):
    """Log audit event to database."""
    db = SessionLocal()
    try:
        log = AuditLogModel(
            action=action,
            identity_id=identity_id,
            details=details
        )
        db.add(log)
        db.commit()
    except Exception as e:
        print(f"Error logging audit: {e}")
    finally:
        db.close()

# ====================== ENDPOINTS ======================

@app.get("/health")
async def health():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/stats")
async def get_stats():
    """
    Get system statistics.
    """
    db = SessionLocal()
    try:
        total_identities = db.query(IdentityModel).count()
        total_embeddings = db.query(EmbeddingModel).count()
        total_recognitions = db.query(RecognitionLogModel).count()
        
        avg_latency = 0
        if total_recognitions > 0:
            logs = db.query(RecognitionLogModel).all()
            total_latency = sum([log.processing_time_ms for log in logs])
            avg_latency = total_latency / total_recognitions
        
        return {
            "total_identities": total_identities,
            "total_embeddings": total_embeddings,
            "total_recognitions": total_recognitions,
            "avg_latency_ms": round(avg_latency, 2),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(DETECTION_CONFIDENCE_THRESHOLD, ge=0, le=1),
    nms_threshold: float = Query(NMS_THRESHOLD, ge=0, le=1),
    min_face_size: int = Query(MIN_FACE_SIZE, ge=10)
):
    """
    Detect faces in an image.
    
    **Parameters:**
    - file: Image file (JPG/PNG)
    - confidence_threshold: Minimum detection confidence (0-1)
    - nms_threshold: Non-Maximum Suppression threshold
    - min_face_size: Minimum face size in pixels
    
    **Returns:**
    - List of detected faces with bounding boxes and landmarks
    """
    start_time = time.time()
    
    try:
        # Read image
        image = read_image_from_upload(file)
        
        # Detect faces
        detector = get_detector()
        result = detector.detect(image)
        detections = result['detections']
        
        # Apply post-processing filters
        detections = filter_by_confidence(detections, confidence_threshold)
        detections = filter_by_size(detections, min_face_size)
        detections = apply_nms(detections, nms_threshold)
        
        latency = (time.time() - start_time) * 1000
        
        return {
            "num_faces": len(detections),
            "detections": detections,
            "processing_time_ms": round(latency, 2),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/recognize")
async def recognize(
    file: UploadFile = File(...),
    threshold: float = Query(RECOGNITION_THRESHOLD, ge=0, le=1),
    top_k: int = Query(TOP_K_MATCHES, ge=1, le=20),
    confidence_threshold: float = Query(DETECTION_CONFIDENCE_THRESHOLD, ge=0, le=1)
):
    """
    Detect and recognize faces in an image.
    
    **Parameters:**
    - file: Image file (JPG/PNG)
    - threshold: Similarity threshold for matching (0-1)
    - top_k: Number of top matches to return
    - confidence_threshold: Minimum detection confidence
    
    **Returns:**
    - List of recognized faces with matched identities and confidence scores
    """
    start_time = time.time()
    
    try:
        # Read image
        image = read_image_from_upload(file)
        
        # Detect faces
        detector = get_detector()
        detection_result = detector.detect(image)
        detections = detection_result['detections']
        
        # Filter detections
        detections = filter_by_confidence(detections, confidence_threshold)
        detections = filter_by_size(detections, MIN_FACE_SIZE)
        detections = apply_nms(detections, NMS_THRESHOLD)
        
        # Extract embeddings and match
        matcher = get_matcher(threshold=threshold, top_k=top_k)
        embedder = get_embedder()
        
        recognition_results = []
        best_match_identity = None
        best_match_confidence = 0
        
        for i, detection in enumerate(detections):
            face_data = detection['face_data']
            
            # Extract embedding
            emb_result = embedder.extract_embedding(image, face_data)
            if not emb_result['success']:
                continue
            
            embedding = emb_result['embedding']
            
            # Match embedding
            match_result = matcher.match_single(embedding, threshold=threshold)
            
            # Format response
            rec_result = {
                "face_id": i,
                "bbox": detection['bbox'],
                "confidence": float(detection['confidence']),
                "landmarks": detection['landmarks'],
                "matched_identity": None,
                "top_matches": []
            }
            
            # Add matched identity if found
            if match_result['is_match'] and match_result['best_match']:
                rec_result["matched_identity"] = {
                    "identity_id": match_result['best_match']['identity_id'],
                    "name": match_result['best_match']['identity_name'],
                    "confidence": match_result['best_match']['confidence']
                }
                
                # Track best match
                if match_result['best_match']['confidence'] > best_match_confidence:
                    best_match_identity = match_result['best_match']['identity_id']
                    best_match_confidence = match_result['best_match']['confidence']
            
            # Add top K matches
            for match in match_result['matches'][:top_k]:
                rec_result["top_matches"].append({
                    "identity_id": match['identity_id'],
                    "name": match['identity_name'],
                    "confidence": match['similarity']
                })
            
            recognition_results.append(rec_result)
        
        latency = (time.time() - start_time) * 1000
        
        # Log recognition event
        log_recognition(
            detected_count=len(recognition_results),
            matched_identity_id=best_match_identity,
            matched_confidence=best_match_confidence,
            processing_time=latency
        )
        
        return {
            "num_faces": len(recognition_results),
            "results": recognition_results,
            "processing_time_ms": round(latency, 2),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")

@app.get("/list_identities")
async def list_identities():
    """
    List all identities in the gallery.
    
    **Returns:**
    - List of all identities with their embedding counts
    """
    db = SessionLocal()
    try:
        identities = db.query(IdentityModel).filter(IdentityModel.is_active == True).all()
        
        result = []
        for identity in identities:
            num_embeddings = db.query(EmbeddingModel).filter(
                EmbeddingModel.identity_id == identity.id
            ).count()
            
            result.append({
                "id": identity.id,
                "name": identity.name,
                "num_embeddings": num_embeddings,
                "created_at": identity.created_at.isoformat(),
                "is_active": identity.is_active
            })
        
        return {
            "total_identities": len(result),
            "identities": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.delete("/delete_identity/{identity_id}")
async def delete_identity(identity_id: int):
    """
    Delete an identity from the gallery.
    
    **Parameters:**
    - identity_id: ID of identity to delete
    
    **Returns:**
    - Confirmation message
    """
    db = SessionLocal()
    try:
        identity = db.query(IdentityModel).filter(
            IdentityModel.id == identity_id
        ).first()
        
        if not identity:
            raise HTTPException(status_code=404, detail=f"Identity {identity_id} not found")
        
        # Delete embeddings
        db.query(EmbeddingModel).filter(
            EmbeddingModel.identity_id == identity_id
        ).delete()
        
        # Delete identity
        db.delete(identity)
        db.commit()
        
        log_audit("DELETE_IDENTITY", identity_id=identity_id)
        
        return {
            "message": f"Identity '{identity.name}' deleted successfully",
            "identity_id": identity_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.post("/batch_recognize")
async def batch_recognize(
    files: list = File(...),
    threshold: float = Query(RECOGNITION_THRESHOLD, ge=0, le=1),
    top_k: int = Query(TOP_K_MATCHES, ge=1, le=20)
):
    """
    Recognize faces in multiple images at once.
    
    **Parameters:**
    - files: List of image files
    - threshold: Similarity threshold
    - top_k: Number of top matches per face
    
    **Returns:**
    - Recognition results for each image
    """
    start_time = time.time()
    results = []
    
    for file in files:
        try:
            image = read_image_from_upload(file)
            
            detector = get_detector()
            detection_result = detector.detect(image)
            detections = detection_result['detections']
            
            detections = filter_by_confidence(detections, DETECTION_CONFIDENCE_THRESHOLD)
            detections = filter_by_size(detections, MIN_FACE_SIZE)
            detections = apply_nms(detections, NMS_THRESHOLD)
            
            matcher = get_matcher(threshold=threshold, top_k=top_k)
            embedder = get_embedder()
            
            file_results = []
            for i, detection in enumerate(detections):
                face_data = detection['face_data']
                emb_result = embedder.extract_embedding(image, face_data)
                
                if not emb_result['success']:
                    continue
                
                embedding = emb_result['embedding']
                match_result = matcher.match_single(embedding, threshold=threshold)
                
                file_results.append({
                    "face_id": i,
                    "bbox": detection['bbox'],
                    "matched_identity": match_result['best_match'],
                    "confidence": match_result['best_match']['confidence'] if match_result['best_match'] else None
                })
            
            results.append({
                "filename": file.filename,
                "num_faces": len(file_results),
                "results": file_results,
                "status": "success"
            })
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    latency = (time.time() - start_time) * 1000
    
    return {
        "total_files": len(files),
        "successful": sum(1 for r in results if r['status'] == 'success'),
        "results": results,
        "processing_time_ms": round(latency, 2),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "name": "Face Recognition Service",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "stats": "GET /stats",
            "detect": "POST /detect",
            "recognize": "POST /recognize",
            "list_identities": "GET /list_identities",
            "delete_identity": "DELETE /delete_identity/{id}",
            "batch_recognize": "POST /batch_recognize",
            "docs": "GET /docs"
        },
        "docs_url": "/docs"
    }

# ====================== ERROR HANDLERS ======================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP_ERROR",
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": str(exc),
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)