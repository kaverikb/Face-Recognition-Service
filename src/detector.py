import cv2
import numpy as np
from insightface.app import FaceAnalysis
from typing import List, Dict
import time

class FaceDetector:
    def __init__(self, model_name='buffalo_l', ctx_id=-1):
        """
        Initialize face detector with InsightFace.
        
        Args:
            model_name: Model name (buffalo_l is recommended)
            ctx_id: -1 for CPU, 0+ for GPU
        """
        print("Initializing Face Detector...")
        self.app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=ctx_id)
        print("✅ Face Detector initialized")
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image.
        
        Args:
            image: BGR image (numpy array)
        
        Returns:
            List of detections with bounding boxes and landmarks
        """
        start_time = time.time()
        
        # Detect faces
        faces = self.app.get(image)
        
        detections = []
        for face in faces:
            # Extract bounding box
            bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
            
            # Extract landmarks (5-point: eyes, nose, mouth corners)
            landmarks = face.kps  # 5 points × 2 (x, y)
            
            detection = {
                'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                'landmarks': landmarks.tolist() if landmarks is not None else [],
                'confidence': float(face.det_score),
                'face_data': face  # Store original face object for embedding extraction
            }
            detections.append(detection)
        
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            'detections': detections,
            'num_faces': len(detections),
            'latency_ms': latency
        }
    
    def detect_and_crop(self, image: np.ndarray, min_size: int = 80) -> List[Dict]:
        """
        Detect faces and crop face regions.
        
        Args:
            image: BGR image
            min_size: Minimum face size (pixels)
        
        Returns:
            List of detections with cropped face images
        """
        result = self.detect(image)
        detections = result['detections']
        
        cropped_faces = []
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = [int(x) for x in detection['bbox']]
            
            # Check minimum size
            width = x2 - x1
            height = y2 - y1
            
            if width < min_size or height < min_size:
                continue
            
            # Crop face region
            face_image = image[y1:y2, x1:x2]
            
            detection['face_image'] = face_image
            detection['face_id'] = i
            cropped_faces.append(detection)
        
        return {
            'detections': cropped_faces,
            'num_valid_faces': len(cropped_faces),
            'latency_ms': result['latency_ms']
        }
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict], 
                       show_confidence: bool = True) -> np.ndarray:
        """
        Draw bounding boxes and landmarks on image.
        
        Args:
            image: BGR image
            detections: List of detections
            show_confidence: Show confidence scores
        
        Returns:
            Image with drawn detections
        """
        result_image = image.copy()
        
        for detection in detections:
            # Draw bounding box
            x1, y1, x2, y2 = [int(x) for x in detection['bbox']]
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw landmarks (5 points)
            if detection['landmarks']:
                landmarks = detection['landmarks']
                for point in landmarks:
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(result_image, (x, y), 3, (0, 0, 255), -1)
            
            # Draw confidence score
            if show_confidence:
                conf = detection['confidence']
                text = f"Conf: {conf:.3f}"
                cv2.putText(result_image, text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return result_image


# Global detector instance
_detector = None

def get_detector():
    """Get or create global detector instance."""
    global _detector
    if _detector is None:
        _detector = FaceDetector()
    return _detector


def detect_faces(image: np.ndarray) -> Dict:
    """
    Convenience function to detect faces.
    
    Args:
        image: BGR image
    
    Returns:
        Detection results
    """
    detector = get_detector()
    return detector.detect(image)


def detect_and_crop_faces(image: np.ndarray) -> Dict:
    """
    Convenience function to detect and crop faces.
    
    Args:
        image: BGR image
    
    Returns:
        Detection results with cropped images
    """
    detector = get_detector()
    return detector.detect_and_crop(image)