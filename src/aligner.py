import cv2
import numpy as np
from typing import Tuple

class FaceAligner:
    def __init__(self):
        """Initialize face aligner."""
        self.reference_points = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]
        ], dtype=np.float32)
    
    def align_face(self, image: np.ndarray, landmarks: np.ndarray, 
                   output_size: int = 112) -> Tuple[np.ndarray, float]:
        """
        Align face using 5-point landmarks.
        
        Args:
            image: BGR image
            landmarks: 5 facial landmarks (5×2 array)
            output_size: Output face size (default 112×112 for ArcFace)
        
        Returns:
            Tuple of (aligned_face, quality_score)
        """
        try:
            landmarks = np.array(landmarks, dtype=np.float32)
            
            if landmarks.shape != (5, 2):
                return None, 0.0
            
            # Compute affine transformation matrix
            M = cv2.estimateAffinePartial2D(
                landmarks, 
                self.reference_points,
                method=cv2.LMEDS
            )[0]
            
            if M is None:
                return None, 0.0
            
            # Apply affine transformation
            aligned_face = cv2.warpAffine(
                image, M, 
                (output_size, output_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )
            
            # Calculate alignment quality (how well landmarks match reference)
            transformed_landmarks = cv2.transform(
                landmarks.reshape(1, -1, 2), M
            ).reshape(5, 2)
            
            error = np.mean(np.linalg.norm(
                transformed_landmarks - self.reference_points, 
                axis=1
            ))
            
            # Quality score: lower error = higher quality
            quality_score = 1.0 / (1.0 + error)
            
            return aligned_face, quality_score
        
        except Exception as e:
            print(f"Alignment error: {e}")
            return None, 0.0
    
    def get_alignment_quality(self, error: float) -> float:
        """
        Convert alignment error to quality score (0-1).
        
        Args:
            error: Alignment error
        
        Returns:
            Quality score (0-1, higher is better)
        """
        return 1.0 / (1.0 + error)