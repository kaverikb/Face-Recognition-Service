import cv2
import numpy as np

def apply_nms(detections, iou_threshold=0.4):
    """
    Apply Non-Maximum Suppression to remove overlapping boxes.
    
    Args:
        detections: List of detection dicts with 'bbox' and 'confidence'
        iou_threshold: IoU threshold for suppression
    
    Returns:
        Filtered detections
    """
    if len(detections) == 0:
        return detections
    
    # Extract boxes and scores
    boxes = np.array([d['bbox'] for d in detections])
    scores = np.array([d['confidence'] for d in detections])
    
    # Convert to format for NMS: [x1, y1, x2, y2]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Calculate areas
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Sort by score descending
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
        
        # Calculate IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Keep boxes with IoU < threshold
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    
    return [detections[i] for i in keep]

def calculate_blur_score(image):
    """
    Calculate Laplacian variance (blur score).
    Higher value = sharper image.
    
    Args:
        image: BGR image
    
    Returns:
        Laplacian variance score
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance

def filter_by_blur(faces, blur_threshold=100):
    """
    Remove faces that are too blurry.
    
    Args:
        faces: List of face dicts with 'image' key
        blur_threshold: Minimum Laplacian variance
    
    Returns:
        Filtered faces with quality scores
    """
    filtered = []
    for face in faces:
        blur_score = calculate_blur_score(face['image'])
        if blur_score >= blur_threshold:
            face['blur_score'] = blur_score
            filtered.append(face)
    return filtered

def filter_by_confidence(detections, confidence_threshold=0.5):
    """
    Filter detections by confidence threshold.
    """
    return [d for d in detections if d['confidence'] >= confidence_threshold]

def filter_by_size(detections, min_size=80):
    """
    Filter detections by minimum face size.
    """
    filtered = []
    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        width = x2 - x1
        height = y2 - y1
        if width >= min_size and height >= min_size:
            filtered.append(d)
    return filtered