import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

import cv2
import numpy as np
from src.detector import get_detector
from src.embedder import get_embedder
from src.matcher import get_matcher
from src.postprocessing import apply_nms, filter_by_size, filter_by_confidence
import time

print("=" * 70)
print("DEMO VIDEO GENERATION")
print("=" * 70)

# Load models
print("\nLoading models...")
detector = get_detector()
embedder = get_embedder()
matcher = get_matcher()
print("✅ Models loaded")

# Video setup
print("\nSetting up video...")
gallery_path = "data/gallery_images"
output_path = "demo/output_demo.mp4"

# Create demo/ folder if needed
os.makedirs("demo", exist_ok=True)

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 10.0, (640, 480))

print(f"Output video: {output_path}")

# Load test images and create frames
images = sorted(os.listdir(gallery_path))[:10]  # Use first 10 images for demo

print(f"\nProcessing {len(images)} frames...")
print("=" * 70)

for idx, img_name in enumerate(images):
    img_path = os.path.join(gallery_path, img_name)
    frame = cv2.imread(img_path)
    
    if frame is None:
        print(f"❌ Could not load {img_name}")
        continue
    
    # Resize frame to 640x480 for consistent video
    frame = cv2.resize(frame, (640, 480))
    
    # Detect faces
    start_time = time.time()
    det_result = detector.detect(frame)
    detections = det_result['detections']
    
    # Filter detections
    detections = filter_by_confidence(detections, 0.5)
    detections = filter_by_size(detections, 80)
    detections = apply_nms(detections, 0.4)
    
    detection_latency = (time.time() - start_time) * 1000
    
    # Process each detected face
    for face_idx, detection in enumerate(detections):
        face_data = detection['face_data']
        x1, y1, x2, y2 = [int(x) for x in detection['bbox']]
        conf = detection['confidence']
        
        # Extract embedding
        emb_result = embedder.extract_embedding(frame, face_data)
        
        if emb_result['success']:
            embedding = emb_result['embedding']
            
            # Match identity
            match_result = matcher.match_single(embedding, threshold=0.6)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add identity label if matched
            if match_result['best_match']:
                identity_name = match_result['best_match']['identity_name']
                match_conf = match_result['best_match']['confidence']
                
                label = f"{identity_name} ({match_conf:.2f})"
                label_color = (0, 255, 0)  # Green for match
            else:
                label = "Unknown"
                label_color = (0, 0, 255)  # Red for no match
            
            # Put text on frame
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
            
            # Draw detection confidence
            cv2.putText(frame, f"det: {conf:.2f}", (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Add frame info
    info_text = f"Frame {idx+1}/{len(images)} | Faces: {len(detections)} | Latency: {detection_latency:.1f}ms"
    cv2.putText(frame, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Write frame to video
    out.write(frame)
    
    print(f"Frame {idx+1}: {img_name} - {len(detections)} face(s) detected")

# Release video writer
out.release()

print("\n" + "=" * 70)
print(f"✅ Demo video created successfully!")
print(f"Output: {output_path}")
print("=" * 70)