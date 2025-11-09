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
print("LIVE FACE RECOGNITION FROM WEBCAM")
print("=" * 70)

# Load models
print("\nLoading models...")
detector = get_detector()
embedder = get_embedder()
matcher = get_matcher()
print("✅ Models loaded")

# Open webcam
print("\nOpening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam")
    sys.exit(1)

print("✅ Webcam opened")
print("\nPress 'q' to stop recording and save video")
print("=" * 70)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('demo/live_recognition.mp4', fourcc, 20.0, 
                      (int(cap.get(3)), int(cap.get(4))))

frame_count = 0
total_detections = 0
total_matches = 0

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame_count += 1
        start_time = time.time()
        
        # Detect faces
        det_result = detector.detect(frame)
        detections = det_result['detections']
        
        # Filter detections
        detections = filter_by_confidence(detections, 0.5)
        detections = filter_by_size(detections, 80)
        detections = apply_nms(detections, 0.4)
        
        total_detections += len(detections)
        
        # Process each face
        for face_idx, detection in enumerate(detections):
            face_data = detection['face_data']
            x1, y1, x2, y2 = [int(x) for x in detection['bbox']]
            det_conf = detection['confidence']
            
            # Extract embedding
            emb_result = embedder.extract_embedding(frame, face_data)
            
            if emb_result['success']:
                embedding = emb_result['embedding']
                
                # Match identity
                match_result = matcher.match_single(embedding, threshold=0.6)
                
                # Draw bounding box
                if match_result['best_match']:
                    box_color = (0, 255, 0)  # Green for match
                    total_matches += 1
                else:
                    box_color = (0, 0, 255)  # Red for no match
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Add identity label
                if match_result['best_match']:
                    identity_name = match_result['best_match']['identity_name']
                    match_conf = match_result['best_match']['confidence']
                    
                    label = f"{identity_name}: {match_conf:.3f}"
                    cv2.putText(frame, label, (x1, y1 - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Show top 3 matches
                    top_matches = match_result['matches'][:3]
                    y_offset = y1 - 60
                    for match in top_matches:
                        match_text = f"{match['identity_name']}: {match['similarity']:.3f}"
                        cv2.putText(frame, match_text, (x1, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        y_offset -= 25
                else:
                    label = "UNKNOWN"
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Detection confidence
                cv2.putText(frame, f"det: {det_conf:.2f}", (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Frame info
        latency = (time.time() - start_time) * 1000
        info_text = f"Frame: {frame_count} | Faces: {len(detections)} | Latency: {latency:.1f}ms"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display
        cv2.imshow('Face Recognition (Press Q to stop)', frame)
        
        # Write to video
        out.write(frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n\nStopping recording...")
            break

except KeyboardInterrupt:
    print("\nInterrupted")

finally:
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print("RECORDING COMPLETE!")
    print("=" * 70)
    print(f"Total frames: {frame_count}")
    print(f"Total faces detected: {total_detections}")
    print(f"Total matches: {total_matches}")
    print(f"Match rate: {(total_matches/total_detections*100):.1f}%" if total_detections > 0 else "No faces detected")
    print(f"\nOutput video saved: demo/live_recognition.mp4")
    print("=" * 70)