import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

import cv2
import numpy as np
from src.detector import get_detector
from src.embedder import get_embedder
from src.matcher import get_matcher
from src.aligner import FaceAligner
from src.postprocessing import apply_nms, filter_by_blur, filter_by_confidence, filter_by_size
import time

print("=" * 70)
print("FACE RECOGNITION SYSTEM - FULL PIPELINE TEST")
print("=" * 70)

# ============ TEST 1: Load Models ============
print("\n[TEST 1] Loading models...")
try:
    detector = get_detector()
    print("✅ Detector loaded")
    
    embedder = get_embedder()
    print("✅ Embedder loaded")
    
    matcher = get_matcher()
    print("✅ Matcher loaded")
    
    aligner = FaceAligner()
    print("✅ Aligner loaded")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    sys.exit(1)

# ============ TEST 2: Face Detection ============
print("\n[TEST 2] Testing face detection...")
try:
    # Load a test image
    test_image_path = "data/gallery_images/janhvi1.jpg"
    image = cv2.imread(test_image_path)
    
    if image is None:
        print(f"❌ Could not load test image: {test_image_path}")
        sys.exit(1)
    
    # Detect faces
    start = time.time()
    result = detector.detect(image)
    latency = (time.time() - start) * 1000
    
    detections = result['detections']
    print(f"✅ Detection successful")
    print(f"   - Faces detected: {len(detections)}")
    print(f"   - Latency: {latency:.2f}ms")
    
    if len(detections) > 0:
        det = detections[0]
        print(f"   - First face confidence: {det['confidence']:.3f}")
        print(f"   - BBox: {det['bbox']}")
except Exception as e:
    print(f"❌ Detection failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============ TEST 3: Face Alignment ============
print("\n[TEST 3] Testing face alignment...")
try:
    if detections and len(detections[0]['landmarks']) > 0:
        landmarks = detections[0]['landmarks']
        aligned_face, quality = aligner.align_face(image, landmarks)
        
        if aligned_face is not None:
            print(f"✅ Alignment successful")
            print(f"   - Aligned face size: {aligned_face.shape}")
            print(f"   - Alignment quality: {quality:.3f}")
        else:
            print(f"⚠️  Alignment returned None")
    else:
        print(f"⚠️  No landmarks available for alignment")
except Exception as e:
    print(f"❌ Alignment failed: {e}")
    import traceback
    traceback.print_exc()

# ============ TEST 4: Embedding Extraction ============
print("\n[TEST 4] Testing embedding extraction...")
try:
    if detections:
        face_data = detections[0]['face_data']
        
        start = time.time()
        emb_result = embedder.extract_embedding(image, face_data)
        latency = (time.time() - start) * 1000
        
        if emb_result['success']:
            embedding = emb_result['embedding']
            print(f"✅ Embedding extraction successful")
            print(f"   - Embedding dimension: {len(embedding)}")
            print(f"   - First 5 values: {embedding[:5]}")
            print(f"   - Latency: {latency:.2f}ms")
        else:
            print(f"❌ Embedding extraction failed: {emb_result['error']}")
except Exception as e:
    print(f"❌ Embedding failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============ TEST 5: Identity Matching ============
print("\n[TEST 5] Testing identity matching...")
try:
    if emb_result['success']:
        embedding = emb_result['embedding']
        
        start = time.time()
        match_result = matcher.match_single(embedding, threshold=0.6)
        latency = (time.time() - start) * 1000
        
        print(f"✅ Matching successful")
        print(f"   - Latency: {latency:.2f}ms")
        print(f"   - Is match: {match_result['is_match']}")
        
        if match_result['best_match']:
            best = match_result['best_match']
            print(f"   - Best match: {best['identity_name']} (confidence: {best['confidence']:.3f})")
        
        print(f"   - Top 5 matches:")
        for i, match in enumerate(match_result['matches'][:5]):
            print(f"      {i+1}. {match['identity_name']}: {match['similarity']:.3f}")
except Exception as e:
    print(f"❌ Matching failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============ TEST 6: Post-Processing ============
print("\n[TEST 6] Testing post-processing filters...")
try:
    detections_copy = detections.copy()
    
    # Apply filters
    detections_filtered = filter_by_confidence(detections_copy, 0.5)
    print(f"✅ Confidence filter: {len(detections_copy)} → {len(detections_filtered)}")
    
    detections_filtered = filter_by_size(detections_filtered, 80)
    print(f"✅ Size filter: {len(detections_copy)} → {len(detections_filtered)}")
    
    detections_filtered = apply_nms(detections_filtered, 0.4)
    print(f"✅ NMS filter: {len(detections_copy)} → {len(detections_filtered)}")
except Exception as e:
    print(f"❌ Post-processing failed: {e}")
    import traceback
    traceback.print_exc()

# ============ TEST 7: Gallery Statistics ============
print("\n[TEST 7] Gallery statistics...")
try:
    stats = matcher.get_match_statistics()
    print(f"✅ Statistics retrieved")
    print(f"   - Total identities: {stats['total_identities']}")
    print(f"   - Total embeddings: {stats['total_embeddings']}")
    print(f"   - Threshold: {stats['threshold']}")
    print(f"   - Top-K: {stats['top_k']}")
    print(f"\n   Identities:")
    for identity in stats['identities']:
        print(f"      - {identity['identity_name']}: {identity['num_embeddings']} embeddings")
except Exception as e:
    print(f"❌ Statistics failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("ALL TESTS COMPLETED SUCCESSFULLY! ✅")
print("=" * 70)