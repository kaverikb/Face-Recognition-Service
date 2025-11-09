import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

import time
import numpy as np
from src.detector import get_detector
from src.embedder import get_embedder
from src.matcher import get_matcher
import cv2

print("=" * 70)
print("CPU PERFORMANCE BENCHMARKING")
print("=" * 70)

def benchmark_detection(num_iterations=10):
    """Benchmark face detection latency."""
    print("\n[BENCHMARK 1] Face Detection...")
    
    try:
        detector = get_detector()
        
        # Load test image
        test_image = cv2.imread("data/gallery_images/janhvi1.jpg")
        if test_image is None:
            print("❌ Could not load test image")
            return None
        
        # Warm up
        _ = detector.detect(test_image)
        
        # Benchmark
        latencies = []
        for i in range(num_iterations):
            start = time.time()
            result = detector.detect(test_image)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            print(f"   Iteration {i+1}: {latency:.2f}ms")
        
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        fps = 1000 / avg_latency
        
        print(f"\n✅ Detection Benchmark Results:")
        print(f"   - Average latency: {avg_latency:.2f}ms")
        print(f"   - Std deviation: {std_latency:.2f}ms")
        print(f"   - FPS: {fps:.2f}")
        print(f"   - Faces detected: {len(result['detections'])}")
        
        return {
            'avg_latency': avg_latency,
            'std_latency': std_latency,
            'fps': fps
        }
    
    except Exception as e:
        print(f"❌ Detection benchmark failed: {e}")
        return None

def benchmark_embedding(num_iterations=10):
    """Benchmark embedding extraction latency."""
    print("\n[BENCHMARK 2] Embedding Extraction...")
    
    try:
        detector = get_detector()
        embedder = get_embedder()
        
        # Load test image
        test_image = cv2.imread("data/gallery_images/janhvi1.jpg")
        if test_image is None:
            print("❌ Could not load test image")
            return None
        
        # Get face data
        result = detector.detect(test_image)
        if not result['detections']:
            print("❌ No faces detected in test image")
            return None
        
        face_data = result['detections'][0]['face_data']
        
        # Warm up
        _ = embedder.extract_embedding(test_image, face_data)
        
        # Benchmark
        latencies = []
        for i in range(num_iterations):
            start = time.time()
            emb = embedder.extract_embedding(test_image, face_data)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            print(f"   Iteration {i+1}: {latency:.2f}ms")
        
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        fps = 1000 / avg_latency
        
        print(f"\n✅ Embedding Benchmark Results:")
        print(f"   - Average latency: {avg_latency:.2f}ms")
        print(f"   - Std deviation: {std_latency:.2f}ms")
        print(f"   - FPS: {fps:.2f}")
        print(f"   - Embedding dimension: {len(emb['embedding'])}")
        
        return {
            'avg_latency': avg_latency,
            'std_latency': std_latency,
            'fps': fps
        }
    
    except Exception as e:
        print(f"❌ Embedding benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def benchmark_matching(num_iterations=100):
    """Benchmark identity matching latency."""
    print("\n[BENCHMARK 3] Identity Matching...")
    
    try:
        detector = get_detector()
        embedder = get_embedder()
        matcher = get_matcher()
        
        # Load test image
        test_image = cv2.imread("data/gallery_images/janhvi1.jpg")
        result = detector.detect(test_image)
        face_data = result['detections'][0]['face_data']
        
        # Get embedding
        emb_result = embedder.extract_embedding(test_image, face_data)
        embedding = emb_result['embedding']
        
        # Warm up
        _ = matcher.match_single(embedding)
        
        # Benchmark
        latencies = []
        for i in range(num_iterations):
            start = time.time()
            match = matcher.match_single(embedding)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            
            if (i + 1) % 10 == 0:
                print(f"   Iteration {i+1}: {latency:.2f}ms")
        
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        fps = 1000 / avg_latency
        
        print(f"\n✅ Matching Benchmark Results:")
        print(f"   - Average latency: {avg_latency:.2f}ms")
        print(f"   - Std deviation: {std_latency:.2f}ms")
        print(f"   - Throughput: {fps:.2f} queries/second")
        
        return {
            'avg_latency': avg_latency,
            'std_latency': std_latency,
            'throughput': fps
        }
    
    except Exception as e:
        print(f"❌ Matching benchmark failed: {e}")
        return None

def benchmark_end_to_end(num_iterations=5):
    """Benchmark full pipeline."""
    print("\n[BENCHMARK 4] End-to-End Pipeline...")
    
    try:
        detector = get_detector()
        embedder = get_embedder()
        matcher = get_matcher()
        
        test_image = cv2.imread("data/gallery_images/janhvi1.jpg")
        
        # Warm up
        result = detector.detect(test_image)
        if result['detections']:
            face_data = result['detections'][0]['face_data']
            emb = embedder.extract_embedding(test_image, face_data)
            if emb['success']:
                _ = matcher.match_single(emb['embedding'])
        
        # Benchmark
        latencies = []
        for i in range(num_iterations):
            start = time.time()
            
            # Detection
            result = detector.detect(test_image)
            if not result['detections']:
                continue
            
            # Embedding
            face_data = result['detections'][0]['face_data']
            emb = embedder.extract_embedding(test_image, face_data)
            if not emb['success']:
                continue
            
            # Matching
            match = matcher.match_single(emb['embedding'])
            
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            print(f"   Iteration {i+1}: {latency:.2f}ms")
        
        avg_latency = np.mean(latencies)
        fps = 1000 / avg_latency
        
        print(f"\n✅ End-to-End Benchmark Results:")
        print(f"   - Average total latency: {avg_latency:.2f}ms")
        print(f"   - FPS: {fps:.2f}")
        
        return {
            'avg_latency': avg_latency,
            'fps': fps
        }
    
    except Exception as e:
        print(f"❌ End-to-end benchmark failed: {e}")
        return None

if __name__ == "__main__":
    print("\nStarting CPU benchmarks...")
    
    results = {}
    results['detection'] = benchmark_detection()
    results['embedding'] = benchmark_embedding()
    results['matching'] = benchmark_matching()
    results['end_to_end'] = benchmark_end_to_end()
    
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    if results['detection']:
        print(f"\nDetection: {results['detection']['avg_latency']:.2f}ms ({results['detection']['fps']:.2f} FPS)")
    
    if results['embedding']:
        print(f"Embedding: {results['embedding']['avg_latency']:.2f}ms ({results['embedding']['fps']:.2f} FPS)")
    
    if results['matching']:
        print(f"Matching: {results['matching']['avg_latency']:.2f}ms ({results['matching']['throughput']:.2f} queries/sec)")
    
    if results['end_to_end']:
        print(f"End-to-End: {results['end_to_end']['avg_latency']:.2f}ms ({results['end_to_end']['fps']:.2f} FPS)")
    
    print("\n" + "=" * 70)