import sys
import os

# Add parent directory to path BEFORE any other imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sqlalchemy.orm import Session
from src.database import SessionLocal, IdentityModel, EmbeddingModel
from src.postprocessing import calculate_blur_score

def init_database():
    """
    Initialize database with 20 images from gallery_images folder.
    Process each image, extract embedding, store in SQLite.
    """
    
    print("Initializing Face Recognition Database...")
    print("=" * 60)
    
    # Initialize InsightFace model
    print("\nLoading ArcFace model...")
    app = FaceAnalysis(name='buffalo_l', providers=['CPUProvider'])
    app.prepare(ctx_id=-1)  # -1 = CPU
    print("✅ ArcFace model loaded")
    
    # Get database session
    db = SessionLocal()
    
    # Get all image files
    gallery_path = "data/gallery_images"
    image_files = sorted([f for f in os.listdir(gallery_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"\nFound {len(image_files)} images in gallery_images/")
    print("=" * 60)
    
    # Group images by person
    people = {}
    for img_file in image_files:
        # Extract person name (e.g., "janhvi" from "janhvi1.jpg")
        person_name = ''.join([c for c in img_file if not c.isdigit()]).replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
        
        if person_name not in people:
            people[person_name] = []
        people[person_name].append(img_file)
    
    print(f"Found {len(people)} people:")
    for person, images in people.items():
        print(f"  - {person}: {len(images)} images")
    
    print("\n" + "=" * 60)
    print("Processing images and extracting embeddings...")
    print("=" * 60)
    
    total_embeddings = 0
    
    # Process each person
    for person_name, image_list in sorted(people.items()):
        print(f"\nProcessing: {person_name}")
        
        # Check if identity already exists
        existing_identity = db.query(IdentityModel).filter(IdentityModel.name == person_name).first()
        
        if existing_identity:
            identity = existing_identity
            print(f"  Identity already exists (ID: {identity.id})")
        else:
            # Create new identity
            identity = IdentityModel(name=person_name, is_active=True)
            db.add(identity)
            db.commit()
            db.refresh(identity)
            print(f"  ✅ Created new identity (ID: {identity.id})")
        
        # Process each image for this person
        for img_file in image_list:
            img_path = os.path.join(gallery_path, img_file)
            
            try:
                # Read image
                image = cv2.imread(img_path)
                
                if image is None:
                    print(f"    ❌ Failed to load {img_file}")
                    continue
                
                # Detect faces in image
                faces = app.get(image)
                
                if len(faces) == 0:
                    print(f"    ⚠️  No face detected in {img_file}")
                    continue
                
                if len(faces) > 1:
                    print(f"    ⚠️  Multiple faces detected in {img_file}, using first one")
                
                # Get first (largest) face
                face = faces[0]
                
                # Extract embedding
                embedding = face.embedding  # 512-dim vector
                
                # Calculate quality scores
                blur_score = calculate_blur_score(image)
                detection_confidence = face.det_score
                
                # Convert embedding to bytes
                embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
                
                # Store in database
                embedding_record = EmbeddingModel(
                    identity_id=identity.id,
                    embedding_vector=embedding_bytes,
                    image_path=img_path,
                    quality_score=blur_score,
                    detection_confidence=float(detection_confidence),
                    alignment_quality=1.0
                )
                
                db.add(embedding_record)
                db.commit()
                
                total_embeddings += 1
                print(f"    ✅ {img_file} - embedding stored (blur: {blur_score:.1f}, conf: {detection_confidence:.3f})")
                
            except Exception as e:
                print(f"    ❌ Error processing {img_file}: {str(e)}")
                continue
    
    print("\n" + "=" * 60)
    print(f"Database initialization complete!")
    print(f"Total embeddings stored: {total_embeddings}")
    
    # Print summary
    identities = db.query(IdentityModel).all()
    print(f"\nSummary:")
    for identity in identities:
        num_embeddings = db.query(EmbeddingModel).filter(EmbeddingModel.identity_id == identity.id).count()
        print(f"  - {identity.name}: {num_embeddings} embeddings")
    
    print("=" * 60)
    
    db.close()

if __name__ == "__main__":
    init_database()