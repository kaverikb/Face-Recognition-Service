import numpy as np
from insightface.app import FaceAnalysis
from typing import Dict, List
import time

class FaceEmbedder:
    def __init__(self, model_name='buffalo_l', ctx_id=-1):
        """
        Initialize face embedder with ArcFace.
        
        Args:
            model_name: Model name (buffalo_l includes ArcFace)
            ctx_id: -1 for CPU, 0+ for GPU
        """
        print("Initializing Face Embedder...")
        self.app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=ctx_id)
        self.embedding_dim = 512  # ArcFace output dimension
        print(f"✅ Face Embedder initialized (embedding dim: {self.embedding_dim})")
    
    def extract_embedding(self, image: np.ndarray, face_data) -> Dict:
        """
        Extract embedding from a face.
        
        Args:
            image: BGR image (full image)
            face_data: Face object from detector (contains landmarks for alignment)
        
        Returns:
            Dictionary with embedding and metadata
        """
        start_time = time.time()
        
        try:
            # Get embedding from face object
            embedding = face_data.embedding  # 512-dim vector
            
            latency = (time.time() - start_time) * 1000  # ms
            
            return {
                'embedding': embedding.astype(np.float32),
                'embedding_normalized': embedding / np.linalg.norm(embedding),  # L2 normalized
                'dim': len(embedding),
                'latency_ms': latency,
                'success': True
            }
        
        except Exception as e:
            return {
                'embedding': None,
                'embedding_normalized': None,
                'dim': self.embedding_dim,
                'latency_ms': (time.time() - start_time) * 1000,
                'success': False,
                'error': str(e)
            }
    
    def extract_embeddings_batch(self, image: np.ndarray, 
                                faces: List) -> Dict:
        """
        Extract embeddings from multiple faces in one image.
        
        Args:
            image: BGR image
            faces: List of face objects from detector
        
        Returns:
            Dictionary with list of embeddings
        """
        start_time = time.time()
        embeddings = []
        
        for i, face in enumerate(faces):
            result = self.extract_embedding(image, face)
            if result['success']:
                embeddings.append({
                    'face_id': i,
                    'embedding': result['embedding'],
                    'embedding_normalized': result['embedding_normalized']
                })
        
        latency = (time.time() - start_time) * 1000
        
        return {
            'embeddings': embeddings,
            'num_embeddings': len(embeddings),
            'latency_ms': latency
        }
    
    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        L2 normalize embedding (for cosine similarity).
        
        Args:
            embedding: Raw embedding vector
        
        Returns:
            Normalized embedding
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    def embedding_to_bytes(self, embedding: np.ndarray) -> bytes:
        """
        Convert embedding to bytes for database storage.
        
        Args:
            embedding: Embedding vector
        
        Returns:
            Bytes representation
        """
        return np.array(embedding, dtype=np.float32).tobytes()
    
    def bytes_to_embedding(self, embedding_bytes: bytes) -> np.ndarray:
        """
        Convert bytes back to embedding vector.
        
        Args:
            embedding_bytes: Bytes from database
        
        Returns:
            Embedding vector
        """
        return np.frombuffer(embedding_bytes, dtype=np.float32)
    
    def cosine_similarity(self, embedding1: np.ndarray, 
                         embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
        
        Returns:
            Cosine similarity score (0-1, higher = more similar)
        """
        # Normalize both embeddings
        e1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        e2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        
        # Cosine similarity is dot product of normalized vectors
        similarity = np.dot(e1_norm, e2_norm)
        
        # Clamp to [0, 1] range
        return float(np.clip(similarity, 0, 1))
    
    def euclidean_distance(self, embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
        
        Returns:
            Euclidean distance (lower = more similar)
        """
        distance = np.linalg.norm(embedding1 - embedding2)
        return float(distance)


# Global embedder instance
_embedder = None

def get_embedder():
    """Get or create global embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = FaceEmbedder()
    return _embedder


def extract_embedding(image: np.ndarray, face_data) -> Dict:
    """
    Convenience function to extract embedding.
    
    Args:
        image: BGR image
        face_data: Face object from detector
    
    Returns:
        Embedding result
    """
    embedder = get_embedder()
    return embedder.extract_embedding(image, face_data)


def extract_embeddings_batch(image: np.ndarray, faces: List) -> Dict:
    """
    Convenience function to extract multiple embeddings.
    
    Args:
        image: BGR image
        faces: List of face objects
    
    Returns:
        Batch embedding result
    """
    embedder = get_embedder()
    return embedder.extract_embeddings_batch(image, faces)


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Convenience function for cosine similarity.
    
    Args:
        emb1: First embedding
        emb2: Second embedding
    
    Returns:
        Similarity score (0-1)
    """
    embedder = get_embedder()
    return embedder.cosine_similarity(emb1, emb2)