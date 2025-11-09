import numpy as np
from typing import List, Dict, Tuple
from src.embedder import get_embedder
from src.database import SessionLocal, EmbeddingModel, IdentityModel
import time

class FaceMatcher:
    def __init__(self, threshold: float = 0.6, top_k: int = 5):
        """
        Initialize face matcher.
        
        Args:
            threshold: Minimum similarity for match (0-1)
            top_k: Return top K matches
        """
        self.embedder = get_embedder()
        self.threshold = threshold
        self.top_k = top_k
        print(f"✅ Face Matcher initialized (threshold: {threshold}, top_k: {top_k})")
    
    def get_gallery_embeddings(self) -> Dict[int, List[np.ndarray]]:
        """
        Load all embeddings from database, organized by identity.
        
        Returns:
            Dictionary: {identity_id: [embedding1, embedding2, ...]}
        """
        db = SessionLocal()
        
        gallery = {}
        
        try:
            # Get all embeddings
            embeddings = db.query(EmbeddingModel).all()
            
            for emb_record in embeddings:
                identity_id = emb_record.identity_id
                
                # Convert bytes to numpy array
                embedding = self.embedder.bytes_to_embedding(emb_record.embedding_vector)
                
                if identity_id not in gallery:
                    gallery[identity_id] = []
                
                gallery[identity_id].append(embedding)
        
        finally:
            db.close()
        
        return gallery
    
    def match_single(self, query_embedding: np.ndarray, 
                    threshold: float = None) -> Dict:
        """
        Match a single query embedding against gallery.
        
        Args:
            query_embedding: Query face embedding
            threshold: Override default threshold
        
        Returns:
            Dictionary with top matches
        """
        if threshold is None:
            threshold = self.threshold
        
        start_time = time.time()
        
        # Load gallery embeddings
        gallery = self.get_gallery_embeddings()
        
        if not gallery:
            return {
                'matches': [],
                'best_match': None,
                'is_match': False,
                'latency_ms': (time.time() - start_time) * 1000
            }
        
        # Calculate similarity with all gallery embeddings
        similarities = []
        
        for identity_id, embeddings_list in gallery.items():
            # Get identity name
            db = SessionLocal()
            identity = db.query(IdentityModel).filter(
                IdentityModel.id == identity_id
            ).first()
            identity_name = identity.name if identity else f"Unknown_{identity_id}"
            db.close()
            
            # Calculate max similarity for this identity
            max_similarity = 0
            for gallery_emb in embeddings_list:
                sim = self.embedder.cosine_similarity(query_embedding, gallery_emb)
                max_similarity = max(max_similarity, sim)
            
            similarities.append({
                'identity_id': identity_id,
                'identity_name': identity_name,
                'similarity': max_similarity
            })
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Get top K matches
        top_matches = similarities[:self.top_k]
        
        # Check if best match exceeds threshold
        best_match = None
        is_match = False
        
        if top_matches:
            best = top_matches[0]
            if best['similarity'] >= threshold:
                is_match = True
                best_match = {
                    'identity_id': best['identity_id'],
                    'identity_name': best['identity_name'],
                    'confidence': float(best['similarity'])
                }
        
        latency = (time.time() - start_time) * 1000
        
        return {
            'matches': top_matches,
            'best_match': best_match,
            'is_match': is_match,
            'num_matches': len(top_matches),
            'latency_ms': latency
        }
    
    def match_batch(self, query_embeddings: List[np.ndarray], 
                   threshold: float = None) -> List[Dict]:
        """
        Match multiple query embeddings.
        
        Args:
            query_embeddings: List of query embeddings
            threshold: Override default threshold
        
        Returns:
            List of match results
        """
        results = []
        
        for query_emb in query_embeddings:
            result = self.match_single(query_emb, threshold)
            results.append(result)
        
        return results
    
    def set_threshold(self, threshold: float):
        """
        Update similarity threshold.
        
        Args:
            threshold: New threshold (0-1)
        """
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1")
        self.threshold = threshold
        print(f"Threshold updated to {threshold}")
    
    def set_top_k(self, k: int):
        """
        Update number of top matches to return.
        
        Args:
            k: Number of top matches
        """
        if k < 1:
            raise ValueError("top_k must be >= 1")
        self.top_k = k
        print(f"top_k updated to {k}")
    
    def get_match_statistics(self) -> Dict:
        """
        Get statistics about gallery.
        
        Returns:
            Dictionary with gallery statistics
        """
        gallery = self.get_gallery_embeddings()
        
        identities = []
        total_embeddings = 0
        
        for identity_id, embeddings_list in gallery.items():
            db = SessionLocal()
            identity = db.query(IdentityModel).filter(
                IdentityModel.id == identity_id
            ).first()
            db.close()
            
            identities.append({
                'identity_id': identity_id,
                'identity_name': identity.name if identity else f"Unknown_{identity_id}",
                'num_embeddings': len(embeddings_list)
            })
            total_embeddings += len(embeddings_list)
        
        return {
            'total_identities': len(gallery),
            'total_embeddings': total_embeddings,
            'identities': identities,
            'threshold': self.threshold,
            'top_k': self.top_k
        }


# Global matcher instance
_matcher = None

def get_matcher(threshold: float = 0.6, top_k: int = 5):
    """Get or create global matcher instance."""
    global _matcher
    if _matcher is None:
        _matcher = FaceMatcher(threshold=threshold, top_k=top_k)
    return _matcher


def match_embedding(query_embedding: np.ndarray, 
                   threshold: float = 0.6, 
                   top_k: int = 5) -> Dict:
    """
    Convenience function to match a query embedding.
    
    Args:
        query_embedding: Query face embedding
        threshold: Similarity threshold
        top_k: Number of top matches
    
    Returns:
        Match results
    """
    matcher = get_matcher(threshold=threshold, top_k=top_k)
    return matcher.match_single(query_embedding, threshold=threshold)


def match_embeddings_batch(query_embeddings: List[np.ndarray],
                          threshold: float = 0.6,
                          top_k: int = 5) -> List[Dict]:
    """
    Convenience function to match multiple embeddings.
    
    Args:
        query_embeddings: List of query embeddings
        threshold: Similarity threshold
        top_k: Number of top matches
    
    Returns:
        List of match results
    """
    matcher = get_matcher(threshold=threshold, top_k=top_k)
    return matcher.match_batch(query_embeddings, threshold=threshold)