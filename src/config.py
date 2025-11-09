# Configuration constants for the FRS system

# Detection settings
DETECTION_CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
MIN_FACE_SIZE = 80  # pixels
BLUR_THRESHOLD = 100  # Laplacian variance

# Recognition settings
RECOGNITION_THRESHOLD = 0.6
TOP_K_MATCHES = 5

# Face alignment
FACE_SIZE = 112  # ArcFace standard size
EMBEDDING_DIM = 512

# Database
DATABASE_URL = "sqlite:///./data/gallery.db"

# Model paths
ARCFACE_MODEL_PATH = "models/embedding"

# API settings
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
MAX_FACES_PER_IMAGE = 10