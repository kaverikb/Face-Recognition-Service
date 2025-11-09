FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/gallery_images/ ./data/gallery_images/
COPY scripts/init_db.py ./scripts/init_db.py

RUN mkdir -p data models demo

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 2. Create `.gitignore`

Create file: `.gitignore` at root of FRS folder

Paste this:
```
# Virtual Environment
venv/
env/
ENV/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Models (Large files - DO NOT PUSH)
models/detection/*.pth
models/detection/*.onnx
models/embedding/*.pth
models/embedding/*.onnx
.insightface/

# Database (Local only)
data/gallery.db
data/test_frames/

# Output videos
demo/live_recognition.mp4
demo/output_demo.mp4

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# OS
Thumbs.db
.DS_Store

# Temporary
*.tmp
*.log
*.bak