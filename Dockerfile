# The API as a container image.
# Ollama is NOT inside: it stays on the host, where it can use the GPU. This container runs on the CPU,
# which is enough for embedding one question and running YOLO on a handful of photos.
FROM python:3.11-slim

# System programs: Tesseract for OCR, plus two libraries that OpenCV (used by YOLO) needs
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch: a fraction of the size of the GPU build, and the container has no GPU anyway
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Libraries first, code second: Docker reuses this slow step as long as requirements.txt doesn't change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
