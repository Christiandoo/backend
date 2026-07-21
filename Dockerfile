# ─── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies (torch CPU build lebih ringan)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dulu (layer caching)
COPY requirements.txt .

# Install PyTorch CPU-only untuk hemat RAM di Railway
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# ─── App ──────────────────────────────────────────────────────────────────────
COPY . .

# Pre-download NLTK stopwords
RUN python -c "import nltk; nltk.download('stopwords', quiet=True)"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
