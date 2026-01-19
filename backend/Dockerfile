FROM python:3.11-slim

WORKDIR /app

# Reduce memory footprint with environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/sentence_transformers
# Limit PyTorch threads (huge memory savings)
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model (faster cold starts)
# The model files go to the cache directory
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy app code
COPY . .

# Expose port for health checks
EXPOSE 8000

# Run server with single worker and longer timeout for Render free tier
# --timeout-keep-alive: Keep connections alive longer
# --limit-concurrency: Limit memory usage
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "120"]
