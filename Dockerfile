# Lightweight reproducible Python environment for AI Chest X-Ray Research Platform
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch and torchvision to keep container lightweight (< 1.5 GB vs > 8 GB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN grep -v "torch" requirements.txt > reqs_no_torch.txt && \
    pip install --no-cache-dir -r reqs_no_torch.txt

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/artifacts /app/reports /app/data && \
    chown -R appuser:appuser /app

# Copy application files
COPY --chown=appuser:appuser configs/ /app/configs/
COPY --chown=appuser:appuser app/ /app/app/
COPY --chown=appuser:appuser artifacts/ /app/artifacts/
COPY --chown=appuser:appuser reports/ /app/reports/
COPY --chown=appuser:appuser scripts/ /app/scripts/

USER appuser

EXPOSE 8000 8501

# Default command: FastAPI inference backend
CMD ["python", "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
