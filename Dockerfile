FROM python:3.10-slim

# Install system deps for sentence-transformers + general health
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source
COPY --chown=appuser:appgroup . .

# Switch to non-root
USER appuser

# Expose FastAPI + Streamlit ports
EXPOSE 8000 8501

# Start both FastAPI and Streamlit via a simple wrapper script
COPY --chown=appuser:appgroup scripts/entrypoint.sh /app/scripts/entrypoint.sh
RUN chmod +x /app/scripts/entrypoint.sh

# Use exec form for proper signal handling (SIGTERM propagation)
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
