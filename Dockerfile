# ==============================================================================
# Multi-Stage Dockerfile for AWS Weather Preprocessing & Dashboard Platform
# ==============================================================================

# Stage 1: Build & Dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=120 --retries=10 --user -r requirements.txt

# Stage 2: Final Lightweight Runtime Image
FROM python:3.11-slim AS runner

# Create non-root security user
RUN groupadd -r appgroup && useradd -r -g appgroup -u 10001 appuser

WORKDIR /app

# Copy installed Python packages from builder stage
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=5000

# Install curl for docker healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application source code
COPY --chown=appuser:appgroup aws_weather_preprocessor /app/aws_weather_preprocessor
COPY --chown=appuser:appgroup web_dashboard /app/web_dashboard
COPY --chown=appuser:appgroup tests /app/tests
COPY --chown=appuser:appgroup *.py /app/
COPY --chown=appuser:appgroup README.md /app/

# Create output and scratch directories with proper permissions
RUN mkdir -p /app/output && chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose web dashboard port
EXPOSE 5000

# Healthcheck monitoring endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/api/metrics || exit 1

# Default command: launch web dashboard
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "web_dashboard.app:app"]
