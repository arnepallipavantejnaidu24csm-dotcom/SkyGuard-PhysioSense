FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir --default-timeout=120 --retries=10 --user -r requirements.txt


FROM python:3.11-slim AS runner

RUN groupadd -r appgroup && useradd -r -g appgroup -u 10001 appuser

WORKDIR /app

COPY --from=builder /root/.local /home/appuser/.local

ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=5000

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=appuser:appgroup aws_weather_preprocessor /app/aws_weather_preprocessor
COPY --chown=appuser:appgroup web_dashboard /app/web_dashboard
COPY --chown=appuser:appgroup tests /app/tests
COPY --chown=appuser:appgroup *.py /app/
COPY --chown=appuser:appgroup README.md /app/

RUN mkdir -p /app/output && chown -R appuser:appgroup /app

USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/api/metrics || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "web_dashboard.app:app"]