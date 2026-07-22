# ============================================================
# ADIS — AI-Powered Domain Intelligence System
# Flask API production image
#
# Multi-stage build:
#   1) builder  — compiles/installs all Python deps into a venv
#   2) runtime  — slim image with only the venv + runtime libs
#
# Serves:  gunicorn -> api.app:create_app()
# Port:    8080 (matches fly.toml internal_port)
# ============================================================

# ------------------------------------------------------------
# Stage 1 — builder
# ------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential covers any package that falls back to an sdist
# (lightgbm / shap / python-Levenshtein normally ship manylinux wheels,
#  but this keeps the build robust across arches).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Isolated virtualenv we can copy wholesale into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# ------------------------------------------------------------
# Stage 2 — runtime
# ------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH"

# libgomp1 = OpenMP runtime required by LightGBM at inference time.
# curl     = used by the container HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user.
RUN groupadd --system adis \
    && useradd --system --gid adis --home-dir /app --shell /usr/sbin/nologin adis

WORKDIR /app

# Copy the pre-built virtualenv from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Copy application source. .dockerignore keeps the extension, tests,
# training code and raw datasets OUT of the image — only what the API
# needs to serve requests (incl. ml/models/*.pkl) is copied.
COPY --chown=adis:adis . .

USER adis

EXPOSE 8080

# Container-level health probe. Uses the root /health probe registered in
# api/app.py (limiter-exempt; 200 when the model is loaded, 503 otherwise,
# and stays 200 if only Redis is down). start-period is generous because
# the model + SHAP explainer take a few seconds to load on cold boot.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -fsS "http://localhost:8080/health" || exit 1

# Exec form => gunicorn is PID 1 and receives SIGTERM directly for a
# clean drain on deploy/stop. All tuning lives in gunicorn.conf.py so
# it can be adjusted via env vars without rebuilding the image.
CMD ["gunicorn", "-c", "gunicorn.conf.py", "api.app:create_app()"]
