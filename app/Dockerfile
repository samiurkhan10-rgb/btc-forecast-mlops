# syntax=docker/dockerfile:1
# Production image for the Flask prediction API.
# Build from the repo root:  docker build -f app/Dockerfile -t samiurk/btc-forecast-api:local .

# ---------- Stage 1: builder - install dependencies into an isolated venv ----------
FROM python:3.10-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy ONLY the requirements first: this layer is rebuilt only when dependencies change,
# not on every code edit (layer caching).
COPY requirements-api.txt /tmp/requirements-api.txt
ARG SP=/opt/venv/lib/python3.10/site-packages
# pip download cache lives in a BuildKit cache mount: re-builds skip re-downloading wheels,
# and the cache never ends up inside the image.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements-api.txt \
    # Shrink the venv: bundled test suites are never used at runtime.
    # numpy is excluded because numpy.testing (imported by scipy) needs numpy/_core/tests.
    && find ${SP} -depth -type d -name tests -not -path "${SP}/numpy/*" -exec rm -rf {} + \
    && find /opt/venv -name "*.pyx" -delete \
    # pip/setuptools are only needed to install, not to run
    && pip uninstall -y setuptools \
    && python -m pip uninstall -y pip \
    # fail the BUILD (not production) if the cleanup broke an import
    && python -c "import xgboost, scipy.sparse, pandas, numpy, flask, gunicorn; print('imports ok')"

# ---------- Stage 2: runtime - slim base + venv + code, running as non-root ----------
FROM python:3.10-slim AS runtime

LABEL org.opencontainers.image.title="btc-forecast-api" \
      org.opencontainers.image.description="Flask + gunicorn API serving a 7-day BTC-USD close forecast" \
      org.opencontainers.image.source="https://github.com/samiurkhan10-rgb/btc-forecast-mlops"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODELS_DIR=/app/models \
    MODEL_STAGE=production \
    LOG_LEVEL=INFO \
    PORT=5000 \
    GUNICORN_WORKERS=2 \
    GUNICORN_THREADS=4 \
    # keep-alive > a reverse proxy's upstream keepalive (ingress-nginx: 60 s) avoids 502 races
    GUNICORN_CMD_ARGS="--keep-alive 75"

# Dedicated unprivileged user (fixed UID so K8s securityContext can reference it)
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# Code is copied AFTER dependencies so code changes don't invalidate the pip layer.
# Only the modules the API imports are shipped (no ETL / training code).
COPY src/__init__.py src/config.py src/features.py src/forecast.py src/logging_config.py ./src/
COPY app/__init__.py app/app.py app/model_store.py ./app/
RUN mkdir -p /app/models && chown app:app /app/models

USER 10001:10001
EXPOSE 5000

# Docker-level health check (K8s uses its own liveness/readiness probes on the same endpoint)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','5000'), timeout=4)" || exit 1

# exec -> gunicorn becomes PID 1 and receives SIGTERM directly (graceful shutdown).
# gthread: each of the 2 workers serves up to 4 requests concurrently, so a quick /health
# probe is never stuck behind slow /predict calls (sync workers handle 1 request at a time,
# which made liveness probes time out under load and the kubelet restart busy pods).
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers ${GUNICORN_WORKERS} --worker-class gthread --threads ${GUNICORN_THREADS} --timeout 60 --graceful-timeout 20 --worker-tmp-dir /dev/shm app.app:app"]
