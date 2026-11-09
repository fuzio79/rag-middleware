# ---------------------------------------------------------------------------
# rag-middleware - CPU-only FastAPI RAG service
#
# The build uses an explicit COPY allowlist on purpose.
# Do NOT reintroduce `COPY . .`: it baked .git/ (full history, including
# long-revoked credentials) and .gitea/ into the production image layer.
# Adding a new runtime module means adding it to the COPY lines below.
# ---------------------------------------------------------------------------

# Pinned by digest, not tag: a tag is a moving pointer, so `python:3.11-slim`
# silently changes the base OS under a rebuild. This is the multi-arch
# manifest list for python:3.11-slim (debian trixie-slim), resolved
# 2026-11-02.
#
# To update deliberately:
#   docker buildx imagetools inspect python:3.11-slim
# then copy the top-level "Digest:" value into the FROM line below.
FROM python:3.11-slim@sha256:db20d8506e2f15b6db39b40b6015ca46bae855dec6dd326d462cfe79ae972893 AS base

# --- runtime system deps ---------------------------------------------------
# curl/iputils-ping are kept for in-container diagnostics via docker-exec,
# and curl backs the HEALTHCHECK below.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl iputils-ping \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/usr/src/app/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/usr/src/app/hf_cache

# HF_HOME above is required for the hf_cache volume to have any effect.
# Without it the model is re-downloaded into the image layer on every
# container recreate, and the mounted volume stays unused.

# --- non-root user ---------------------------------------------------------
# uid/gid are build args so the image is not tied to one host's ids.
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd -g "${APP_GID}" appgroup \
 && useradd -u "${APP_UID}" -g appgroup -m appuser

WORKDIR /usr/src/app

# --- dependencies ----------------------------------------------------------
COPY requirements.txt ./

# Install CPU-only torch FIRST. This service never touches CUDA - the embedding
# model runs on CPU - but the default torch wheel pulls the entire NVIDIA stack
# (cudnn, cublas, cusolver, cusparse, cufft, nccl, nvshmem, triton): ~2.5 GB of
# wheels and a measured 10.3 GB image. The +cpu build installed here satisfies
# the sentence-transformers requirement, so pip leaves it alone on the next line.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.8.0

RUN pip install --no-cache-dir -r requirements.txt

# --- application code (explicit allowlist) ---------------------------------
# main.py is the uvicorn entrypoint shim; the application itself is the `rag`
# package. Every runtime module must be listed here.
# The Open WebUI tool is NOT shipped from this repo - it lives in a separate
# tools repository, which is what OWUI actually loads.
COPY main.py ./
COPY rag/__init__.py rag/api.py rag/app.py rag/config.py rag/direct.py \
     rag/documents.py rag/scrapers.py rag/security.py rag/state.py rag/store.py \
     ./rag/

# Cache + vector store are mount points, not build content.
RUN mkdir -p /usr/src/app/hf_cache /usr/src/app/local_knowledge_base \
 && chown -R appuser:appgroup /usr/src/app

USER appuser
EXPOSE 8000

# /health must stay shallow: it must not make outbound API calls.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
