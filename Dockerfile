FROM python:3.11-slim@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e

# CPU-only torch keeps the image ~2GB instead of ~8GB
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/hf-cache \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# deps first: the hash-locked layer only rebuilds when the lock changes
COPY ci/requirements-docker.txt ./ci/
RUN pip install --require-hashes --no-deps -r ci/requirements-docker.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY oev ./oev
RUN pip install --no-deps .

# non-root runtime user; owns only the cache dir
RUN useradd --create-home --uid 1000 oev \
    && mkdir -p /app/hf-cache /app/checkpoints \
    && chown -R oev:oev /app/hf-cache
USER oev

# checkpoint: mount a local .pt at /app/checkpoints/oev-tiny.pt, or point
# OEV_CHECKPOINT at a Hugging Face id (repo/filename, e.g.
# divyanshudhruv/oev-typed/student-r2b-oev-tiny.pt) to download at startup.
# Override the command for --host/--port/--device flags if needed.
ENV OEV_CHECKPOINT=/app/checkpoints/oev-tiny.pt
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('OEV_PORT','8000')}/health\", timeout=4)" || exit 1

ENTRYPOINT ["oev-serve", "--host", "0.0.0.0", "--port", "8000"]
