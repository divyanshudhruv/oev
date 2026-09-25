FROM python:3.11-slim

# CPU-only torch keeps the image ~2GB instead of ~8GB
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/hf-cache \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# torch first: it is the big, rarely-changing layer
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY oev ./oev
RUN pip install ".[backbone,serve]"

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
