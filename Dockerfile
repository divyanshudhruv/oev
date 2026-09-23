FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image ~2GB instead of ~8GB
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY oev ./oev
RUN pip install --no-cache-dir -e ".[backbone,serve]"

# mount or download checkpoints into /app/checkpoints
EXPOSE 8000
ENTRYPOINT ["oev-serve", "--checkpoint", "/app/checkpoints/oev-tiny.pt", "--host", "0.0.0.0", "--port", "8000"]
