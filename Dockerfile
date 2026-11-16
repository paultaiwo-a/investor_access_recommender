# Serving image: FastAPI on :8000 (default) or the Streamlit demo on :8501.
# Build the serving artifacts first (make serving), they are copied into the image.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY api ./api
COPY app ./app
RUN uv sync --frozen --no-dev
COPY artifacts/serving ./artifacts/serving

ENV RECS_SERVING_DIR=/app/artifacts/serving PATH="/app/.venv/bin:$PATH"
EXPOSE 8000 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
