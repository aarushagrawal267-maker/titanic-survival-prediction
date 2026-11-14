# Titanic ML Lab — container image (works on Render, Railway, Fly.io, Hugging Face Spaces, any VPS)
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TITANIC_DEMO_MODE=true \
    PORT=8501

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# Run as a non-root user that can still write models/ and outputs/ (needed if models are re-trained)
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8501\")}/_stcore/health')" || exit 1
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true"]
