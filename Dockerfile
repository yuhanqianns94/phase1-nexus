FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PHASE1_DEPLOYMENT=public \
    PHASE1_DATA_ROOT=/var/data/phase1 \
    JAX_PLATFORMS=cpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app
COPY requirements-deploy.txt ./requirements-deploy.txt
RUN python -m pip install --no-cache-dir -r requirements-deploy.txt
COPY v3/ ./v3/
RUN python -m pip install --no-cache-dir --no-deps -e ./v3

# The persistent volume is mounted at /var/data on Render.
RUN mkdir -p /var/data/phase1
WORKDIR /app/v3
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8501') + '/_stcore/health', timeout=4)"
CMD ["python", "-m", "phase1_web.server"]
