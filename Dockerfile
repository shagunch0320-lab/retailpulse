FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=true \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true

WORKDIR /app
RUN groupadd --system --gid 10001 retailpulse \
    && useradd --system --uid 10001 --gid retailpulse --home-dir /app retailpulse \
    && mkdir -p /app/data/processed /app/.streamlit \
    && chown -R retailpulse:retailpulse /app
COPY --chown=retailpulse:retailpulse pyproject.toml README.md ./
COPY --chown=retailpulse:retailpulse src ./src
COPY --chown=retailpulse:retailpulse app.py ./app.py
RUN pip install --no-cache-dir . \
    && pip check

USER 10001:10001

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)" || exit 1
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.fileWatcherType=none"]