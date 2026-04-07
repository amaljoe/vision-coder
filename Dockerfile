# VisionCoder OpenEnv Server
# Runs the FastAPI environment server, compatible with the OpenEnv framework.
#
# Build:  docker build -t vision-coder-env .
# Run:    docker run -p 8080:8080 vision-coder-env
#
# The server exposes:
#   POST /reset  — start a new episode (returns target screenshot)
#   POST /step   — submit HTML for scoring (returns reward)
#   GET  /state  — episode metadata

FROM python:3.11-slim

WORKDIR /app

# System deps: Playwright requires Chromium + shared libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl gnupg ca-certificates \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libasound2 libpango-1.0-0 libpangocairo-1.0-0 \
    fonts-liberation fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for layer caching
COPY pyproject.toml requirements.txt ./

# Install Python packages
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    httpx \
    pydantic \
    torch --index-url https://download.pytorch.org/whl/cpu \
    transformers \
    datasets \
    huggingface-hub \
    pillow \
    scikit-image \
    beautifulsoup4 \
    html5lib \
    lxml \
    playwright \
    open-clip-torch

# Install Playwright's Chromium browser
RUN playwright install chromium --with-deps

# Copy project source
COPY vcoder/ ./vcoder/
COPY openenv/ ./openenv/

# Install the vcoder package in editable mode
RUN pip install --no-cache-dir -e .

# Expose the OpenEnv HTTP port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/state || exit 1

CMD ["uvicorn", "openenv.server.app:app", "--host", "0.0.0.0", "--port", "8080"]
