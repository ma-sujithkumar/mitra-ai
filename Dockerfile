FROM node:22-bookworm-slim AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package*.json ./
# The current frontend lockfile is not accepted by Linux npm ci because of
# optional platform-package metadata, so resolve frontend deps inside this
# disposable builder layer without mutating the repository lockfile.
RUN npm install --package-lock=false

COPY frontend/ ./
RUN npm run build


FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/huggingface \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=7860

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        libgomp1 \
        nginx \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/nginx /tmp/huggingface /tmp/matplotlib

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

COPY backend/ backend/
COPY llm/ llm/
COPY model_library/ model_library/
COPY config.ini config.ini
COPY sdd_prompt sdd_prompt
COPY --from=frontend-builder /app/frontend/dist frontend/dist
COPY deploy/nginx.conf /etc/nginx/nginx.conf
COPY deploy/start.sh deploy/start.sh

RUN chmod +x deploy/start.sh \
    && mkdir -p .mitra/logs

EXPOSE 7860

CMD ["./deploy/start.sh"]
