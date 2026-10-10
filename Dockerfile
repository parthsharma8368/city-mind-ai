# syntax=docker/dockerfile:1
# Backend image: FastAPI + SUMO (eclipse-sumo PyPI wheel, no apt SUMO needed).
# Build:  docker build -t citymind-api --target api .
# Frontend image: docker build -t citymind-web --target web --build-arg VITE_API_URL=http://localhost:8000/api .

FROM python:3.11-slim AS py-build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY pyproject.toml requirements.lock README.md ./
COPY src ./src
COPY api ./api
RUN python -m venv /opt/venv && /opt/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && /opt/venv/bin/pip install -c requirements.lock ".[sumo]"

FROM python:3.11-slim AS api
ENV PYTHONUNBUFFERED=1 PATH=/opt/venv/bin:$PATH
RUN useradd --create-home --uid 10001 citymind
COPY --from=py-build /opt/venv /opt/venv
WORKDIR /app
COPY --chown=citymind src ./src
COPY --chown=citymind api ./api
COPY --chown=citymind configs ./configs
# Scenario metadata + metrics are mounted/copied at runtime (see docker-compose.yml); weights are never baked in.
ENV SUMO_HOME=/opt/venv/lib/python3.11/site-packages/sumo
USER citymind
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/docs')" || exit 1
# 0.0.0.0 is for the container namespace only; compose publishes it on host loopback. Set CITYMIND_API_KEY.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM node:lts-slim AS web-build
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend ./
ARG VITE_API_URL=http://localhost:8000/api
ARG VITE_WS_URL=ws://localhost:8000/ws/simulation
ENV VITE_API_URL=$VITE_API_URL VITE_WS_URL=$VITE_WS_URL
RUN npm run build

FROM nginxinc/nginx-unprivileged:stable-alpine AS web
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=web-build /web/dist /usr/share/nginx/html
EXPOSE 8080
