FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

COPY DoCodeDev/frontend/package.json ./package.json
COPY DoCodeDev/frontend/tsconfig.json ./tsconfig.json
COPY DoCodeDev/frontend/tsconfig.node.json ./tsconfig.node.json
COPY DoCodeDev/frontend/vite.config.ts ./vite.config.ts
COPY DoCodeDev/frontend/index.html ./index.html
COPY DoCodeDev/frontend/src ./src

RUN npm install && npm run build

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install \
    fastapi \
    "httpx>=0.27.0" \
    "pydantic>=2.7.0" \
    "uvicorn[standard]>=0.30.0" \
    "typing-extensions>=4.7.1" \
    hatchling \
    editables

COPY weav-ai-core /src/weav-ai-core
COPY weav-ai-providers /src/weav-ai-providers
COPY weav-ai-runtime /src/weav-ai-runtime
COPY DoCodeDev /src/DoCodeDev
COPY --from=frontend-builder /app/frontend/dist /src/DoCodeDev/frontend/dist

RUN pip install --no-deps --no-build-isolation \
    -e /src/weav-ai-core \
    -e /src/weav-ai-providers \
    -e /src/weav-ai-runtime \
    -e /src/DoCodeDev

EXPOSE 8110

CMD ["uvicorn", "docode.main:app", "--host", "0.0.0.0", "--port", "8110"]
