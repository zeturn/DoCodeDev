FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY WeavInt/packages/weav-core /src/weav-core
COPY weav-ai-core /src/weav-ai-core
COPY weav-ai-providers /src/weav-ai-providers
COPY weav-ai-runtime /src/weav-ai-runtime
COPY DoCodeDev /src/DoCodeDev

RUN pip install --no-cache-dir \
    fastapi \
    "httpx>=0.27.0" \
    "pydantic>=2.7.0" \
    "uvicorn[standard]>=0.30.0" \
    "typing-extensions>=4.7.1" \
    hatchling \
 && pip install --no-cache-dir --no-deps -e /src/weav-core -e /src/weav-ai-core -e /src/weav-ai-providers -e /src/weav-ai-runtime -e /src/DoCodeDev

EXPOSE 8110

CMD ["uvicorn", "docode.main:app", "--host", "0.0.0.0", "--port", "8110"]
