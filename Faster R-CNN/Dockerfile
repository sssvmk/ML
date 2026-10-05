# Real-time inference image. For a CPU-only image, install torch/torchvision from the CPU index first.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY config ./config
ENV REGISTERED_MODEL_NAME=fasterrcnn-mobilenet-v3-large-320-fpn-voc2007
# MLFLOW_TRACKING_URI (and credentials) are injected at deploy time, never baked in; or set MODEL_URI directly.
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
