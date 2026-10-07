# Build AFTER a run has produced runs/<run_id>/champion:
#   docker build --build-arg MODEL_DIR=runs/<run_id>/champion -t aeclf-serve .
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"
ARG MODEL_DIR=champion
COPY ${MODEL_DIR} /model
ENV AECLF_MODEL_DIR=/model PORT=8000
# Weights are plain tensors, but use the torch/scikit-learn versions in runs/<run_id>/run_context.json (encoder.joblib is a joblib file).
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["python", "-m", "aeclf.serving.app"]
