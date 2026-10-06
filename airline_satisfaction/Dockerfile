# Inference service image. Build AFTER a run has produced runs/<run_id>/champion:
#   docker build --build-arg MODEL_DIR=runs/<run_id>/champion -t airsat-serve .
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]" && pip install --no-cache-dir xgboost lightgbm
ARG MODEL_DIR=champion
COPY ${MODEL_DIR} /model
ENV AIRSAT_MODEL_DIR=/model PORT=8000
# NOTE: the bundle is a pickle: serve it with the SAME scikit-learn/xgboost/lightgbm(/torch) versions it was trained with
# (see runs/<run_id>/run_context.json -> env). Behind a TLS-inspecting proxy add the corporate CA and set SSL_CERT_FILE/REQUESTS_CA_BUNDLE.
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["python", "-m", "airsat.serving.app"]
