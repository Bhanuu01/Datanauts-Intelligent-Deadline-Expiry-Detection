FROM python:3.10-slim

WORKDIR /app

COPY components/training/requirements.txt /app/requirements.txt
RUN grep -v '^torch' /app/requirements.txt > /app/requirements.no-torch.txt && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --timeout 180 --retries 10 "setuptools<81" vcs-versioning setuptools-scm && \
    pip install --no-cache-dir --timeout 180 --retries 10 --index-url https://download.pytorch.org/whl/cpu torch==2.1.0 && \
    pip install --no-cache-dir --timeout 180 --retries 10 -r /app/requirements.no-torch.txt && \
    pip install --no-cache-dir --timeout 180 --retries 10 "optimum[onnxruntime]==1.23.3"

COPY components/training /app/components/training
COPY components/platform_automation /app/components/platform_automation
COPY components/serving /app/components/serving
COPY components/common /app/components/common

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["python", "/app/components/platform_automation/run_retrain_cycle.py"]
