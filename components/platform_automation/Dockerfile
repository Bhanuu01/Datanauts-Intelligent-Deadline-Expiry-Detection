FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY components/training/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --timeout 180 --retries 10 vcs-versioning setuptools-scm && \
    pip install --no-cache-dir --timeout 180 --retries 10 -r /app/requirements.txt

COPY components/training /app/components/training
COPY components/platform_automation /app/components/platform_automation

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["python", "/app/components/platform_automation/run_retrain_cycle.py"]
