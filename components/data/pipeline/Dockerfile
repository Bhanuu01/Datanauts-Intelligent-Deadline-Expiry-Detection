FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY scripts/ ./scripts/
ENV PYTHONUNBUFFERED=1
CMD ["bash", "scripts/run_pipeline.sh"]
