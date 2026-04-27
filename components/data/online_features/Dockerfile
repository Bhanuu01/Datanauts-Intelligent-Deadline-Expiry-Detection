FROM python:3.11-slim
WORKDIR /app
COPY components/data/online_features/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY components/data/online_features/feature_service.py .
COPY components/common /app/components/common
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
CMD ["uvicorn", "feature_service:app", "--host", "0.0.0.0", "--port", "8000"]
