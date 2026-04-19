FROM python:3.11-slim
WORKDIR /app
RUN pip install pandas great-expectations==0.18.8 numpy
COPY ingestion_quality_check.py .
COPY training_set_quality_check.py .
COPY drift_monitor.py .
ENV PYTHONUNBUFFERED=1
CMD ["python3", "ingestion_quality_check.py"]
