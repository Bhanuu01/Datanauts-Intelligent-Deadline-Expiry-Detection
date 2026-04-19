FROM python:3.11-slim
WORKDIR /app
RUN pip install python-openstackclient
COPY batch_pipeline.py .
ENV PYTHONUNBUFFERED=1
CMD ["python3", "batch_pipeline.py"]
