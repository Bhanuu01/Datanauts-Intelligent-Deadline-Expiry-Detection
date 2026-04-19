FROM python:3.11-slim

WORKDIR /app

COPY components/inference_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY components/inference_service/app.py /app/app.py
COPY components/training /app/training

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
