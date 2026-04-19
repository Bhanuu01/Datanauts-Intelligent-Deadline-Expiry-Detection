FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
COPY feature_service.py .
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "feature_service:app", "--host", "0.0.0.0", "--port", "8000"]
