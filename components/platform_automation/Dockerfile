FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY components/training/requirements.txt /app/requirements.txt
COPY components/serving/requirements.onnx_quant.txt /app/requirements-onnx.txt
RUN pip install --no-cache-dir --timeout 180 --retries 10 -r /app/requirements.txt && \
    pip install --no-cache-dir --timeout 180 --retries 10 -r /app/requirements-onnx.txt

ENV HF_HOME=/opt/hf-cache
ENV TRANSFORMERS_CACHE=/opt/hf-cache
RUN python -c "from transformers import AutoModel, AutoModelForTokenClassification, AutoTokenizer; AutoTokenizer.from_pretrained('roberta-base'); AutoModel.from_pretrained('roberta-base'); AutoTokenizer.from_pretrained('dslim/bert-base-NER'); AutoModelForTokenClassification.from_pretrained('dslim/bert-base-NER')"

COPY components/training /app/components/training
COPY components/platform_automation /app/components/platform_automation
COPY components/serving /app/components/serving
COPY components/data/batch_pipeline/output /app/components/data/batch_pipeline/output
COPY components/data/gx_quality/data /app/components/data/gx_quality/data

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

CMD ["python", "/app/components/platform_automation/run_retrain_cycle.py"]
