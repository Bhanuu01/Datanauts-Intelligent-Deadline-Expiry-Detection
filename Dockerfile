FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN pip install --no-cache-dir --timeout 120 \
    transformers==4.46.3 \
    datasets==2.19.0 \
    mlflow==2.13.0 \
    seqeval \
    boto3 \
    psycopg2-binary \
    scikit-learn \
    python-dateutil \
    huggingface-hub \
    accelerate

COPY src/ /app/src/

ENV PYTHONPATH=/app
ENV GIT_PYTHON_REFRESH=quiet

# Run order:
# Step 1 (once):    python src/build_dataset.py
# Step 2 NER:       python src/train_ner.py --model [baseline|bert_ner_v1|bert_ner_v2|bert_ner_v3|bert_base_cased]
# Step 2 CLF:       python src/train_classifier.py --model [baseline|roberta_clf_v1|roberta_clf_v2|roberta_clf_v3]

CMD ["python", "src/train_ner.py", "--model", "bert_ner_v1"]
