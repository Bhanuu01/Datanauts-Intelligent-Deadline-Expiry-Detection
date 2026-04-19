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
    # unicodedata is Python stdlib — no install needed

COPY src/ /app/src/
COPY config/ /app/config/

ENV PYTHONPATH=/app
ENV GIT_PYTHON_REFRESH=quiet

# Run order (or use run_all.sh):
# Step 1 (once):    python src/build_dataset.py
# Step 2 NER:       python src/train_ner.py    --model [baseline|bert_ner_v1..v5|bert_base_cased]
# Step 2 CLF:       python src/train_classifier.py --model [baseline|roberta_clf_v1..v6]
# Step 3 Evaluate:  python src/evaluate.py     --clf_model <path> --ner_model <path>
# Step 4 Feedback:  python src/feedback_loop.py --collect --predictions predict_out.json
# Step 5 Predict:   python src/predict.py      --clf_model <path> --ner_model <path> --sentences "..."

CMD ["python", "src/train_ner.py", "--model", "bert_ner_v5"]
