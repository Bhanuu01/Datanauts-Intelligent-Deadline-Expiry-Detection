# Datanauts — Intelligent Deadline & Expiry Detection

An MLOps pipeline that automatically detects deadlines, expiration dates, and renewal terms from contract text using a two-stage NLP architecture: a sentence-level classifier (RoBERTa) gates sentences, then a token-level NER model (BERT) extracts exact date spans.

---

## Architecture

```
Raw contract text (OCR / Paperless-ngx)
        │
        ▼
 build_dataset.py          ← CUAD dataset → 46K labeled sentences
        │
        ├──► train_ner.py          ← BERT token classifier (7-class BIO)
        └──► train_classifier.py   ← RoBERTa sequence classifier (4-class)
                                          │
                              New contract at inference time
                                          │
                                     predict.py
                            CLF filters → NER extracts dates
                                          │
                              Structured JSON output
                            (event_type + date + confidence)
                                          │
                                    evaluate.py        ← end-to-end metrics
                                    feedback_loop.py   ← low-confidence queue
```

---

## Models

| Model | Base | Task | Test F1 |
|-------|------|------|---------|
| `bert_ner_v5` | `dslim/bert-base-NER` | Token classification (NER) | 0.67 |
| `roberta_clf_v5` | `roberta-base` | Sequence classification (CLF) | 0.80 |

### NER Labels
`O` · `B-EXP_DATE` · `I-EXP_DATE` · `B-START_DATE` · `I-START_DATE` · `B-DURATION` · `I-DURATION`

### Classifier Labels
`none` · `expiration` · `effective` · `renewal`

---

## Dataset

Source: [CUAD v1](https://huggingface.co/datasets/cuad) — 510 real-world commercial contracts.

| Split | Contracts | Sentences |
|-------|-----------|-----------|
| Train | 373 | 46,879 |
| Val | 68 | 10,715 |
| Test | 69 | 8,640 |

**Build strategy:**
- **Positive sentences** extracted from CUAD clause columns (`Expiration Date`, `Effective Date`, `Renewal Term`)
- **None sentences** sampled from general OCR text (5:1 ratio)
- **NER labels** generated via regex date-span detector (8+ date formats)
- **Class weights** applied to handle imbalance (`effective`: 10×, `renewal`: 10×)

---

## Project Structure

```
├── src/
│   ├── build_dataset.py         # Dataset builder (hybrid clause-column strategy)
│   ├── train_ner.py             # BERT NER training with weighted loss
│   ├── train_classifier.py      # RoBERTa classifier training with weighted loss
│   ├── train_ner_ray_tune.py    # Ray Tune HPO for NER
│   ├── train_classifier_ray_tune.py  # Ray Tune HPO for classifier
│   ├── predict.py               # Inference pipeline (CLF gate → NER extraction)
│   ├── evaluate.py              # End-to-end evaluation on test set
│   └── feedback_loop.py         # Low-confidence flagging and retrain trigger
├── config/
│   └── config.yaml              # Model configs, hyperparameters
├── samples/
│   ├── ner_sample.json          # Example NER output format
│   └── clf_sample.json          # Example classifier output format
├── Dockerfile                   # GPU training image (PyTorch 2.1 + CUDA 12.1)
├── docker-compose-mlflow.yaml   # MLflow tracking stack (MinIO + PostgreSQL)
├── run_all.sh                   # Full pipeline orchestration script
└── run_demo.sh                  # Quick 1-epoch demo run
```

---

## Setup & Running

### 1. Start MLflow tracking stack
```bash
docker-compose -f docker-compose-mlflow.yaml up -d
# MLflow UI: http://localhost:8000
```

### 2. Build training image
```bash
docker build -t deadline-training:phase2 .
```

### 3. Build dataset
```bash
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  deadline-training:phase2 \
  python src/build_dataset.py
```

### 4. Train models
```bash
# NER (~15 min on H100)
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/tmp \
  deadline-training:phase2 \
  python src/train_ner.py --model bert_ner_v5

# Classifier (~2 min on H100)
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/tmp \
  deadline-training:phase2 \
  python src/train_classifier.py --model roberta_clf_v5
```

### 5. Run inference
```bash
docker run --rm \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/outputs:/tmp \
  deadline-training:phase2 \
  python src/predict.py \
    --clf_model /tmp/deadline-clf-roberta_clf_v5 \
    --ner_model /tmp/deadline-ner-bert_ner_v5 \
    --contract_id "contract_001" \
    --sentences \
      "This Agreement shall expire on December 31, 2025." \
      "The term automatically renews for successive one-year periods."
```

**Example output:**
```json
{
  "contract_id": "contract_001",
  "has_deadline": true,
  "uncertain": false,
  "events": [
    {
      "event_type": "expiration",
      "deadline_date": "2025-12-31",
      "deadline_type": "explicit",
      "confidence": 0.9991,
      "uncertain": false,
      "source_sentence": "This Agreement shall expire on December 31, 2025.",
      "class_scores": {"none": 0.0001, "expiration": 0.9991, "effective": 0.0006, "renewal": 0.0002}
    },
    {
      "event_type": "renewal",
      "deadline_date": null,
      "deadline_type": "computable",
      "confidence": 0.999,
      "uncertain": false,
      "source_sentence": "The term automatically renews for successive one-year periods.",
      "class_scores": {"none": 0.0003, "expiration": 0.0003, "effective": 0.0003, "renewal": 0.999}
    }
  ]
}
```

`deadline_type: "computable"` means a duration was found but no absolute date — the caller computes: `start_date + duration`.  
`uncertain: true` means confidence < 0.7 — sentence is queued for human review via `feedback_loop.py`.

### 6. Run all variants (full benchmark)
```bash
bash run_all.sh
```

---

## MLflow Experiment Tracking

All training runs log to MLflow automatically:
- **Parameters:** model name, base model, lr, epochs, batch size, dataset sizes
- **Metrics:** F1, precision, recall per entity/class, accuracy, training time
- **Artifacts:** classification report text

Experiments: `deadline-detection-ner` · `deadline-detection-classifier`

---

## Feedback Loop

Sentences where `confidence < 0.7` are written to a review queue by `feedback_loop.py`. When 100 uncertain examples accumulate, it triggers a retrain signal. This enables continuous improvement as the model encounters edge cases in production.

---

## Environment

- Base image: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
- Key packages: `transformers==4.46.3`, `datasets==2.19.0`, `mlflow==2.13.0`, `seqeval`, `scikit-learn`
- GPU tested on: NVIDIA H100 (~91GB VRAM)