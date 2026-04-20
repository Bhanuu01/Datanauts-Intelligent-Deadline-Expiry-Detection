import json
import os
import re
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

from fastapi import FastAPI
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.onnxruntime import ORTModelForTokenClassification
from prometheus_client import Counter
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from transformers import AutoTokenizer
from transformers import pipeline


CLF_MODEL_PATH = Path(Path.cwd() / "onnx_quantized_clf")
NER_MODEL_PATH = Path(Path.cwd() / "onnx_quantized_ner")
if "ONNX_CLF_MODEL_PATH" in os.environ:
    CLF_MODEL_PATH = Path(os.environ["ONNX_CLF_MODEL_PATH"])
if "ONNX_NER_MODEL_PATH" in os.environ:
    NER_MODEL_PATH = Path(os.environ["ONNX_NER_MODEL_PATH"])
if "ONNX_MODEL_PATH" in os.environ:
    base_path = Path(os.environ["ONNX_MODEL_PATH"])
    if not CLF_MODEL_PATH.exists():
        CLF_MODEL_PATH = base_path / "onnx_quantized_clf"
    if not NER_MODEL_PATH.exists():
        NER_MODEL_PATH = base_path / "onnx_quantized_ner"

FEEDBACK_LOG_PATH = Path(os.environ.get("FEEDBACK_LOG_PATH", "/data/serving_feedback.jsonl"))
CONFIDENCE_THRESHOLD = float(os.environ.get("ONNX_CONFIDENCE_THRESHOLD", "0.7"))

app = FastAPI(title="Deadline Detection ONNX Quantized Service")
Instrumentator().instrument(app).expose(app)

CONFIDENCE_METRIC = Histogram(
    "deadline_onnx_confidence_score",
    "Combined confidence score of the ONNX serving path.",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)
PREDICTION_COUNTER = Counter(
    "deadline_onnx_predictions_total",
    "Total documents processed by the ONNX serving path.",
)
UNCERTAIN_COUNTER = Counter(
    "deadline_onnx_uncertain_predictions_total",
    "Total ONNX predictions flagged as uncertain for review.",
)
CORRECTION_COUNTER = Counter(
    "deadline_onnx_user_corrections_total",
    "Total user corrections received by the ONNX serving path.",
)

clf_model = ORTModelForSequenceClassification.from_pretrained(str(CLF_MODEL_PATH))
clf_tokenizer = AutoTokenizer.from_pretrained(str(CLF_MODEL_PATH))
clf_pipeline = pipeline("text-classification", model=clf_model, tokenizer=clf_tokenizer, top_k=None)

ner_model = ORTModelForTokenClassification.from_pretrained(str(NER_MODEL_PATH))
ner_tokenizer = AutoTokenizer.from_pretrained(str(NER_MODEL_PATH))
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")


class DocumentRequest(BaseModel):
    document_id: str
    ocr_text: str


class FeedbackRequest(BaseModel):
    document_id: str
    event_type: str = "none"
    confidence: float = 0.0
    original_prediction: str | None = None
    user_corrected_value: str | None = None
    notes: str | None = None


def split_into_sentences(text: str) -> List[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "classifier_model_path": str(CLF_MODEL_PATH),
        "ner_model_path": str(NER_MODEL_PATH),
        "feedback_log_path": str(FEEDBACK_LOG_PATH),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


@app.post("/predict")
async def predict(req: DocumentRequest) -> Dict[str, Any]:
    PREDICTION_COUNTER.inc()
    sentences = split_into_sentences(req.ocr_text)

    events: List[Dict[str, Any]] = []
    has_conflict = False

    for sentence in sentences:
        clf_results = clf_pipeline(sentence)[0]
        class_scores = {entry["label"]: float(entry["score"]) for entry in clf_results}
        top_class = max(class_scores, key=class_scores.get)
        top_score = class_scores[top_class]

        if top_class == "none":
            continue

        ner_results = ner_pipeline(sentence)
        extracted_dates = [entity["word"] for entity in ner_results]
        ner_confidence = float(ner_results[0]["score"]) if ner_results else 1.0
        combined_confidence = (top_score + ner_confidence) / 2
        is_uncertain = combined_confidence < CONFIDENCE_THRESHOLD

        CONFIDENCE_METRIC.observe(combined_confidence)
        if is_uncertain:
            UNCERTAIN_COUNTER.inc()

        events.append(
            {
                "event_type": top_class,
                "deadline_date": extracted_dates[0] if extracted_dates else None,
                "date_candidates": extracted_dates,
                "conflict_flag": len(extracted_dates) > 1,
                "confidence": round(combined_confidence, 4),
                "uncertain": is_uncertain,
                "source_sentence": sentence,
                "class_scores": class_scores,
            }
        )
        if len(extracted_dates) > 1:
            has_conflict = True

    return {
        "contract_id": req.document_id,
        "has_deadline": len(events) > 0,
        "uncertain": any(event["uncertain"] for event in events),
        "multi_date_conflict": has_conflict,
        "events": events,
        "mode": "onnx_quantized_two_stage",
    }


@app.post("/feedback")
async def capture_feedback(req: FeedbackRequest) -> Dict[str, Any]:
    CORRECTION_COUNTER.inc()
    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = req.model_dump()
    with FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    return {"status": "feedback logged for retraining"}
