import json
import os
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

from fastapi import FastAPI
from optimum.onnxruntime import ORTModelForTokenClassification
from prometheus_client import Counter
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from transformers import AutoTokenizer
from transformers import pipeline


MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "/models/onnx_quantized_model")
FEEDBACK_LOG_PATH = Path(os.getenv("FEEDBACK_LOG_PATH", "/data/serving_feedback.jsonl"))

app = FastAPI(title="Deadline Detection ONNX Quantized Service")
Instrumentator().instrument(app).expose(app)

CONFIDENCE_METRIC = Histogram(
    "deadline_onnx_confidence_score",
    "Confidence score of extracted entities from the quantized ONNX deadline model.",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)
PREDICTION_COUNTER = Counter(
    "deadline_onnx_predictions_total",
    "Total ONNX deadline predictions served.",
)
CORRECTION_COUNTER = Counter(
    "deadline_onnx_user_corrections_total",
    "Total user corrections received by the ONNX serving path.",
)


model = ORTModelForTokenClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


class DocumentRequest(BaseModel):
    document_id: str
    ocr_text: str


class FeedbackRequest(BaseModel):
    document_id: str
    original_prediction: str
    user_corrected_value: str


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "feedback_log_path": str(FEEDBACK_LOG_PATH),
    }


@app.post("/predict")
async def predict(req: DocumentRequest) -> Dict[str, Any]:
    results = ner(req.ocr_text)
    confidence = float(results[0]["score"]) if results else 0.0

    PREDICTION_COUNTER.inc()
    if confidence > 0:
        CONFIDENCE_METRIC.observe(confidence)

    deadlines: List[Dict[str, Any]] = [
        {
            "extracted_text": entity["word"],
            "event_type": "deadline_entity",
            "confidence": float(entity["score"]),
        }
        for entity in results
    ]

    return {
        "document_id": req.document_id,
        "deadlines": deadlines,
        "mode": "onnx_quantized",
    }


@app.post("/feedback")
async def capture_feedback(req: FeedbackRequest) -> Dict[str, Any]:
    CORRECTION_COUNTER.inc()
    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(req.model_dump()) + "\n")

    return {"status": "feedback logged for retraining"}
