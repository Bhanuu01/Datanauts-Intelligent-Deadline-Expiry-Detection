import importlib.util
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from dateutil import parser as dateutil_parser
from fastapi import FastAPI
from fastapi import HTTPException
from prometheus_client import CONTENT_TYPE_LATEST
from prometheus_client import Counter
from prometheus_client import Histogram
from prometheus_client import generate_latest
from pydantic import BaseModel
from starlette.responses import Response


DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{1,2},?\s+\d{4}"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}"
    r"|\b\d{4}-\d{2}-\d{2}",
    re.IGNORECASE,
)

EVENT_KEYWORDS = {
    "expiration": ("expire", "expiration", "expiry", "terminate", "termination", "end date"),
    "effective": ("effective", "start date", "commence", "commencement", "begins on"),
    "renewal": ("renew", "renewal", "auto-renew", "extension", "successive term"),
    "agreement": ("agreement date", "executed on", "signed on", "dated as of"),
    "notice_period": ("notice period", "notice to terminate", "prior written notice", "days notice"),
}


class PredictRequest(BaseModel):
    document_id: str
    ocr_text: str
    document_type: str = "unknown"
    filename: str = "document.pdf"


app = FastAPI(title="Deadline Detection Inference Service")

PREDICTION_REQUESTS = Counter(
    "deadline_inference_requests_total",
    "Total number of inference requests processed by the deadline inference service.",
    labelnames=("mode",),
)
PREDICTION_FAILURES = Counter(
    "deadline_inference_failures_total",
    "Total number of inference failures raised by the deadline inference service.",
)
PREDICTION_EVENTS = Counter(
    "deadline_inference_events_total",
    "Total number of events returned by the deadline inference service.",
    labelnames=("mode",),
)
PREDICTION_LATENCY = Histogram(
    "deadline_inference_latency_seconds",
    "Latency of prediction requests handled by the deadline inference service.",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def classify_sentence(sentence: str) -> str:
    lowered = sentence.lower()
    for event_type, keywords in EVENT_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return event_type
    return "deadline"


def parse_date(text: str) -> Optional[str]:
    try:
        return dateutil_parser.parse(text, dayfirst=False).strftime("%Y-%m-%d")
    except Exception:
        return None


def fallback_predict(document_id: str, sentences: List[str]) -> Dict[str, Any]:
    events: List[Dict[str, Any]] = []

    for sentence in sentences:
        match = DATE_RE.search(sentence)
        if not match:
            continue
        events.append(
            {
                "event_type": classify_sentence(sentence),
                "deadline_date": parse_date(match.group(0)),
                "deadline_type": "explicit",
                "confidence": 0.5,
                "uncertain": True,
                "source_sentence": sentence,
                "class_scores": {"fallback": 1.0},
            }
        )

    return {
        "contract_id": document_id,
        "has_deadline": bool(events),
        "uncertain": bool(events),
        "events": events,
        "mode": "fallback",
    }


@lru_cache(maxsize=1)
def load_predict_module():
    candidates = [
        Path(__file__).resolve().parent / "training" / "src" / "predict.py",
        Path(__file__).resolve().parent / "components" / "training" / "src" / "predict.py",
        Path("/app/training/src/predict.py"),
        Path("/app/components/training/src/predict.py"),
    ]
    module_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if module_path is None:
        return None

    spec = importlib.util.spec_from_file_location("training_predict", module_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_paths_available() -> bool:
    clf_model = os.getenv("CLF_MODEL_PATH", "")
    ner_model = os.getenv("NER_MODEL_PATH", "")
    return bool(clf_model and ner_model and Path(clf_model).exists() and Path(ner_model).exists())


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "predict_module_loaded": load_predict_module() is not None,
        "models_available": model_paths_available(),
    }


@app.post("/predict")
def predict(request: PredictRequest) -> Dict[str, Any]:
    start = perf_counter()
    try:
        sentences = split_sentences(request.ocr_text)
        candidate_sentences = [
            sentence
            for sentence in sentences
            if DATE_RE.search(sentence) or any(keyword in sentence.lower() for values in EVENT_KEYWORDS.values() for keyword in values)
        ]

        if not candidate_sentences:
            result = {
                "contract_id": request.document_id,
                "has_deadline": False,
                "uncertain": False,
                "events": [],
                "mode": "empty",
            }
            PREDICTION_REQUESTS.labels(mode="empty").inc()
            return result

        predict_module = load_predict_module()
        if predict_module is not None and model_paths_available():
            try:
                result = predict_module.predict(
                    sentences=candidate_sentences,
                    clf_model_path=os.environ["CLF_MODEL_PATH"],
                    ner_model_path=os.environ["NER_MODEL_PATH"],
                    contract_id=request.document_id,
                    confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.7")),
                )
                result["mode"] = "model"
                result["candidate_sentences"] = len(candidate_sentences)
                PREDICTION_REQUESTS.labels(mode="model").inc()
                PREDICTION_EVENTS.labels(mode="model").inc(len(result.get("events", [])))
                return result
            except Exception as exc:
                PREDICTION_FAILURES.inc()
                raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

        result = fallback_predict(request.document_id, candidate_sentences)
        result["candidate_sentences"] = len(candidate_sentences)
        result["document_type"] = request.document_type
        result["filename"] = request.filename
        PREDICTION_REQUESTS.labels(mode="fallback").inc()
        PREDICTION_EVENTS.labels(mode="fallback").inc(len(result.get("events", [])))
        return result
    finally:
        PREDICTION_LATENCY.observe(perf_counter() - start)


@app.get("/config")
def config() -> Dict[str, Any]:
    return {
        "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.7")),
        "clf_model_path": os.getenv("CLF_MODEL_PATH", ""),
        "ner_model_path": os.getenv("NER_MODEL_PATH", ""),
    }


@app.post("/dry-run")
def dry_run(request: PredictRequest) -> Dict[str, Any]:
    result = predict(request)
    return {"request": json.loads(request.model_dump_json()), "result": result}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
