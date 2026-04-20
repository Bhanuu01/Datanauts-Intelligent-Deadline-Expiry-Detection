from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer
from optimum.onnxruntime import ORTModelForTokenClassification
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter

app = FastAPI()

# 1. Standard Operational Monitoring (Latency, Requests)
Instrumentator().instrument(app).expose(app)

# 2. Custom Model Output Monitoring
CONFIDENCE_METRIC = Histogram(
    "model_confidence_score", 
    "Confidence score of the extracted deadline",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
)
PREDICTION_COUNTER = Counter("model_predictions_total", "Total predictions made")

# Load the hybrid Quantized ONNX model
model = ORTModelForTokenClassification.from_pretrained("./onnx_quantized_model")
tokenizer = AutoTokenizer.from_pretrained("./onnx_quantized_model")
ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

class DocumentRequest(BaseModel):
    document_id: str
    ocr_text: str

@app.post("/predict")
async def predict(req: DocumentRequest):
    results = ner(req.ocr_text)
    
    # Calculate average or max confidence for the metrics
    confidence = float(results[0]["score"]) if results else 0.0
    
    # Track the metrics
    PREDICTION_COUNTER.inc()
    if confidence > 0:
        CONFIDENCE_METRIC.observe(confidence)
    
    deadlines = [{"extracted_text": e["word"], "event_type": "DUE_DATE", "confidence": float(e["score"])} for e in results]
    return {"document_id": req.document_id, "deadlines": deadlines}


class FeedbackRequest(BaseModel):
    document_id: str
    original_prediction: str
    user_corrected_value: str

# Track how often users have to correct the model
CORRECTION_COUNTER = Counter("model_user_corrections", "Count of user overrides")

@app.post("/feedback")
async def capture_feedback(req: FeedbackRequest):
    CORRECTION_COUNTER.inc()
    
    # In the final integrated system, this data should be written to 
    # the Data team's database so they can use it for retraining!
    print(f"Feedback logged for doc {req.document_id}: User corrected '{req.original_prediction}' to '{req.user_corrected_value}'")
    
    return {"status": "feedback logged for retraining"}
