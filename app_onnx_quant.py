from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer
from optimum.onnxruntime import ORTModelForTokenClassification, ORTModelForSequenceClassification
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter
import re

app = FastAPI()

# --- 1. PROMETHEUS METRICS ---
Instrumentator().instrument(app).expose(app)
CONFIDENCE_METRIC = Histogram("model_confidence_score", "Confidence score", buckets=[0.5, 0.7, 0.8, 0.9, 1.0])
PREDICTION_COUNTER = Counter("model_predictions_total", "Total documents processed")
UNCERTAIN_COUNTER = Counter("model_uncertain_predictions", "Count of uncertain predictions flagged for review")

# --- 2. LOAD BOTH MODELS ---
# Model 1: The Sequence Classifier (The Gatekeeper)
clf_model = ORTModelForSequenceClassification.from_pretrained("./onnx_quantized_clf")
clf_tokenizer = AutoTokenizer.from_pretrained("./onnx_quantized_clf")
clf_pipeline = pipeline("text-classification", model=clf_model, tokenizer=clf_tokenizer, top_k=None)

# Model 2: The NER Extractor (The Date Finder)
ner_model = ORTModelForTokenClassification.from_pretrained("./onnx_quantized_ner")
ner_tokenizer = AutoTokenizer.from_pretrained("./onnx_quantized_ner")
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")

class DocumentRequest(BaseModel):
    document_id: str
    ocr_text: str

# Helper function to split text into sentences
def split_into_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]

@app.post("/predict")
async def predict(req: DocumentRequest):
    PREDICTION_COUNTER.inc()
    sentences = split_into_sentences(req.ocr_text)
    
    events = []
    has_conflict = False
    
    for sentence in sentences:
        # STAGE 1: Run Classifier
        clf_results = clf_pipeline(sentence)[0] 
        # Format the top_k results into a dictionary of {label: score}
        class_scores = {res['label']: float(res['score']) for res in clf_results}
        
        # Find the highest scoring class
        top_class = max(class_scores, key=class_scores.get)
        top_score = class_scores[top_class]
        
        # If the gatekeeper says it's NOT a deadline, skip to the next sentence!
        if top_class == "none":
            continue
            
        # STAGE 2: Run NER only on positive sentences
        ner_results = ner_pipeline(sentence)
        extracted_dates = [e["word"] for e in ner_results]
        
        # Calculate combined confidence (average of CLF and NER)
        ner_confidence = float(ner_results[0]["score"]) if ner_results else 1.0
        combined_confidence = (top_score + ner_confidence) / 2
        CONFIDENCE_METRIC.observe(combined_confidence)
        
        # SAFEGUARDING: Check for uncertainty (matching the feedback loop requirements)
        is_uncertain = bool(combined_confidence < 0.7)
        if is_uncertain:
            UNCERTAIN_COUNTER.inc()
            
        events.append({
            "event_type": top_class,
            "deadline_date": extracted_dates[0] if extracted_dates else None,
            "date_candidates": extracted_dates,
            "conflict_flag": len(extracted_dates) > 1,
            "confidence": round(combined_confidence, 4),
            "uncertain": is_uncertain,
            "source_sentence": sentence,
            "class_scores": class_scores
        })
        
        if len(extracted_dates) > 1:
            has_conflict = True

    # Match the exact JSON structure expected by the architecture
    return {
        "contract_id": req.document_id,
        "has_deadline": len(events) > 0,
        "uncertain": any(e["uncertain"] for e in events),
        "multi_date_conflict": has_conflict,
        "events": events
    }