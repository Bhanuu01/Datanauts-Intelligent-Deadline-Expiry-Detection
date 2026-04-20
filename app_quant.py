import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

app = FastAPI()

# Load the base model
model_id = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForTokenClassification.from_pretrained(model_id)

# COMPRESSION: Dynamically quantize the model layers to 8-bit integer (INT8)
quantized_model = torch.quantization.quantize_dynamic(
    base_model, {torch.nn.Linear}, dtype=torch.qint8
)

ner = pipeline("ner", model=quantized_model, tokenizer=tokenizer, aggregation_strategy="simple")

class DocumentRequest(BaseModel):
    document_id: str
    ocr_text: str

@app.post("/predict")
async def predict(req: DocumentRequest):
    results = ner(req.ocr_text)
    deadlines = [{"extracted_text": e["word"], "event_type": "DUE_DATE", "confidence": float(e["score"])} for e in results]
    return {"document_id": req.document_id, "deadlines": deadlines}
