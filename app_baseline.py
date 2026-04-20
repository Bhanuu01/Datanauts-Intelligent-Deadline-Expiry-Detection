from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
# Dummy model simulating your extraction pipeline
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

class DocumentRequest(BaseModel):
    document_id: str
    ocr_text: str

@app.post("/predict")
async def predict(req: DocumentRequest):
    results = ner(req.ocr_text)
    deadlines = [{"extracted_text": e["word"], "event_type": "DUE_DATE", "confidence": float(e["score"])} for e in results]
    return {"document_id": req.document_id, "deadlines": deadlines}
