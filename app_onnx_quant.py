from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer
from optimum.onnxruntime import ORTModelForTokenClassification

app = FastAPI()

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
    deadlines = [{"extracted_text": e["word"], "event_type": "DUE_DATE", "confidence": float(e["score"])} for e in results]
    return {"document_id": req.document_id, "deadlines": deadlines}
