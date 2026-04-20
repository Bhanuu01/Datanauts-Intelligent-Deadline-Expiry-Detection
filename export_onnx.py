from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoTokenizer

model_id = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = ORTModelForTokenClassification.from_pretrained(model_id, export=True)

tokenizer.save_pretrained("./onnx_model")
model.save_pretrained("./onnx_model")
