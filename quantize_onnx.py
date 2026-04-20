from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# Apply dynamic INT8 quantization optimized for standard CPUs (AVX2 instructions)
dqconfig = AutoQuantizationConfig.avx2(is_static=False)

# --- 1. Quantize the Classifier (RoBERTa Gatekeeper) ---
print("Quantizing the RoBERTa Classifier...")
# Assuming your export script saved the base ONNX classifier here:
clf_quantizer = ORTQuantizer.from_pretrained("./onnx_model_clf")
clf_quantizer.quantize(save_dir="./onnx_quantized_clf", quantization_config=dqconfig)
print("Classifier successfully quantized to INT8!")

# --- 2. Quantize the NER Model (BERT Extractor) ---
print("Quantizing the BERT NER model...")
# Assuming your export script saved the base ONNX NER model here:
ner_quantizer = ORTQuantizer.from_pretrained("./onnx_model_ner")
ner_quantizer.quantize(save_dir="./onnx_quantized_ner", quantization_config=dqconfig)
print("NER Model successfully quantized to INT8!")