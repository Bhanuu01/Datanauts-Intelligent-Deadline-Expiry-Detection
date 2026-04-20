from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# Point it to your existing uncompressed ONNX model
quantizer = ORTQuantizer.from_pretrained("./onnx_model")

# Apply dynamic INT8 quantization optimized for standard CPUs (AVX2 instructions)
dqconfig = AutoQuantizationConfig.avx2(is_static=False)

# Save the new hybrid model to a new folder
quantizer.quantize(save_dir="./onnx_quantized_model", quantization_config=dqconfig)
print("ONNX Model successfully quantized to INT8!")
