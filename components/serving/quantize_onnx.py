import os

from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig


SOURCE_MODEL_PATH = os.getenv("ONNX_MODEL_SOURCE_PATH", "./onnx_model")
OUTPUT_MODEL_PATH = os.getenv("ONNX_MODEL_OUTPUT_PATH", "./onnx_quantized_model")


def main() -> None:
    quantizer = ORTQuantizer.from_pretrained(SOURCE_MODEL_PATH)
    quantization_config = AutoQuantizationConfig.avx2(is_static=False)
    quantizer.quantize(save_dir=OUTPUT_MODEL_PATH, quantization_config=quantization_config)
    print(f"ONNX model successfully quantized to INT8 at {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    main()
