import os
from pathlib import Path

from optimum.exporters.onnx import main_export
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig


BASE_DIR = Path(os.getenv("ONNX_WORKDIR", ".")).resolve()

HF_CLF_SOURCE_PATH = Path(os.getenv("HF_CLF_SOURCE_PATH", "/models/classifier"))
HF_NER_SOURCE_PATH = Path(os.getenv("HF_NER_SOURCE_PATH", "/models/ner"))

ONNX_CLF_EXPORT_PATH = Path(os.getenv("ONNX_CLF_EXPORT_PATH", str(BASE_DIR / "onnx_model_clf")))
ONNX_NER_EXPORT_PATH = Path(os.getenv("ONNX_NER_EXPORT_PATH", str(BASE_DIR / "onnx_model_ner")))

ONNX_CLF_QUANTIZED_PATH = Path(
    os.getenv("ONNX_CLF_QUANTIZED_PATH", str(BASE_DIR / "onnx_quantized_clf"))
)
ONNX_NER_QUANTIZED_PATH = Path(
    os.getenv("ONNX_NER_QUANTIZED_PATH", str(BASE_DIR / "onnx_quantized_ner"))
)


def export_if_needed(source_path: Path, output_path: Path, task: str) -> None:
    if output_path.exists() and any(output_path.iterdir()):
        print(f"Skipping export for {task}; ONNX directory already exists at {output_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {task} model from {source_path} -> {output_path}")
    main_export(
        model_name_or_path=str(source_path),
        output=str(output_path),
        task=task,
    )


def quantize(source_path: Path, output_path: Path, label: str) -> None:
    print(f"Quantizing {label} from {source_path} -> {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    quantizer = ORTQuantizer.from_pretrained(str(source_path))
    quantization_config = AutoQuantizationConfig.avx2(is_static=False)
    quantizer.quantize(save_dir=str(output_path), quantization_config=quantization_config)
    print(f"{label} successfully quantized to INT8 at {output_path}")


def main() -> None:
    export_if_needed(HF_CLF_SOURCE_PATH, ONNX_CLF_EXPORT_PATH, "text-classification")
    export_if_needed(HF_NER_SOURCE_PATH, ONNX_NER_EXPORT_PATH, "token-classification")

    quantize(ONNX_CLF_EXPORT_PATH, ONNX_CLF_QUANTIZED_PATH, "RoBERTa classifier")
    quantize(ONNX_NER_EXPORT_PATH, ONNX_NER_QUANTIZED_PATH, "BERT NER model")


if __name__ == "__main__":
    main()
