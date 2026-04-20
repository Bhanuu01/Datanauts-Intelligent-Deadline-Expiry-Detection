## Optimized ONNX Serving

This directory integrates the serving teammate's quantized ONNX inference path into the
shared repo layout without replacing the primary `deadline-inference` service.

- `app_onnx_quant.py`: FastAPI app for the quantized ONNX token classifier
- `quantize_onnx.py`: helper script to convert an ONNX export into an INT8-quantized model
- `deployment_trigger.py`: lightweight promotion helper for the optimized serving path
- `Dockerfile.onnx_quant`: runtime image for the ONNX service

Expected model layout on the shared PVC:

- `/models/onnx_quantized_model`

Optional feedback is appended to:

- `/data/serving_feedback.jsonl`
