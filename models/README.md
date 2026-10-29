# Model assets

Binary model files are generated artifacts and are intentionally ignored until they have been converted or trained and validated. Do not substitute random weights.

Expected files:

- `sid-sony.onnx` — the official Sony SID TensorFlow checkpoint converted by `tools/convert_sid.py`.
- `sid-sony-fp16.onnx` — numerically gated FP16-weight SID for capable WebGPU devices.
- `pseudo-raw-adapter.onnx` — the trained adapter exported by `tools/export_adapter.py`.
- `pseudo-raw-adapter-fp16.onnx` — numerically gated FP16-weight adapter.
- `metadata.json` — hashes, names, parameter count, and validation state consumed by the browser.

`python tools/package_models.py` hashes validated assets and marks them available. The Vite pre-build hook copies only deployment assets into `public/models/`.
