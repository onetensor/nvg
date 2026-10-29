# SID Live

SID Live is a static, privacy-preserving camera website that adapts processed Safari video into a four-channel **pseudo-RAW** representation and runs a Sony *Learning to See in the Dark* (SID) U-Net locally. The production site uses the jointly tuned SID; the untouched original checkpoint remains in the repository as a numerically validated reproducibility artifact. Camera frames are never uploaded.

The repository deliberately distinguishes implemented code from validated artifacts. Random or newly initialized weights are never presented as SID. `models/metadata.json` keeps enhancement disabled until the official checkpoint and trained adapter have been converted, measured, hashed, and packaged.

## What is implemented

- Rear-camera capture through `getUserMedia`, with actual settings shown in the debug overlay.
- `requestVideoFrameCallback` scheduling with a one-frame in-flight limit; new frames are dropped instead of queued.
- Direct `HTMLVideoElement` → WebGPU external texture preprocessing.
- Inverse-sRGB linearization and deterministic RGGB synthetic-Bayer packing.
- ONNX Runtime Web/WebGPU inference behind an `InferenceBackend` interface.
- GPU I/O binding: preprocessing, learned adapter, gain, SID, temporal filtering, and display share GPU storage buffers.
- A tiny learned RGB → pseudo-RAW adapter (265,732 parameters at width 64), selected by validation ablation, followed by an optional low-learning-rate joint SID quality pass.
- Logarithmic manual gain from 1× to 300× and a smoothed histogram-based automatic controller.
- Geometry and training tooling from 180p through 1080p; the production iPhone site defaults to 320×180 to keep the full SID network usable on a phone.
- Adaptive resolution, measured rolling timing/FPS, frame counters, split comparison, and motion-aware temporal stabilization.
- Dynamic FP16 joint SID and adapter graphs, reusable GPU tensors, and an iPhone Canvas2D presentation fallback for WebKit's unreliable WebGPU canvas compositor.
- Exact TensorFlow-checkpoint conversion, TensorFlow/PyTorch/ONNX validation, adapter training/export/evaluation, and browser fixture tooling.

## Run locally

```bash
npm install
npm run dev
```

Open the HTTPS URL printed by your host. `localhost` is accepted as a secure context on desktop, but another phone cannot use a plain `http://<LAN-IP>` camera page. Use an HTTPS tunnel or deploy the static build.

Production build:

```bash
npm test
npm run build
npm run preview
```

`dist/` is a conventional static site. Deploy it to Cloudflare Pages, Vercel, GitHub Pages, Replit, or any HTTPS static host. No server, database, account, or API key is used at runtime. The Vite pre-build hook copies only model assets marked available in `models/metadata.json`.

`public/_headers` supplies Cloudflare Pages isolation/camera headers and `vercel.json` supplies their Vercel equivalents. If another host does not serve those files as configuration, set COOP to `same-origin`, COEP to `require-corp`, and allow camera access from `self`. GitHub Pages cannot set custom response headers; verify the ORT build you deploy does not require shared-memory WASM there.

## GitHub Pages

The workflow at `.github/workflows/deploy-pages.yml` tests, builds, and deploys
the site on pushes to `master` or `main`. It also requests Pages enablement for
the repository. For `onetensor/nvg`, the deployed site is:

`https://onetensor.github.io/nvg/`

The Vite build uses relative asset URLs, so the repository subpath works for
the app, validation fixture, benchmark page, WASM bundle, and model files.
The production sync deliberately publishes only the dynamic FP16 joint SID
and FP16 adapter. Those two files are preloaded in parallel and passed directly
to ONNX Runtime; FP32, static-shape, and original-SID graphs are not downloaded.
GitHub Pages provides HTTPS (required for camera access), but not custom
COOP/COEP headers; the deployed browser/device still needs to prove that the
selected ONNX Runtime WebGPU build works without them.

## Camera permissions

Tap **Start rear camera** in response to a user gesture. Safari then presents its camera permission prompt. The constraint prefers the environment camera at 1280×720 and 30 FPS, but the application reads and reports the settings Safari actually grants. Permissions remain controlled by Safari/iOS Settings.

## Model preparation and validation

The reference is [cchen156/Learning-to-See-in-the-Dark](https://github.com/cchen156/Learning-to-See-in-the-Dark), released under the MIT license. Cite:

> Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun. “Learning to See in the Dark.” CVPR 2018.

Install the Python dependencies in an environment with PyTorch and TensorFlow:

```bash
pip install -r requirements-train.txt
```

Convert the official Sony checkpoint:

```bash
python tools/convert_sid.py \
  --checkpoint /path/to/checkpoint/Sony/model.ckpt \
  --output models/sid-sony.pt
```

The converter transposes every TensorFlow Slim convolution and the four anonymous transpose-convolution variables into the exact NCHW PyTorch graph, then exports a dynamic-shape ONNX model. It does not retrain SID.

For a fixed packed official Sony RAW crop, generate a TensorFlow reference and compare all candidates:

```bash
python tools/make_sid_sample.py --raw /path/to/official.ARW --ratio 100 --output test-results/sid/input.npz
python tools/export_reference_outputs.py \
  --checkpoint /path/to/checkpoint/Sony/model.ckpt \
  --input test-results/sid/input.npz \
  --output test-results/sid/reference.npz
python tools/verify_sid.py \
  --reference test-results/sid/reference.npz \
  --pytorch models/sid-sony.pt \
  --onnx models/sid-sony.onnx \
  --output test-results/sid
```

The report contains MAE, maximum absolute error, and PSNR, plus a TensorFlow/PyTorch/ONNX visual strip. `tools/package_models.py` only marks validation successful when TensorFlow-vs-ONNX MAE is below `1e-5` and maximum error is below `5e-4`.

The packaged conversion was checked on the held-out official Sony test pair
`10003_00_0.04s.ARW` → `10003_00_10s.ARW` at its true 250× exposure ratio.
For the fixed center crop, TensorFlow vs ONNX measured MAE
`2.3806e-8`, maximum error `7.1526e-7`, and PSNR `148.45 dB`.
The FP16-weight graph vs FP32 measured MAE `2.3670e-5` and maximum error
`1.7659e-4` on a real packed-RAW crop.

`validation.html` runs generated SID-only, adapter-only, and adapter→gain→SID
fixtures through ONNX Runtime Web's WebGPU provider and reports browser MAE,
maximum error, PSNR, and elapsed time against saved framework references.

## Adapter and optional joint training

The first stage keeps SID in `eval()` with `requires_grad_(False)`. Gradients flow through it only to the adapter:

```text
processed dark RGB → inverse sRGB → adapter → pseudo-RAW × gain → frozen SID → target RGB
```

The initial adapter is:

```text
PixelUnshuffle(2) → Conv 12→64 → 3 residual blocks → Conv 64→4
deterministic pseudo-Bayer + 0.10 × tanh(learned residual) → clamp
```

The final projection is zero-initialized, so step zero is exactly the
deterministic baseline. This avoids beginning near 0.5 pseudo-RAW, which would
saturate after the 30–300× SID gain and suppress useful output-loss gradients.
Width 64 and residual scale 0.10 were retained after a seven-variant validation
ablation; the adapter remains well under the 1M-parameter budget.

Its loss is Charbonnier output reconstruction + `0.2 × (1 - SSIM)` and, when true packed short-exposure RAW exists, `10 × L1(pseudo-RAW, packed RAW)`. The RAW loss is numerically much smaller than the output loss; this weight was selected by ablation and the output-space objective remains primary. Training inputs receive randomized white balance, color mixing, tone/gamma, black/highlight clipping, quantization, and occasional JPEG/chroma damage.

Accepted dataset layouts:

1. Official Sony `short/`, `long/`, and `Sony_{train,val,test}_list.txt` files.
2. `{train,val,test}.jsonl` manifests whose rows contain `input`, `target`, optional `raw`, and `gain` paths/values relative to the dataset root.

The manifest path also accepts aligned PNG/JPEG pairs. When `raw` is omitted,
training uses only the end-to-end reconstruction objective; mix those records
with RAW-backed Sony records to keep the pseudo-RAW latent anchored. Verify the
license of any external dataset before releasing derived weights.

Train and evaluate:

```bash
python -m training.train_adapter \
  --data /path/to/dataset \
  --sid models/sid-sony.pt \
  --output training/runs/adapter

python -m training.evaluate_adapter \
  --data /path/to/dataset \
  --sid models/sid-sony.pt \
  --adapter training/runs/adapter/adapter-latest.pt \
  --output test-results/adapter/evaluation.json

python tools/export_adapter.py \
  --checkpoint training/runs/adapter/adapter-latest.pt \
  --output models/pseudo-raw-adapter.onnx
```

After selecting a frozen-SID adapter checkpoint, the optional quality pass
jointly fine-tunes the adapter and pretrained SID with a much smaller SID
learning rate:

```bash
python -m training.train_joint \
  --data /path/to/cached-dataset \
  --sid models/sid-sony.pt \
  --adapter training/runs/adapter/adapter-best.pt \
  --output training/runs/joint
```

This changes SID weights, so it is packaged as a separately named quality
model. The original converted checkpoint and its equivalence fixtures remain
available in the repository, but are excluded from the production website.

Evaluation reports learned and deterministic-baseline PSNR/SSIM on the same held-out samples. Retain the learned adapter only if it materially improves the frozen SID result.

The selected jointly tuned 265,732-parameter adapter + SID pair measured
`19.0805 dB` / `0.601934` SSIM over 100 held-out patches. The prior best
frozen-SID adapter measured `18.1962 dB` / `0.551309`; deterministic
pseudo-Bayer through the original SID measured `15.5947 dB` / `0.298455`.
For reference, the original frozen SID fed true packed Bayer RAW measured
`29.0610 dB` / `0.774073` on the same patches. Joint tuning improves the
processed-RGB path without claiming to recover physical Bayer measurements.

Package hashes and readiness:

```bash
python tools/package_models.py \
  --models models \
  --sid-validation test-results/sid/metrics.json \
  --sid-fp16-validation test-results/sid/fp16.json \
  --adapter-fp16-validation test-results/adapter/fp16.json \
  --adapter-checkpoint training/runs/adapter/adapter-latest.pt
```

The adapter code in this repository is project code; a trained checkpoint inherits the applicable dataset terms. Record the exact dataset revision and license alongside any released weights.

## Why pseudo-RAW is not RAW

Safari exposes ISP-processed video, not Bayer sensor measurements. Apple’s pipeline has already demosaiced, denoised, sharpened, white-balanced, color-corrected, tone-mapped, clipped, quantized, and possibly temporally combined the image. That process is not invertible.

The four adapter channels are therefore a compact latent interface that frozen SID can interpret. They are called **pseudo-RAW** everywhere and are not claimed to reconstruct physical sensor values. The manual RGGB path exists as a deterministic experimental baseline.

## WebGPU and iPhone status

The application requires a secure context and `navigator.gpu`. It fails closed to Original mode when WebGPU, SID assets, or a compatible execution provider is unavailable.

There is an important platform caveat: the current [ONNX Runtime Web support table](https://onnxruntime.ai/docs/get-started/with-javascript/web.html) does not list WebGPU as supported on Safari/iOS, even though WebGPU availability in Safari itself is evolving. Consequently, code/build equivalence does **not** constitute proof of the primary iPhone target. A real-device pass must confirm session creation, operator coverage, memory use, image quality, and timing on the exact iOS/Safari release. If ORT’s WebGPU provider remains unavailable there, the backend boundary is the intended place for a narrowly scoped custom WGSL SID implementation; the application must not silently call WASM and claim the WebGPU target was met.

## Benchmarks

No device numbers are fabricated. The debug overlay measures the live end-to-end queue completion time and rolling processed FPS, along with CPU-side stage waits. Record results here only after testing a packaged, validated model:

| Device | iOS / Safari | Mode | Resolution | Total ms | Processed FPS | Quality notes |
|---|---|---|---:|---:|---:|---|
| _not measured_ | — | — | — | — | — | — |

The visible 320×180 frame is internally padded to 320×192 and cropped after SID, avoiding invalid U-Net skip shapes without distorting the displayed 16:9 region. A 640×360 quality option remains available through the same dynamic graph.

Production defaults to 320×180, internally padded to 320×192. The prior
640×384 path costs about 27.7 GMAC per frame across adapter + SID; the new
default costs about 6.9 GMAC, almost exactly one quarter. On iPhone WebKit,
graph capture and preallocated ONNX outputs are disabled because those advanced
I/O bindings are not reliably implemented there. The established JSEP WebGPU
path is used instead of ONNX Runtime's newer native WebGPU path. Enhanced frames
are cropped and packed to RGBA8 on the GPU, copied through a three-buffer
asynchronous readback ring, and drawn through Canvas2D to avoid the iOS WebGPU
canvas invisibility/flicker bug. The readback no longer blocks inference, and
the iPhone path imports a `VideoFrame` snapshot instead of the camera-backed
`HTMLVideoElement` because WebKit may return black external-texture samples for
the latter. A one-pixel visible camera edge prevents fully occluded video from
being throttled. Histogram readback runs once
per 30 processed frames, temporal filtering defaults Off, and the frame pump
keeps at most one inference in flight so overload drops frames instead of
building latency.

## Known limitations

- A trained adapter can only generalize as far as its simulated ISP and fine-tuning data. iPhone-specific paired data would likely improve it.
- Camera exposure/ISO metadata is not assumed because Safari does not expose it consistently; automatic gain uses luminance statistics.
- GPU histogram readback occurs every 30 processed frames and is only 1 KiB, but it still introduces an occasional synchronization point.
- Temporal filtering is deliberately lightweight and may ghost; Off or Low is safest for movement.
- Dynamic ONNX shapes trade graph-capture speed for runtime resolution switching. Separate static models may benchmark better on a fixed target.
- The official SID authors explicitly note that the pretrained RAW model does not work directly on ISP-produced JPEG/PNG input. Whether the learned adapter closes enough of that domain gap is the central experiment, not an assumed outcome.

## Repository map

- `src/` — camera scheduling, controls, profiling, WebGPU preprocessing/rendering, and inference abstraction.
- `shaders/` — external-texture preprocessing, pseudo-RAW gain, temporal filtering, and display.
- `training/` — exact SID graph, tiny adapter, dataset/ISP pipeline, loss, training, and evaluation.
- `tools/` — official checkpoint conversion, reference export, numerical verification, ONNX export, hashes, and artifact upload.
- `models/` — metadata plus generated validated model artifacts.
- `tests/` — deterministic application unit tests; generated numerical/visual reports belong in `test-results/`.

## Privacy and licenses

Live inference is entirely client-side. The site sends no frames, telemetry, or camera metadata anywhere. Static hosting naturally downloads JavaScript, WASM, metadata, and model files.

SID’s original implementation is MIT licensed; retain its license and citation with converted weights. ONNX Runtime Web is MIT licensed. Dataset and trained-adapter redistribution terms must be checked and documented for the exact dataset revision used.
