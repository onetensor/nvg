# SID Live — Agent Handoff

Last updated: 2026-10-29 (America/Toronto)

This is the operational handoff for continuing work on SID Live. Read this
before changing the camera path, model packaging, training setup, or deployment.
The full product intent remains in [`SPEC.md`](SPEC.md), and the durable user
documentation is in [`README.md`](README.md).

## Executive status

- Repository: `https://github.com/onetensor/nvg`
- Branch: `master`
- Audited implementation baseline: `50de041` (`Harden iPhone camera ingestion pipeline`)
- Live site: `https://onetensor.github.io/nvg/`
- Cache-busted audited build: `https://onetensor.github.io/nvg/?v=50de041`
- Successful Pages run: `https://github.com/onetensor/nvg/actions/runs/40825512720`
- Production assets: dynamic FP16 joint-tuned SID plus dynamic FP16 adapter only
- iPhone result for `50de041` (reported by the user): input is no longer black,
  but the image was rotated 90° counter-clockwise, the bottom half was black,
  and it ran at about 6 FPS although the per-stage timings summed to ~15 ms
  while `total` was ~90 ms. All three are addressed by the follow-up below;
  **the follow-up has not yet been tested on a physical iPhone.**

## Follow-up after the first `50de041` phone report

### Rotation and the black bottom half

WebKit's direct `copyExternalImageToTexture()` of the camera copies the
unrotated landscape sensor buffer (1280×720) while the `<video>` reports upright
portrait dimensions (720×1280). Copying with the portrait size into a portrait
texture filled the top ~56% with a sideways image and left the rest black.

Fix (`src/cameraInput.ts`, `src/preprocessing/gpuPreprocess.ts`,
`shaders/preprocessTexture.wgsl`):

- Direct copies use the buffer size from a short-lived `VideoFrame`
  (`visibleRect`/`coded*`), closed immediately, and the texture matches it.
- The shader takes `quarter_turns` (clockwise) and maps upright UV into the
  buffer; cover scaling is computed on the upright size.
- During startup validation a `thumbnail` entry point averages the copied buffer
  into 16×16 cells; the CPU compares that against an upright 64→16 Canvas2D
  `drawImage` probe at all four quarter turns (`estimateOrientation`). The best
  correlated turn is used. A path is rejected if >10% of lit reference cells are
  black in the copy or correlation is poor; flat scenes are retried up to 45
  samples before falling back to the metadata guess. The console logs the
  verdict, e.g. `video-copy orientation: 90° clockwise (r=…)`.
- Order stays `video-copy` → `video-frame-copy` → `canvas-copy`. Canvas copy
  is correct by construction (it now cover-crops at processing resolution) but
  synchronizes the GPU in Chrome and was measurably slower there.
- `?cameraPath=<path>` forces one path for on-device diagnosis; a forced path
  is never abandoned on an orientation mismatch.

Verified in headless Chrome on a real Intel GPU: all four rotations recovered
(r=0.998), and the exact partial-copy layout from the phone report is rejected
with "44% of the camera frame is missing".

### FPS

Two scheduling bugs, independent of model cost:

1. Inference ran on every camera frame even when all readback slots were busy;
   the renderer then dropped the result. The GPU spent most of its time on
   frames that were never displayed, and latency grew to ~1 s. Now
   `Canvas2DRenderer.ready` (fewer than two frames awaiting display) gates
   `processFrame`; overload skips camera frames before any GPU work.
2. `SIDPipeline.process()` awaited the histogram readback, which waits for the
   whole GPU queue. That was the gap between the stage sum and `total`. Stats
   are now a promise consumed asynchronously by auto gain.

A/B with an iPhone UA and a fake camera on the Intel iGPU, same session:
original build displayed 1.9–2.8 FPS (readback 740–960 ms) while its
"processed" counter claimed 11–15 FPS; the fix displayed 8–14 FPS (readback
120–220 ms) with processed == displayed. These are desktop numbers, not iPhone
numbers.

Stage timings in the overlay are CPU submission times. True GPU cost is visible
as `displayed` FPS (≈ GPU throughput now) and `readback` (≈ 2 × GPU frame
time). ORT kernel profiling at 180p on the Intel iGPU: adapter ≈ 20 ms GPU
(seven 64→64 3×3 convs at 96×160 ≈ 4 GMAC, more than SID), SID ≈ 58 ms
(Conv2DMatMul 50 ms, ConvTranspose 3.7 ms, Concat 3.2 ms). Further FPS on the
phone therefore mostly requires a cheaper adapter/SID, not pipeline changes.

The earlier working tree was clean when the original handoff was written, and
`50de041` was reviewed by a second agent. The follow-up has not had a
second-agent review.

## What the user wants

The user wants the best practical low-light enhancement possible in an iPhone
Safari website, with maximum live FPS. They prefer proactive engineering and
experimentation over repeatedly asking for obvious choices. Relevant explicit
preferences:

- Optimize for as much FPS as possible.
- Use the best jointly trained model; do not make the phone download every model
  variant or both FP32 and FP16.
- Training may use a RunPod RTX PRO 6000 Blackwell.
- If more RunPod work is needed, create a new network volume; do not reuse the
  unrelated pre-existing volume.
- The RunPod/Hugging Face secret is named `HF_TOKEN` and may only resolve inside
  the pod.
- Hugging Face namespace is `verysweatydog/nvg`; the dataset repository was
  empty when inspected. Validated artifacts are uploaded to the private model
  repository of the same name/type.
- The user expects changes to be audited thoroughly before another phone test.

Do not expose `HF_TOKEN` in logs, source, commits, or documentation.

## Current product architecture

The production path is:

```text
iPhone rear camera
  -> browser video frame
  -> inverse-sRGB linear RGB preprocessing
  -> 265,732-parameter pseudo-RAW adapter
  -> exposure gain
  -> jointly fine-tuned Sony SID U-Net
  -> GPU RGBA8 packing
  -> asynchronous GPU readback ring
  -> visible Canvas2D output on iPhone
```

All model execution uses ONNX Runtime Web 1.23.2 with its WebGPU execution
provider. Preprocessing, adapter, gain, SID, and output packing use buffers on
the ORT WebGPU device and queue.

On non-iPhone WebGPU browsers, preprocessing may use a `GPUExternalTexture` and
output may use the WebGPU canvas. On Apple mobile WebKit, the compatibility
path deliberately uses a copied regular texture for camera input and Canvas2D
for visible presentation.

### Deployed model selection

[`tools/sync-web-models.mjs`](tools/sync-web-models.mjs) deliberately publishes
only:

- `sid-sony-joint-fp16.onnx` — about 15.5 MB
- `pseudo-raw-adapter-fp16.onnx` — about 541 KB
- deployment `metadata.json`

The original SID, FP32 graphs, and static-shape graphs remain tracked in
`models/` for reproducibility but are not copied to `public/models/` or
downloaded by production. Both deployed `sid` and `sidJoint` metadata keys
resolve to the selected joint FP16 graph.

## Latest iPhone incident

The phone initially loaded the model but showed a green/flat enhanced canvas.
The device was clearly doing GPU work and getting warm. Representative pre-fix
debug output was:

```text
camera          720x1280  27.6 fps
processing      320x180 (pad 320x192)
processed       20.4 fps
preprocess      0.20 ms
adapter         1.71 ms
SID             2.22 ms
postprocess     0.00 ms
total           11.44 ms
SID gain        29.9x
input           0.0000 mean / p50 / p90
black           100.0%
output          0.047 min / 0.069 mean / 0.110 max
```

An earlier build spent roughly 54 ms in postprocessing and ran around 14 FPS.
Moving output to an asynchronous readback ring removed that wait, but the input
remained exactly black. A subsequent `VideoFrame` lifetime change held frames
until `queue.onSubmittedWorkDone()` and caused a severe regression to roughly
2 FPS while still producing black input. That behavior was deployed in
`cdccaa9` and must not be reintroduced.

The high automatic gain (sometimes approximately 300x) was a consequence of
the all-zero input, not evidence that gain control itself was the root cause.

## What `50de041` changed

### Camera input recovery

Apple mobile WebKit now tries these input paths automatically:

1. direct `HTMLVideoElement` -> `copyExternalImageToTexture()`
2. `VideoFrame(video)` -> `copyExternalImageToTexture()`
3. Canvas2D staging -> `copyExternalImageToTexture()`

The ordering is in [`src/cameraInput.ts`](src/cameraInput.ts), and the
implementation is in
[`src/preprocessing/gpuPreprocess.ts`](src/preprocessing/gpuPreprocess.ts).

Important invariants:

- The copied camera texture must have `COPY_DST | TEXTURE_BINDING |
  RENDER_ATTACHMENT`. WebGPU requires `RENDER_ATTACHMENT` for
  `copyExternalImageToTexture()`, even though this code never renders into that
  texture.
- A `VideoFrame` used as the copy source is closed immediately after the copy
  is issued. The external source is captured by the copy operation at call
  time. Do **not** keep it alive through `queue.onSubmittedWorkDone()`; that
  starves Safari's camera decoder and caused the 2 FPS regression.
- The first use of each copy source is wrapped in an awaited WebGPU validation
  error scope.
- The first preprocessing bind/dispatch is also wrapped in an awaited
  validation scope. This is a one-time startup cost, not a per-frame wait.
- The copied-texture shader is
  [`shaders/preprocessTexture.wgsl`](shaders/preprocessTexture.wgsl). The
  external-texture shader remains [`shaders/preprocess.wgsl`](shaders/preprocess.wgsl).
- WGSL integer-to-float conversions must stay explicit, such as
  `vec2f(vec2u(...))`. The previous non-standard `vec2f(u32, u32)` constructors
  failed strict Chrome WebGPU shader compilation and could silently leave the
  model input at zero on permissive/error-obscuring implementations.

### Silent-black detection

The preprocessing histogram has 257 counters:

- bins 0-255: linear luminance histogram
- bin 256: pixels whose RGB is exactly `(0, 0, 0)`

During initial validation, a 16x16 Canvas2D CPU probe checks whether the video
contains nonzero pixels. A path that produces essentially 100% exact-zero GPU
input advances to the next source. If every path fails, the UI reports an
explicit error rather than silently running SID on zero input.

The debug overlay reports:

- `camera path`: `video-copy`, `video-frame-copy`, or `canvas-copy`
- `exact zero`: percentage of exact-zero input samples
- luminance mean, p50, p90, and near-black percentage

### Geometry

Portrait cameras now use portrait processing geometry:

- 180p: `180x320`, padded to `192x320`
- 360p: `360x640`, padded to `384x640`

Landscape remains `320x180`/`640x360`. Cover-style UV scaling prevents the
camera image from being squeezed. See [`src/geometry.ts`](src/geometry.ts).
Production currently ships dynamic-shape models, so portrait tensor shapes are
supported. If static 360p graphs are ever redeployed, they are landscape-shaped
and must not be selected for portrait input without separate portrait exports.

### Output presentation

The iPhone path packs the SID float output to RGBA8 on the GPU and uses three
asynchronous MAP_READ slots. Inference does not await presentation. The renderer
drops a display request when every slot is busy and suppresses stale mapped
frames by sequence number. See [`src/renderer.ts`](src/renderer.ts).

New presentation diagnostics:

- `displayed`: actual Canvas2D presentation FPS
- `readback`: submission-to-presentation latency
- `display drop`: cumulative frames skipped because all readback slots were busy

`non-finite` is shown as an em dash on this path because RGBA8 packing cannot
truthfully diagnose non-finite float values. Do not change it back to a fake
zero.

The camera video keeps a one-pixel visible edge in Enhanced mode because Safari
may throttle a fully occluded media element.

## Validation already completed for `50de041`

Do not repeat these checks unless relevant code changes:

- `git diff --check` passed.
- `npx tsc --noEmit` passed.
- `npm test -- --run` passed: 14/14 tests across four files.
- `npm run build` passed.
- All six WGSL modules compiled with zero diagnostics in an actual headless
  Chrome WebGPU device.
- A synthetic Canvas -> `copyExternalImageToTexture` -> copied-texture shader ->
  readback test passed. An RGB `(128, 64, 32)` source produced approximately
  `(0.21586, 0.05127, 0.01444)`, matching inverse sRGB; all 16 samples reached
  the histogram and exact-zero count was zero.
- A second agent reviewed camera texture flags, validation scopes, fallback
  order, frame lifetime, histogram health checks, orientation, WGSL binding
  layout, async output capacity/drop handling, and shared-queue ordering. It
  found no remaining P0/P1 blocker.
- GitHub Pages build and deployment succeeded for `50de041`.
- The live hashed bundle was inspected and contains `video-copy`,
  `video-frame-copy`, `canvas-copy`, exact-zero failure handling, preprocessing
  validation, and `RENDER_ATTACHMENT` usage.
- Both intended FP16 model URLs return HTTP 200 from the live site.

Only a physical Safari/WebGPU run can close the remaining platform risk.

## First action for the next agent

Ask the user to open the cache-busted build of the latest deployed commit
(`https://onetensor.github.io/nvg/?v=<commit>`). In addition to the fields
below, the console line `<path> orientation: …` (or `… rejected: …`) shows
which camera path was chosen and why.

They should fully close the old tab first to prevent an old module graph from
remaining in memory. Have them enable Debug and provide either a screenshot or
these fields after several seconds:

```text
camera
processing
processed
preprocess / adapter / SID / postprocess / total
camera path
SID gain
input mean / p50 / p90
black
exact zero
output min / mean / max
displayed
readback
display drop
frames received / processed / skipped
```

Interpret the result as follows:

| Observation | Meaning / next move |
|---|---|
| `exact zero < 100%` and a visible image | Camera ingestion is recovered. Focus only on quality/FPS tuning. |
| Path settles on `video-copy` | Best expected camera path; measure thermals and display FPS. |
| Path settles on `video-frame-copy` | Direct video copy is broken on that Safari, but the GPU-copy fallback works. |
| Path settles on `canvas-copy` | Both direct media copy forms are broken; expect more CPU cost and consider a target-resolution canvas optimization. |
| Explicit “every camera upload path” error | Capture Safari console/device/iOS version; the self-test has localized the problem to media ingestion. |
| Nonzero input, processed FPS high, displayed FPS low | Presentation/readback bottleneck, not model inference. Inspect `readback` and `display drop`. |
| Camera FPS itself collapses | Media decoder starvation, visibility throttling, thermal throttling, or an accidental queue wait. Search for new per-frame `onSubmittedWorkDone()` waits first. |
| Output remains flat with nonzero input | Inspect gain, adapter/SID tensor contents, and output packer; camera ingestion is no longer the primary suspect. |

If another iteration is required, add a query-controlled diagnostic override for
the input path rather than making three separate deployments. For example,
`?cameraPath=canvas-copy` could force one path while retaining automatic mode as
the default. Keep it out of the normal UI unless it proves useful.

## Remaining runtime caveats and likely optimizations

These are non-blocking but are the most plausible next engineering work:

1. The direct media-copy paths upload the full `720x1280` camera frame (roughly
   3.7 MB) before sampling it down to 180p/360p. This is still GPU-native and
   should be tested before optimizing. If it is materially expensive, the
   Canvas fallback can crop/resize to the processing target, but that trades GPU
   bandwidth for CPU/media conversion work.
2. The Canvas fallback currently stretches unusual source aspect ratios into
   its staging size. Current iPhone 9:16 input matches the portrait targets, so
   this is not a blocker. If generalized, perform a center-cover crop in
   `drawImage` before the copy.
3. Orientation is selected from camera settings at startup. Rotation while the
   stream is running is not yet a first-class state transition.
4. iPhone presentation necessarily crosses GPU -> CPU -> Canvas2D because the
   direct WebGPU canvas was not visible reliably. If mapped readback dominates,
   evaluate smaller display resolution, fewer presented frames than inference
   frames, or a proven Safari compositor workaround. Do not restore a blocking
   per-frame readback.
5. Histogram readback is about 1 KiB and occurs once every 30 validated frames,
   but it is still an occasional synchronization point.
6. The current Pages workflow emits a non-blocking warning that some GitHub
   actions targeting Node 20 are being forced to Node 24. The workflow itself
   completed successfully.

## Model and training handoff

### Adapter architecture

[`training/adapter_model.py`](training/adapter_model.py) defines:

```text
linear RGB
  -> PixelUnshuffle(2)                 # 12 packed channels
  -> 3x3 Conv (12 -> 64)
  -> two 3x3 residual blocks
  -> 3x3 Conv (64 -> 64)
  -> one 3x3 residual block
  -> 1x1 Conv (64 -> 4)
  -> 0.10 * tanh residual
  + deterministic RGGB pseudo-Bayer
  -> clamp [0, 1]
```

It has 265,732 parameters. The final projection is zero-initialized, making
step zero exactly the deterministic baseline. PixelUnshuffle(2) matches the
2x2 RGGB cell; there is no reason to force every learned convolution to a
“Bayer 4x4” kernel. Stacked 3x3 layers already give a much larger effective
context (maximum path is about 17x17 in packed coordinates, approximately
34x34 source pixels) at better parameter/compute efficiency than one huge
kernel.

### Training data and procedure

- Actual selected training data: official Sony SID `Sony2025.zip`.
- Integrity target: 2,928 ARW files.
- Model card provenance: all 1,865 training records with four cached crops per
  record for the selected run.
- The Hugging Face dataset `verysweatydog/nvg` was empty; use
  [`training/runpod_dataset.sh`](training/runpod_dataset.sh) to fetch the
  official archive when needed.
- The low source-record throughput seen during uncached preparation was RAW
  decoding, augmentation, patch creation, and end-to-end backpropagation
  through SID—not evidence that the 266K adapter itself was computationally
  large. Cached `.npz` patches, multiple workers, pinned memory, BF16, TF32, and
  cuDNN benchmarking are already supported.
- Frozen-SID training objective: Charbonnier output loss + `0.2 * (1 - SSIM)` +
  `10 * L1(pseudo-RAW, packed RAW)` when RAW is available.
- Selected frozen adapter variant: width 64, residual scale 0.10, learning rate
  `1e-5`.
- A subsequent joint pass fine-tuned adapter and SID, with defaults `3e-6` for
  adapter and `3e-7` for SID. This is the model deployed in production. It is a
  low-learning-rate full SID fine-tune, not a from-scratch SID replacement.

### Quality results

On 100 held-out patches, the selected joint pair measured:

| Path | PSNR | SSIM |
|---|---:|---:|
| Joint adapter + joint SID | 19.0805 dB | 0.601934 |
| Deterministic pseudo-Bayer + joint SID | 16.0266 dB | 0.337266 |
| True packed Bayer RAW + joint SID | 29.1597 dB | 0.772826 |

For the original frozen SID comparison in [`README.md`](README.md):

| Path | PSNR | SSIM |
|---|---:|---:|
| Prior best frozen-SID adapter | 18.1962 dB | 0.551309 |
| Deterministic pseudo-Bayer + original SID | 15.5947 dB | 0.298455 |
| True packed Bayer RAW + original SID | 29.0610 dB | **0.774073** |

The true-Bayer number is the answer to “what is the SSIM of the original model
with Bayer RAW input?” in the evaluation used here: `0.774073`.

### External data

The user suggested
`ARM4588/Lowlight-Smartphone-Dataset`. It has **not** been incorporated. Before
using it:

- verify licensing and redistribution terms;
- inspect whether pairs are aligned and whether exposure metadata is usable;
- do not treat processed smartphone images as true Bayer supervision;
- use output-only reconstruction for records without packed RAW;
- mix with Sony RAW-backed examples so the pseudo-RAW latent remains anchored;
- use a held-out device/domain split to avoid reporting memorized scene gains.

Additional diverse, licensed processed-camera data could improve device/domain
robustness, but runtime correctness on the phone should be proven first.

### RunPod notes

Scripts assume paths under `/workspace` and can be overridden with environment
variables. Relevant entry points:

- [`training/runpod_prepare.sh`](training/runpod_prepare.sh)
- [`training/runpod_dataset.sh`](training/runpod_dataset.sh)
- [`training/runpod_validate.sh`](training/runpod_validate.sh)
- [`training/runpod_train.sh`](training/runpod_train.sh)
- [`training/run_adapter_ablations.sh`](training/run_adapter_ablations.sh)
- [`training/run_raw_supervision_ablations.sh`](training/run_raw_supervision_ablations.sh)
- [`training/package_selected_adapter.sh`](training/package_selected_adapter.sh)

The live pod/volume IDs are not stored in the repository. Inspect the RunPod
account before provisioning anything. If a new training run is authorized, use
an RTX PRO 6000 Blackwell and create a new network volume rather than attaching
the unrelated existing volume. Confirm `HF_TOKEN` from inside the pod; the
scripts contain a `/proc/1/environ` fallback for RunPod secret injection.

Packaging refuses to upload to a public Hugging Face repository. The intended
artifact target is private model repo `verysweatydog/nvg`.

## Code map

| Area | Primary files |
|---|---|
| App orchestration | `src/main.ts`, `src/camera.ts`, `src/platform.ts` |
| Camera fallback and health | `src/cameraInput.ts`, `src/preprocessing/gpuPreprocess.ts` |
| Preprocessing shaders | `shaders/preprocess.wgsl`, `shaders/preprocessTexture.wgsl` |
| ONNX WebGPU bridge | `src/inference/onnxWebgpu.ts`, `src/inference/sid.ts` |
| Adapter gain stage | `src/inference/adapter.ts`, `shaders/gain.wgsl` |
| Output | `src/renderer.ts`, `shaders/readback.wgsl`, `shaders/postprocess.wgsl` |
| Profiling/debug | `src/profiler.ts`, `src/types.ts` |
| Geometry | `src/geometry.ts` |
| Model packaging | `tools/sync-web-models.mjs`, `tools/package_models.py` |
| Training | `training/adapter_model.py`, `training/train_adapter.py`, `training/train_joint.py` |
| Deployment | `.github/workflows/deploy-pages.yml`, `vite.config.ts` |

## Routine commands

```bash
# Fast correctness gate
git diff --check
npx tsc --noEmit
npm test -- --run
npm run build

# Local development
npm run dev

# Inspect exactly what production will ship
find public/models -maxdepth 1 -type f -printf '%f %s\n' | sort
find dist/models -maxdepth 1 -type f -printf '%f %s\n' | sort

# Deployment status
gh run list --repo onetensor/nvg --limit 5

# Current repository state
git status --short
git log -8 --oneline --decorate
```

`npm run build` invokes the model sync and rewrites `public/models/` to the
production FP16-only selection. Do not manually add all tracked graphs to
`public/models/`.

## Guardrails

- Preserve user changes and inspect a dirty worktree before editing.
- Do not report simulated or desktop FPS as iPhone FPS.
- Do not call pseudo-RAW physical Bayer reconstruction.
- Do not silently fall back to CPU/WASM while claiming WebGPU execution.
- Do not deploy unvalidated/random weights.
- Do not restore per-frame `queue.onSubmittedWorkDone()` on iPhone.
- Do not start inference while `SIDPipeline.ready` is false, and do not await
  histogram statistics inside a frame; both serialize or waste GPU work.
- Do not close a `GPUExternalTexture`-backing `VideoFrame` early if the code is
  changed back to external-texture import; the current code avoids this lifetime
  problem by copying into a regular texture.
- Keep copied-camera texture usage flags and strict WGSL casts intact.
- Keep the production download set to the selected joint FP16 SID and FP16
  adapter unless the user explicitly requests a different experiment.
- After a runtime change, have a second agent review the diff before asking the
  user for another physical phone test.

## Suggested priority order

1. Obtain and interpret the first physical-iPhone report for `50de041`.
2. If input is fixed, measure actual displayed FPS and thermals before changing
   the ingestion pipeline again.
3. Localize any remaining bottleneck using the new `camera path`, `exact zero`,
   `displayed`, `readback`, and `display drop` fields.
4. Only after runtime correctness is stable, consider broader smartphone data,
   additional joint fine-tuning, or architectural experiments.
5. Record real device/iOS/Safari results in the README benchmark table.

