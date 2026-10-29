# Project: Live “Learning to See in the Dark” iPhone Website

## Objective

Build a website that opens in iPhone Safari, requests the rear camera, and displays a live enhanced low-light view using the original pretrained **Learning to See in the Dark (SID)** model.

The project should first prove that the official SID model can run correctly in the browser. It should then adapt Safari's processed camera frames to SID's expected 4-channel RAW-like representation.

The primary adaptation strategy should ultimately be a **small learned RGB/YUV → pseudo-RAW adapter** trained while keeping SID frozen.

The final runtime should therefore be approximately:

```text
iPhone camera
      ↓
Safari RGB/YUV frame
      ↓
small learned pseudo-RAW adapter
      ↓
4-channel SID-compatible representation
      ↓
pretrained frozen SID
      ↓
enhanced RGB
      ↓
live WebGPU display
```

The goal is not to recover physically exact Bayer RAW measurements.

The goal is to learn a compact intermediate representation that makes the pretrained SID network useful on iPhone browser video.

Do not train a replacement SID network in the first version.

Do not use SIED diffusion.

Do not require:

- native iOS development;
- Xcode;
- a Mac;
- an Apple Developer account;
- App Store distribution;
- server-side inference.

The deliverable should be an ordinary HTTPS website deployable to Replit, Cloudflare Pages, GitHub Pages, Vercel, or similar.

---

# Primary success criteria

The finished system should:

1. Work in current iPhone Safari with WebGPU.
2. Request and display the rear camera.
3. Perform all live inference locally on the iPhone.
4. Run the original pretrained Sony SID model.
5. Validate browser SID output against the original implementation.
6. Support a simple deterministic RGB → pseudo-Bayer baseline.
7. Support a learned lightweight RGB → pseudo-RAW adapter.
8. Keep SID frozen when training the adapter.
9. Display the enhanced result live.
10. Provide Original / Enhanced comparison.
11. Report measured inference time and FPS.
12. Support multiple internal processing resolutions.
13. Drop stale frames rather than accumulating latency.
14. Be deployable as a normal static HTTPS website.

Performance targets:

```text
Ideal:
~720p neural processing at ~30 FPS

Acceptable:
~720p at ~15 FPS

Fallback:
960×540 or 640×360 inference
upscaled for display
```

Image quality is more important than guaranteeing exactly 30 FPS.

---

# Architecture overview

The intended final system is:

```text
                       iPhone Safari

                            │
                            ▼
                   getUserMedia()
                            │
                            ▼
                   camera RGB/YUV
                            │
                            ▼
                GPU preprocessing
                            │
                            ▼
             ┌───────────────────────┐
             │ pseudo-RAW adapter    │
             │                       │
             │ small learned CNN     │
             │ RGB → 4 channels      │
             └───────────────────────┘
                            │
                            ▼
                SID-compatible tensor
                            │
                            ▼
             ┌───────────────────────┐
             │ pretrained Sony SID   │
             │                       │
             │ frozen ~7.8M U-Net    │
             └───────────────────────┘
                            │
                            ▼
                     enhanced RGB
                            │
                            ▼
                       WebGPU canvas
```

The pseudo-RAW adapter should be substantially smaller than SID.

Target:

```text
Adapter:
< 1M parameters preferred

SID:
~7.8M parameters

Total:
roughly 8–9M parameters or less
```

---

# Important conceptual constraint

The original SID network was trained on packed RAW Bayer sensor measurements.

Safari's normal video camera API provides ISP-processed video.

Therefore this mapping is impossible to invert perfectly:

```text
RAW Bayer
    ↓
Apple ISP
    ↓
processed camera video
```

The ISP may perform:

- demosaicing;
- denoising;
- sharpening;
- white balance;
- color correction;
- tone mapping;
- black-level clipping;
- dynamic-range compression;
- quantization;
- temporal processing.

Some information may be irreversibly lost.

Therefore the project must **not** define success as:

> reconstruct the exact original Bayer RAW frame.

Instead define success as:

> produce a compact 4-channel latent representation that lies close enough to SID's expected input domain for frozen SID to reconstruct a useful bright image.

Call this representation:

**pseudo-RAW**

throughout the code and documentation.

---

# Development phases

Implement the project in the following order.

Do not skip directly to adapter training.

---

# Phase 1 — faithful SID validation

Before working on live camera input, establish that the original pretrained SID model works correctly outside its original TensorFlow codebase.

Use the official Sony SID implementation and pretrained weights.

Reference repository:

```text
cchen156/Learning-to-See-in-the-Dark
```

Use official Sony test RAW files.

The validation path is:

```text
official Sony RAW
      ↓
official Bayer packing
      ↓
original TensorFlow SID
      ↓
reference RGB output
```

versus:

```text
same official Sony RAW
      ↓
same Bayer packing
      ↓
converted SID implementation
      ↓
candidate RGB output
```

Verify numerical equivalence before trying browser inference.

Measure:

```text
mean absolute error
maximum absolute error
PSNR between outputs
SSIM if useful
```

Save visual comparisons.

---

# Phase 2 — browser SID execution

Get the validated SID model running in the browser.

Preferred first attempt:

```text
TensorFlow checkpoint
      ↓
conversion
      ↓
ONNX
      ↓
ONNX Runtime Web / WebGPU
```

If this works reliably on the actual target iPhone, keep it.

Do not write a complete custom neural runtime prematurely.

If a runtime compatibility or performance problem makes that approach unusable, implement only the necessary model operations using WebGPU/WGSL.

Use an inference abstraction:

```text
interface InferenceBackend {
    initialize()
    run(input)
    dispose()
}
```

Possible implementations:

```text
ONNXWebGPUBackend
CustomWebGPUBackend
```

The rest of the application must not depend strongly on a particular inference runtime.

---

# Phase 3 — live-camera pipeline

Use:

```javascript
navigator.mediaDevices.getUserMedia()
```

Prefer:

```text
facingMode: environment
1280×720
30 FPS
```

but query actual resulting settings after opening the stream.

Use:

```javascript
video.requestVideoFrameCallback(...)
```

for frame scheduling.

Do not queue frames.

If inference is still running when a new frame arrives:

```text
discard old/stale frame
process newest available frame
```

Live latency is more important than processing every frame.

---

# Phase 4 — deterministic pseudo-Bayer baseline

Before training anything, implement a simple deterministic adapter.

This gives a baseline and proves the complete live pipeline.

Start with approximately:

```text
camera RGB
    ↓
inverse sRGB-style transfer / approximate linearization
    ↓
pixel-unshuffle 2×
    ↓
12 channels
    ↓
select/average into:
R
G1
B
G2
    ↓
4-channel pseudo-RAW
```

Alternative equivalent method:

```text
RGB
 ↓
construct synthetic RGGB mosaic
 ↓
pack 2×2 Bayer cells
 ↓
4 channels
```

Support an adjustable SID input gain.

Suggested manual test values:

```text
10×
30×
50×
100×
200×
300×
```

This deterministic baseline is not expected to be optimal.

Its purpose is to establish:

```text
camera
→ preprocessing
→ SID
→ live output
```

before adding training.

---

# Phase 5 — learned pseudo-RAW adapter

Once the deterministic baseline works, implement the primary adaptation strategy:

```text
processed RGB
      ↓
small neural adapter A
      ↓
4-channel pseudo-RAW
      ↓
frozen pretrained SID S
      ↓
enhanced image
```

SID must remain frozen during adapter training.

Only train the adapter initially.

---

# Adapter architecture

Keep the adapter small.

Preferred starting architecture:

```text
input RGB: H×W×3

PixelUnshuffle(2)

H/2 × W/2 × 12

Conv 3×3:
12 → 32

LeakyReLU

Residual block:
32 → 32 → 32

Residual block:
32 → 32 → 32

Conv 3×3:
32 → 32

Residual block:
32 → 32 → 32

Conv 1×1 or 3×3:
32 → 4

output transform / clamp
```

This should remain comfortably below ~500k parameters.

If quality is clearly insufficient, permit a larger version:

```text
12 → 32 → 64 → 64 → 32 → 4
```

Target:

```text
< 1M adapter parameters
```

unless benchmarks clearly justify increasing it.

Do not build a full inverse-ISP model before testing this small adapter.

---

# Adapter output

The adapter outputs:

```text
B × 4 × H/2 × W/2
```

representing SID-compatible pseudo-RAW channels.

The output does not need to correspond exactly to physical:

```text
R
G1
B
G2
```

unless adding that constraint improves results.

The adapter is allowed to learn any 4-channel representation that frozen SID can interpret effectively.

This is intentional.

---

# Adapter training objective

Use frozen SID as part of the training graph:

```text
dark processed input x

        ↓

adapter A(x)

        ↓

pseudo-RAW

        ↓

frozen SID S(A(x))

        ↓

predicted clean image

        ↓

compare to target y
```

The core output-space loss is:

```text
L_output = |S(A(x)) - y|
```

Start with L1 or Charbonnier reconstruction loss.

Suggested initial loss:

```text
L =
1.0 × Charbonnier(predicted_RGB, target_RGB)
+
0.2 × (1 - SSIM(predicted_RGB, target_RGB))
```

Do not initially use:

- GAN loss;
- diffusion loss;
- adversarial training;
- heavy perceptual losses.

Prioritize stable reconstruction.

---

# Optional RAW supervision

If paired or approximately paired true RAW data are available, add:

```text
L_raw =
|A(x_processed) - packed_raw|
```

Then train:

```text
L_total =
λ_output × L_output
+
λ_raw × L_raw
```

Suggested starting point:

```text
λ_output = 1.0
λ_raw = 0.1
```

Tune experimentally.

The downstream enhanced RGB result remains the primary goal.

Do not force the adapter to exactly recreate sensor RAW if doing so harms SID output.

---

# Distillation interpretation

Treat training conceptually as:

```text
adapter = learned domain translator

processed iPhone-like video domain
             ↓
SID RAW domain
```

SID acts as a frozen reconstruction prior/decoder.

The adapter should learn only enough inverse-ISP/domain transformation to make SID useful.

---

# Training data strategy

For the first experiment, do not require a custom iPhone dataset.

Use existing datasets that provide some combination of:

- RAW;
- processed RGB;
- low-light input;
- clean/reference image.

Create training tuples approximately like:

```text
dark processed RGB input
true/approximate RAW input if available
clean reference RGB
```

If a dataset contains RAW short exposure and long-exposure ground truth:

1. process the short RAW through a realistic ISP pipeline to create the adapter's RGB input;
2. retain the original packed short RAW as optional RAW supervision;
3. use the long-exposure SID target as output supervision.

This gives:

```text
short RAW
   ├────────────→ true packed RAW
   │
   └→ simulated ISP → compressed/processed RGB
                            ↓
                         adapter
                            ↓
                        pseudo-RAW
                            ↓
                           SID
                            ↓
                 long-exposure target RGB
```

This is ideal for initial adapter training.

---

# Simulated camera/ISP degradation

The adapter should not only train on pristine processed images.

Randomize the forward ISP approximation to make it robust.

Possible augmentations:

```text
white balance changes
gamma/tone curves
color matrices
local contrast
sharpening
denoising
quantization
8-bit / 10-bit conversion
JPEG-style compression
chroma subsampling
shadow clipping
black-level changes
highlight clipping
noise reduction
```

The goal is not to precisely emulate Apple initially.

The goal is to prevent the adapter from depending on one synthetic ISP.

---

# Optional iPhone-specific fine-tuning

Do not make this mandatory for v1.

But structure training so later it is easy to fine-tune the adapter using data from the actual target phone.

If suitable paired iPhone data can later be collected:

```text
Safari/recorded dark frame
          ↓
adapter
          ↓
SID
          ↓
compare to cleaner reference
```

Fine-tune only the adapter first.

Do not fine-tune SID unless adapter-only training clearly saturates.

---

# Exposure/gain conditioning

The original SID pipeline multiplies its short-exposure RAW by an exposure ratio.

The live browser does not have a paired long exposure.

Therefore make exposure/gain a model input or preprocessing parameter.

At minimum expose:

```text
SID Gain
```

with a logarithmic range approximately:

```text
1× to 300×
```

Provide:

```text
Auto
10×
30×
50×
100×
200×
300×
```

or a continuous slider.

---

# Adapter conditioning

If useful camera information is exposed by Safari, allow the adapter to receive conditioning values such as:

```text
exposure time
ISO
exposure compensation
frame rate
mean luminance
luminance percentiles
```

Do not depend on metadata that Safari does not expose.

A simple architecture is sufficient:

```text
metadata/statistics
       ↓
small MLP
       ↓
feature modulation parameters
       ↓
adapter CNN
```

This is optional after the basic adapter works.

---

# Automatic gain

Implement a basic automatic gain controller based on frame luminance.

Use statistics such as:

```text
mean luminance
10th percentile
50th percentile
90th percentile
clipped-black fraction
```

Target gain should change smoothly over time.

Do not allow frame-to-frame gain oscillation.

Apply exponential smoothing to automatic gain.

---

# Camera preprocessing

Keep the live preprocessing pipeline approximately:

```text
camera frame
     ↓
YUV/RGB → RGB if necessary
     ↓
normalize to [0,1]
     ↓
approximate inverse display gamma / linearization
     ↓
optional exposure normalization
     ↓
adapter
     ↓
pseudo-RAW
     ↓
SID
```

Do not perform expensive CPU image conversion if a WebGPU shader can do it.

---

# GPU data path

Avoid:

```text
video
→ canvas
→ getImageData()
→ JavaScript Float32Array
→ model
```

for normal operation.

Prefer:

```text
HTMLVideoElement / VideoFrame
        ↓
GPU external texture
        ↓
WebGPU preprocessing
        ↓
adapter
        ↓
SID
        ↓
WebGPU postprocessing
        ↓
canvas
```

Keep camera frames and model intermediate data GPU-resident where practical.

A slow CPU fallback may exist only for debugging.

---

# Processing resolutions

Support at minimum:

```text
640×360
960×540
1280×720
```

Optionally:

```text
1920×1080
```

The camera may capture at higher resolution while inference runs lower.

Upscale output for display.

Add:

```text
Resolution:
Auto
360p
540p
720p
1080p
```

---

# Performance benchmark

On startup or from the debug panel, benchmark the complete:

```text
preprocess
+
adapter
+
SID
+
postprocess
```

pipeline.

Measure actual processing time.

Suggested selection logic:

```text
< 30 ms:
30 FPS capable

30–60 ms:
roughly 15–30 FPS

60–80 ms:
15 FPS class

> 80 ms:
reduce processing resolution
```

These are guidelines, not hard assumptions.

Do not fabricate estimated FPS.

---

# Temporal stabilization

Do not train a video model in this project phase.

After adapter + SID works, optionally implement lightweight temporal stabilization:

```text
current enhanced frame
+
previous enhanced frame
+
difference/motion mask
        ↓
stabilized output
```

Start with a motion-aware exponential moving average.

Expose:

```text
Temporal:
Off
Low
Medium
High
```

Avoid obvious ghosting.

If temporal filtering harms moving scenes, default it off or low.

---

# UI

Keep the UI minimal.

Main view:

```text
full-screen enhanced camera
```

Controls:

```text
Start Camera

Enhanced / Original

Adapter:
Manual Bayer
Learned Pseudo-RAW

Resolution:
Auto
360p
540p
720p
1080p

SID Gain:
Auto / slider

Target:
30 FPS
15 FPS
Max Quality

Temporal:
Off / Low / Medium / High

Debug
```

During development, retain the ability to switch between deterministic and learned adapters.

That comparison is important.

---

# Split comparison mode

Provide an optional split-screen mode:

```text
┌─────────────┬─────────────┐
│             │             │
│  ORIGINAL   │  ENHANCED   │
│             │             │
└─────────────┴─────────────┘
```

or draggable divider if easy.

This is useful for judging whether the model actually recovers useful detail rather than only making the image brighter.

---

# Debug overlay

Display rolling averages for:

```text
camera resolution
processing resolution
camera FPS
processed FPS

preprocess ms
adapter ms
SID ms
postprocess ms
total ms

selected backend

WebGPU available
FP16 available

adapter mode
SID gain
automatic gain value

frames received
frames processed
frames skipped
```

Do not let profiling itself materially reduce performance when the overlay is disabled.

---

# Model validation tests

Create desktop and browser equivalence tests.

## SID test

For official SID samples:

```text
official TensorFlow SID
versus
converted SID
versus
browser SID
```

## Adapter test

For a fixed processed RGB tensor:

```text
PyTorch adapter
versus
exported adapter
versus
browser adapter
```

## End-to-end test

For the same fixed input:

```text
PyTorch:
adapter → SID

versus

browser:
adapter → SID
```

Measure:

```text
MAE
max error
PSNR
```

Save visual outputs.

---

# Adapter evaluation

Evaluate the learned adapter against the deterministic pseudo-Bayer baseline.

For a held-out validation set report:

```text
PSNR
SSIM
LPIPS if useful
```

for final enhanced RGB.

Also inspect:

```text
dark detail recovery
noise
color
black clipping
highlight clipping
false texture
banding
```

The main question is:

> Does the learned adapter materially improve frozen SID's output compared with simply constructing synthetic Bayer channels?

If not, do not retain unnecessary adapter complexity.

---

# Failure criterion

If:

1. SID is correctly converted;
2. official RAW tests work;
3. the deterministic live adapter is poor;
4. the learned pseudo-RAW adapter is properly trained;
5. the learned adapter still gives obviously unusable results on real Safari camera video;

then conclude that the iPhone ISP output has moved too far from the information/domain SID requires.

Do not endlessly increase adapter complexity without evidence.

At that point preserve the existing web pipeline and make the inference backend replaceable by an existing pretrained RGB restoration model.

Do not train a new full restoration model unless separately requested.

---

# Model export

Preferred deployment format:

```text
adapter model
+
SID model
```

They may initially remain separate for debugging.

If the runtime benefits substantially, later fuse them into a single graph:

```text
RGB
 ↓
adapter
 ↓
SID
 ↓
RGB
```

Only fuse after separate models have been validated.

---

# FP16

Use FP16 where supported and numerically safe.

Aim for:

```text
adapter weights: FP16
SID weights: FP16
activations: FP16 where safe
```

Keep an FP32 compatibility/debug path.

Verify FP16 output quality against FP32.

---

# Hosting

The final application should be a static/client-side project.

A host such as Replit only needs to serve:

```text
HTML
CSS
JavaScript / TypeScript bundle
adapter weights
SID weights
WGSL shaders
metadata
```

No frames should be uploaded.

No inference server.

No database.

No user account.

No API key.

---

# Suggested repository structure

```text
/
├── index.html
├── package.json
├── vite.config.ts
│
├── src/
│   ├── main.ts
│   ├── camera.ts
│   ├── renderer.ts
│   ├── profiler.ts
│   ├── autoGain.ts
│   ├── temporal.ts
│   │
│   ├── preprocessing/
│   │   ├── linearize.ts
│   │   ├── syntheticBayer.ts
│   │   └── gpuPreprocess.ts
│   │
│   ├── inference/
│   │   ├── backend.ts
│   │   ├── sid.ts
│   │   ├── adapter.ts
│   │   ├── onnxWebgpu.ts
│   │   └── customWebgpu.ts
│   │
│   └── ui/
│       └── controls.ts
│
├── models/
│   ├── sid-sony.onnx
│   ├── pseudo-raw-adapter.onnx
│   └── metadata.json
│
├── shaders/
│   ├── preprocess.wgsl
│   ├── synthetic-bayer.wgsl
│   ├── temporal.wgsl
│   └── postprocess.wgsl
│
├── training/
│   ├── adapter_model.py
│   ├── train_adapter.py
│   ├── dataset.py
│   ├── synthetic_isp.py
│   ├── losses.py
│   └── evaluate_adapter.py
│
├── tools/
│   ├── convert_sid.py
│   ├── export_sid.py
│   ├── verify_sid.py
│   ├── export_adapter.py
│   └── export_reference_outputs.py
│
├── tests/
│   ├── sid-equivalence/
│   ├── adapter-equivalence/
│   └── end-to-end/
│
└── README.md
```

---

# README requirements

Document:

```text
how to run locally
how to deploy
Safari/iOS requirements
how camera permissions work

where SID weights came from
SID license/citation
how the TensorFlow model was converted

how the pseudo-RAW adapter works
how the adapter was trained
training datasets
adapter license

why pseudo-RAW is not physically exact RAW
known ISP limitations

benchmark results
tested iPhone models if available

known browser/WebGPU issues
```

---

# Deliverables

The project is considered complete when it includes:

1. Deployable HTTPS website.
2. Working iPhone rear-camera capture.
3. WebGPU execution.
4. Converted pretrained Sony SID model.
5. Verified SID equivalence on official RAW test data.
6. Deterministic RGB → synthetic-Bayer baseline.
7. Tiny learned pseudo-RAW adapter.
8. Adapter training pipeline with SID frozen.
9. Adapter export pipeline.
10. Browser adapter + SID inference.
11. Original / Enhanced toggle.
12. Manual and automatic SID gain.
13. Multiple processing resolutions.
14. Live FPS/inference profiler.
15. Frame-skipping logic to minimize latency.
16. Optional simple temporal stabilization.
17. Comparison between deterministic and learned adapters.
18. Documentation of real-device image quality and performance.

---

# Engineering priorities

Follow this order:

```text
1. Faithfully convert SID
2. Verify SID numerically
3. Build camera → GPU → display
4. Run deterministic pseudo-Bayer → SID live
5. Measure quality and FPS
6. Build tiny learned adapter
7. Train adapter through frozen SID
8. Deploy adapter + SID
9. Compare against baseline
10. Optimize performance
11. Add temporal stabilization
12. Polish UI
```

---

# Non-goals

Do not initially:

- train a new SID model;
- fine-tune SID;
- build SIED;
- implement diffusion;
- implement BRVE;
- build a giant inverse ISP;
- require RAW camera access;
- build a native iOS application;
- require Xcode;
- require a Mac;
- require an Apple developer subscription;
- upload camera frames to a server;
- build backend inference;
- add accounts;
- optimize for multiple browsers before iPhone Safari works;
- build a complicated learned temporal model.

---

# Core experiment

The project should ultimately answer two questions.

## Experiment A

Can the existing pretrained SID network produce useful live enhancement from an approximate synthetic Bayer representation constructed from Safari camera frames?

```text
Safari RGB
    ↓
deterministic mapping
    ↓
pseudo-Bayer
    ↓
SID
```

## Experiment B

Can a very small learned adapter do substantially better?

```text
Safari-like RGB
      ↓
tiny learned adapter
      ↓
SID-compatible pseudo-RAW
      ↓
frozen pretrained SID
      ↓
enhanced output
```

The important metric is not whether the adapter reconstructs physically exact RAW.

The important metric is:

**Does adapter + frozen SID recover useful dark-scene visibility substantially better than the deterministic baseline while remaining fast enough for live browser inference?**

That is the central goal of the project.