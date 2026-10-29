---
license: mit
library_name: onnxruntime
pipeline_tag: image-to-image
tags:
  - low-light
  - webgpu
  - onnx
  - pseudo-raw
---

# SID Live model bundle

This private bundle contains two Sony *Learning to See in the Dark* (SID)
variants: the official checkpoint converted without retraining, and an
optional quality variant jointly fine-tuned with a small RGB-to-pseudo-RAW
adapter. FP32 and numerically checked FP16-weight graphs are included. The
official graph and its TensorFlow-equivalence fixture remain untouched.

The adapter consumes inverse-sRGB linearized RGB and produces four latent
channels at half spatial resolution. These channels are **pseudo-RAW**, not a
reconstruction of physical Bayer measurements. The bundle is intended for the
client-only WebGPU application in the accompanying source repository.

## Provenance

- SID source: Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun, “Learning to
  See in the Dark,” CVPR 2018.
- Upstream implementation/checkpoint:
  `cchen156/Learning-to-See-in-the-Dark` (MIT).
- Adapter/joint-tuning data: official Sony SID archive (`Sony2025.zip`), all
  1,865 training records with four cached crops per record.
- Adapter architecture: 265,732 parameters (width 64); zero-initialized 0.10-bounded residual
  over deterministic pseudo-Bayer.

`metadata.json` records SHA-256 hashes and numerical validation results.
`evaluation.json` compares the jointly tuned pair with deterministic
pseudo-Bayer and true packed RAW on 100 held-out samples. The bundle also
retains the original frozen SID graph for faithful reproduction and direct
comparison. The learned graph is uploaded only when it passes the configured
PSNR/SSIM improvement gate.

On held-out official Sony sample `10003_00_0.04s.ARW` at 250×, the converted
FP32 ONNX SID matched the original TensorFlow output at `2.3806e-8` MAE,
`7.1526e-7` maximum error, and `148.45 dB` PSNR. FP16 weights matched FP32 at
`2.3670e-5` MAE and `1.7659e-4` maximum error on a real packed-RAW crop.

## Adapter held-out evaluation

| Samples | Method | PSNR | SSIM |
|---:|---|---:|---:|
| 100 | Joint-tuned adapter + SID | 19.0805 dB | 0.601934 |
| 100 | Deterministic pseudo-Bayer + joint-tuned SID | 16.0266 dB | 0.337266 |
| 100 | True packed Bayer RAW + joint-tuned SID | 29.1597 dB | 0.772826 |

Improvement: **+3.0538 dB PSNR** and
**+0.264669 SSIM**. Gate accepted: `true`.


## Limitations

Safari camera frames have already passed through a proprietary ISP, so the
adapter cannot recover information that was clipped or discarded. Training
uses randomized ISP-like degradation rather than paired iPhone captures;
device-specific fine-tuning may be necessary. Redistribution and use of
trained weights remain subject to the terms of the source dataset as well as
the included software license.
