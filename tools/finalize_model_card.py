from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Insert measured adapter evaluation into the model card")
    parser.add_argument("--card", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--learned-label", default="Learned pseudo-RAW")
    parser.add_argument("--manual-label", default="Deterministic pseudo-Bayer")
    parser.add_argument("--raw-label", default="True packed Bayer RAW (SID upper bound)")
    args = parser.parse_args()
    card = Path(args.card)
    report = json.loads(Path(args.evaluation).read_text())["summary"]
    marker = "<!-- ADAPTER_EVALUATION -->"
    current = card.read_text()
    section = f"""## Adapter held-out evaluation

| Samples | Method | PSNR | SSIM |
|---:|---|---:|---:|
| {report['samples']} | {args.learned_label} | {report['learned_psnr']:.4f} dB | {report['learned_ssim']:.6f} |
| {report['samples']} | {args.manual_label} | {report['manual_psnr']:.4f} dB | {report['manual_ssim']:.6f} |

Improvement: **{report['psnr_improvement']:+.4f} dB PSNR** and
**{report['ssim_improvement']:+.6f} SSIM**. Gate accepted: `{str(report['accepted']).lower()}`.
"""
    if "true_raw_psnr" in report:
        section = section.replace(
            f"| {report['samples']} | {args.manual_label} | {report['manual_psnr']:.4f} dB | {report['manual_ssim']:.6f} |",
            f"| {report['samples']} | {args.manual_label} | {report['manual_psnr']:.4f} dB | {report['manual_ssim']:.6f} |\n"
            f"| {report['true_raw_samples']} | {args.raw_label} | {report['true_raw_psnr']:.4f} dB | {report['true_raw_ssim']:.6f} |",
        )
    if marker in current:
        updated = current.replace(marker, section)
    elif "## Adapter held-out evaluation" in current and "## Limitations" in current:
        prefix = current.split("## Adapter held-out evaluation", 1)[0]
        suffix = current.split("## Limitations", 1)[1]
        updated = prefix + section + "\n\n## Limitations" + suffix
    else:
        raise ValueError(f"Missing model-card evaluation marker/section: {marker}")
    card.write_text(updated)
    print(section)


if __name__ == "__main__":
    main()
