from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="models")
    parser.add_argument("--sid-validation", help="metrics.json from tools/verify_sid.py")
    parser.add_argument("--adapter-checkpoint")
    parser.add_argument("--sid-fp16-validation", help="FP16 numerical comparison report")
    parser.add_argument("--sid-joint-fp16-validation", help="joint SID FP16 numerical comparison report")
    parser.add_argument("--adapter-fp16-validation", help="FP16 numerical comparison report")
    parser.add_argument("--sid-static-fp16-validation")
    parser.add_argument("--sid-joint-static-fp16-validation")
    parser.add_argument("--adapter-static-fp16-validation")
    args = parser.parse_args()
    root = Path(args.models)
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    for key, filename in (
        ("sid", "sid-sony.onnx"),
        ("sidJoint", "sid-sony-joint.onnx"),
        ("adapter", "pseudo-raw-adapter.onnx"),
    ):
        if key not in metadata:
            continue
        path = root / filename
        metadata[key]["available"] = path.exists()
        metadata[key]["sha256"] = digest(path) if path.exists() else None
        fp16_path = root / filename.replace(".onnx", "-fp16.onnx")
        if "fp16" in metadata[key]:
            metadata[key]["fp16"]["available"] = fp16_path.exists()
            metadata[key]["fp16"]["sha256"] = digest(fp16_path) if fp16_path.exists() else None
        fixed = metadata[key].get("fixed360p")
        if fixed:
            fixed_path = root / Path(fixed["url"]).name
            fixed["available"] = fixed_path.exists()
            fixed["sha256"] = digest(fixed_path) if fixed_path.exists() else None
            fixed_fp16 = fixed.get("fp16")
            if fixed_fp16:
                fixed_fp16_path = root / Path(fixed_fp16["url"]).name
                fixed_fp16["available"] = fixed_fp16_path.exists()
                fixed_fp16["sha256"] = digest(fixed_fp16_path) if fixed_fp16_path.exists() else None
    for key, report_path in (
        ("sid", args.sid_fp16_validation),
        ("sidJoint", args.sid_joint_fp16_validation),
        ("adapter", args.adapter_fp16_validation),
    ):
        if report_path:
            report = json.loads(Path(report_path).read_text())
            metadata[key]["fp16"]["mae"] = report["mae"]
            metadata[key]["fp16"]["maxError"] = report["max_error"]
    for key, report_path in (
        ("sid", args.sid_static_fp16_validation),
        ("sidJoint", args.sid_joint_static_fp16_validation),
        ("adapter", args.adapter_static_fp16_validation),
    ):
        if report_path:
            report = json.loads(Path(report_path).read_text())
            metadata[key]["fixed360p"]["fp16"]["mae"] = report["mae"]
            metadata[key]["fixed360p"]["fp16"]["maxError"] = report["max_error"]
    if args.adapter_checkpoint:
        import torch
        payload = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=False)
        metadata["adapter"]["parameterCount"] = int(payload["parameter_count"])
    if args.sid_validation:
        report = json.loads(Path(args.sid_validation).read_text())
        values = report["tensorflow_vs_onnx"]
        metadata["validation"] = {
            "verified": values["mae"] < 1e-5 and values["max_error"] < 5e-4,
            "referenceSample": str(args.sid_validation),
            "mae": values["mae"],
            "maxError": values["max_error"],
            "psnr": values["psnr"],
        }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
