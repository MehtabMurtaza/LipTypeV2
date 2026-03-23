from __future__ import annotations

"""
Batch mouth-crop videos using the Auto-AVSR conventional alignment pipeline.

This script intentionally lives outside the TF pipeline to avoid mixing
PyTorch/ibug dependencies into training/inference.

Prereqs:
  - Clone https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages
  - Create a separate env with: torch, torchvision, torchaudio, opencv-python, scikit-image
  - For retinaface+FAN landmarks, install ibug packages used by Auto-AVSR.

Usage (example):
  python scripts/autoavsr_crop_mouth_batch.py ^
    --autoavsr-root /path/to/Visual_Speech_Recognition_for_Multiple_Languages ^
    --input-root LipType_Test_Dataset ^
    --glob \"**/data/silent speech/*.mp4\" ^
    --output-root preprocess_out/mouth_aligned_mp4 ^
    --detector retinaface
"""

from dataclasses import dataclass
from pathlib import Path
import sys
import traceback

import argparse


@dataclass(frozen=True)
class Args:
    autoavsr_root: Path
    input_root: Path
    glob: str
    output_root: Path
    detector: str
    out_ext: str
    overwrite: bool
    max_files: int


def _parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--autoavsr-root", type=Path, required=True)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--glob", type=str, default="**/*.mp4")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--detector", type=str, default="retinaface", choices=["retinaface", "mediapipe"])
    p.add_argument(
        "--out-ext",
        type=str,
        default="mp4",
        help="Output extension (container). Use 'mp4' (recommended).",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-files", type=int, default=0, help="If >0, stop after N videos.")
    ns = p.parse_args()
    out_ext = str(ns.out_ext).strip().lstrip(".").lower()
    if out_ext == "":
        raise SystemExit("--out-ext must be a non-empty extension like 'mp4'.")
    return Args(
        autoavsr_root=ns.autoavsr_root,
        input_root=ns.input_root,
        glob=ns.glob,
        output_root=ns.output_root,
        detector=ns.detector,
        out_ext=out_ext,
        overwrite=bool(ns.overwrite),
        max_files=int(ns.max_files),
    )


def main() -> int:
    args = _parse_args()

    # Make Auto-AVSR repo importable.
    sys.path.insert(0, str(args.autoavsr_root))

    try:
        import cv2  # noqa: F401
        import torch  # noqa: F401
        import torchvision
        from pipelines.data.data_module import AVSRDataLoader

        if args.detector == "retinaface":
            from pipelines.detectors.retinaface.detector import LandmarksDetector
        else:
            from pipelines.detectors.mediapipe.detector import LandmarksDetector
    except Exception as e:
        print("Failed to import Auto-AVSR dependencies.")
        print("Error:", e)
        print("Make sure you cloned the Auto-AVSR repo and installed its deps in this env.")
        return 2

    det = LandmarksDetector()
    # Keep RGB crops; we will resize/format later for LipType.
    loader = AVSRDataLoader(modality="video", speed_rate=1, transform=False, detector=args.detector, convert_gray=False)

    in_files = sorted(args.input_root.glob(args.glob))
    if not in_files:
        print(f"No files matched {args.glob} under {args.input_root}")
        return 1

    n_ok = 0
    n_fail = 0

    for idx, in_path in enumerate(in_files):
        if args.max_files > 0 and idx >= args.max_files:
            break

        rel = in_path.relative_to(args.input_root)
        out_path = (args.output_root / rel).with_suffix(f".{args.out_ext}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not args.overwrite:
            continue

        try:
            landmarks = det(str(in_path))
            data = loader.load_data(str(in_path), landmarks)  # torch tensor [T,H,W,C]

            # Write mp4 using torchvision, preserving fps if possible.
            cap = cv2.VideoCapture(str(in_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()

            vid = data
            if hasattr(vid, "detach"):
                vid = vid.detach().cpu()
            # Ensure uint8 for write_video
            if vid.dtype != torch.uint8:
                vid = vid.clamp(0, 255).to(torch.uint8)

            torchvision.io.write_video(str(out_path), vid, float(fps))
            n_ok += 1
        except Exception:
            n_fail += 1
            print(f"[FAIL] {in_path}")
            traceback.print_exc(limit=2)

    print(f"Done. ok={n_ok} fail={n_fail} out_root={args.output_root}")
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

