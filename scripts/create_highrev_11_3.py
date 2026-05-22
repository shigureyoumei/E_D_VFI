#!/usr/bin/env python3
"""Create a HighREV 11+3 dataset view from HighREV_full GT/event data.

The original HighREV_full blur frames in this workspace are spaced for 11+1.
For 11+3 training, this script regenerates blur images by averaging 11
consecutive GT frames every 14 frames. GT and event folders are symlinked to
avoid duplicating the full dataset.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="datasets/HighREV_full", help="Source HighREV_full root")
    parser.add_argument("--dst", default="/work/HAIRDESC/naran/datasets/EDVFI/HighREV_11_3", help="Destination dataset root")
    parser.add_argument("--m", type=int, default=11, help="Number of GT frames per blur")
    parser.add_argument("--n", type=int, default=3, help="Skipped/interpolated frames between blur inputs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated blur images")
    return parser.parse_args()


def rel_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(os.path.relpath(src.resolve(), dst.parent.resolve()), dst)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image.astype(np.float32)


def write_blur(gt_paths: list[Path], out_path: Path) -> None:
    accumulator = None
    for gt_path in gt_paths:
        image = read_image(gt_path)
        accumulator = image if accumulator is None else accumulator + image
    blur = np.clip(np.rint(accumulator / len(gt_paths)), 0, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), blur):
        raise RuntimeError(f"Failed to write image: {out_path}")


def create_sequence(src_sequence: Path, dst_sequence: Path, m: int, n: int, overwrite: bool) -> int:
    gt_dir = src_sequence / "gt"
    event_dir = src_sequence / "event"
    dst_blur_dir = dst_sequence / "blur"

    gt_paths = sorted(gt_dir.glob("*.png"))
    event_paths = sorted(event_dir.glob("*.npz"))
    usable_frames = min(len(gt_paths), len(event_paths))
    step = m + n
    if usable_frames < 2 * m + n:
        return 0

    rel_symlink(gt_dir, dst_sequence / "gt")
    rel_symlink(event_dir, dst_sequence / "event")

    created = 0
    for start in range(0, usable_frames - m + 1, step):
        center = start + m // 2
        out_path = dst_blur_dir / f"{center:08d}_0.png"
        if out_path.exists() and not overwrite:
            continue
        write_blur(gt_paths[start : start + m], out_path)
        created += 1
    return created


def main() -> None:
    args = parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.dst)
    if not src_root.is_dir():
        raise FileNotFoundError(f"Source dataset root does not exist: {src_root}")

    total = 0
    for split in ("train", "val"):
        split_root = src_root / split
        if not split_root.is_dir():
            continue
        for src_sequence in sorted(p for p in split_root.iterdir() if p.is_dir()):
            dst_sequence = dst_root / split / src_sequence.name
            count = create_sequence(src_sequence, dst_sequence, args.m, args.n, args.overwrite)
            print(f"{split}/{src_sequence.name}: generated_or_updated_blur={count}")
            total += count

    print(f"Done. Dataset: {dst_root} generated_or_updated_blur={total}")


if __name__ == "__main__":
    main()
