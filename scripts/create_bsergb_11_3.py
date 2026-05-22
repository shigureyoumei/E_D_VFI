#!/usr/bin/env python3
"""Create a HighREV-style 2+1 dataset view from BSERGB.

Output layout:
    dst/
      train/<sequence>/{blur,gt,event}
      test/<sequence>/{blur,gt,event}

The existing RuisiEventRecurrentDataset expects event npz files with column
vectors and swaps event["x"] and event["y"] internally. BSERGB stores event
coordinates as fixed-point values, so this script scales coordinates by /32 and
writes swapped keys:
    saved x = scaled y
    saved y = scaled x
This makes the effective coordinates used by RuisiEventRecurrentDataset match
the RGB image pixel grid.

BSERGB has larger motion than GoPro/HighREV, so the default temporal layout is
2+1 instead of 11+3:
    blur_left  = average of 2 sharp frames
    middle     = 1 skipped/interpolated sharp frame
    blur_right = average of 2 sharp frames
Each sample therefore has 5 GT frames and 6 event files. With
return_deblur_voxel=True, the network image input has 8 channels:
    3 RGB + 1 left deblur voxel + 3 RGB + 1 right deblur voxel.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="/work/HAIRDESC/naran/datasets/bs_ergb", help="Source BSERGB root")
    parser.add_argument("--dst", default="/work/HAIRDESC/naran/datasets/EDVFI/BSERGB_2_1", help="Destination HighREV-style dataset root")
    parser.add_argument("--m", type=int, default=2, help="2+1 config: number of GT frames per endpoint blur")
    parser.add_argument("--n", type=int, default=1, help="2+1 config: skipped/interpolated frames between blur inputs")
    parser.add_argument("--blur-exposure-frames", type=int, default=2, help="Number of sharp frames averaged into each synthetic BSERGB blur")
    parser.add_argument("--train-split", default="3_TRAINING", help="BSERGB split used for output train/")
    parser.add_argument("--test-split", default="4_TEST", help="BSERGB split used for output test/")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated blur/event files")
    parser.add_argument("--compress-events", action="store_true", help="Write compressed event npz files")
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


def write_blur(gt_paths: list[Path], out_path: Path, overwrite: bool) -> bool:
    if out_path.exists() and not overwrite:
        return False

    accumulator = None
    for gt_path in gt_paths:
        image = read_image(gt_path)
        accumulator = image if accumulator is None else accumulator + image

    blur = np.clip(np.rint(accumulator / len(gt_paths)), 0, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), blur):
        raise RuntimeError(f"Failed to write image: {out_path}")
    return True


def convert_event_npz(src_path: Path, dst_path: Path, width: int, height: int, overwrite: bool, compress: bool) -> bool:
    if dst_path.exists() and not overwrite:
        return False

    event = np.load(src_path)
    x = event["x"].astype(np.float32) / 32.0
    y = event["y"].astype(np.float32) / 32.0
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)

    timestamp = event["timestamp"][:, None]
    polarity = event["polarity"][:, None]

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    save_fn = np.savez_compressed if compress else np.savez
    save_fn(
        dst_path,
        x=y[:, None].astype(np.float32),
        y=x[:, None].astype(np.float32),
        timestamp=timestamp,
        polarity=polarity,
    )
    return True


def blur_source_paths(image_paths: list[Path], window_start: int, m: int, exposure_frames: int) -> list[Path]:
    if exposure_frames < 1:
        raise ValueError(f"blur exposure must be >= 1, got {exposure_frames}")
    if exposure_frames > m:
        raise ValueError(f"blur exposure {exposure_frames} cannot be larger than m={m}")

    if exposure_frames == m:
        first = window_start
    else:
        center = window_start + m // 2
        first = center - (exposure_frames - 1) // 2
    last = first + exposure_frames
    return image_paths[first:last]


def create_sequence(src_sequence: Path, dst_sequence: Path, m: int, n: int, blur_exposure_frames: int,
                    overwrite: bool, compress: bool) -> tuple[int, int]:
    image_dir = src_sequence / "images"
    event_dir = src_sequence / "events"
    if not image_dir.is_dir() or not event_dir.is_dir():
        print(f"Skip {src_sequence}: missing images/ or events/", flush=True)
        return 0, 0

    image_paths = sorted(image_dir.glob("*.png"))
    event_paths = sorted(event_dir.glob("*.npz"))
    if len(image_paths) < 2 * m + n or len(event_paths) < 2 * m + n - 1:
        print(f"Skip {src_sequence}: too short images={len(image_paths)} events={len(event_paths)}", flush=True)
        return 0, 0

    first_image = cv2.imread(str(image_paths[0]), cv2.IMREAD_UNCHANGED)
    if first_image is None:
        raise RuntimeError(f"Failed to read image: {image_paths[0]}")
    height, width = first_image.shape[:2]

    rel_symlink(image_dir, dst_sequence / "gt")

    converted_events = 0
    for src_event in event_paths:
        dst_event = dst_sequence / "event" / src_event.name
        converted_events += int(convert_event_npz(src_event, dst_event, width, height, overwrite, compress))

    created_blurs = 0
    step = m + n
    usable_frames = min(len(image_paths), len(event_paths) + 1)
    for start in range(0, usable_frames - m + 1, step):
        center = start + m // 2
        out_path = dst_sequence / "blur" / f"{center:08d}_0.png"
        created_blurs += int(write_blur(blur_source_paths(image_paths, start, m, blur_exposure_frames), out_path, overwrite))

    return created_blurs, converted_events


def create_split(src_root: Path, dst_root: Path, src_split: str, dst_split: str, m: int, n: int,
                 blur_exposure_frames: int, overwrite: bool, compress: bool) -> tuple[int, int, int]:
    split_root = src_root / src_split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Source split does not exist: {split_root}")

    total_sequences = 0
    total_blurs = 0
    total_events = 0
    for src_sequence in sorted(p for p in split_root.iterdir() if p.is_dir()):
        dst_sequence = dst_root / dst_split / src_sequence.name
        blur_count, event_count = create_sequence(src_sequence, dst_sequence, m, n, blur_exposure_frames, overwrite, compress)
        if blur_count or event_count or (dst_sequence / "gt").exists():
            total_sequences += 1
        total_blurs += blur_count
        total_events += event_count
        print(f"{dst_split}/{src_sequence.name}: generated_blur={blur_count} converted_event={event_count}", flush=True)

    return total_sequences, total_blurs, total_events


def main() -> None:
    args = parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.dst)
    if not src_root.is_dir():
        raise FileNotFoundError(f"Source dataset root does not exist: {src_root}")

    summary = []
    summary.append(("train",) + create_split(src_root, dst_root, args.train_split, "train", args.m, args.n, args.blur_exposure_frames, args.overwrite, args.compress_events))
    summary.append(("test",) + create_split(src_root, dst_root, args.test_split, "test", args.m, args.n, args.blur_exposure_frames, args.overwrite, args.compress_events))

    for split, sequences, blurs, events in summary:
        print(f"Summary {split}: sequences={sequences} generated_blur={blurs} converted_event={events}", flush=True)
    print(f"Done. Dataset: {dst_root}", flush=True)


if __name__ == "__main__":
    main()
