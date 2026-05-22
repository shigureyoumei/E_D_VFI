#!/usr/bin/env python3
"""Create a HighREV-style 3+1 dataset view from BSERGB.

This variant is for testing a slightly longer synthetic BSERGB blur:
    blur_left  = average of 3 sharp frames
    middle     = 1 skipped/interpolated sharp frame
    blur_right = average of 3 sharp frames

Each training sample has 7 GT frames and 8 event files. With
return_deblur_voxel=True, the network image input has 10 channels:
    3 RGB + 2 left deblur voxel + 3 RGB + 2 right deblur voxel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from create_bsergb_11_3 import create_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="/work/HAIRDESC/naran/datasets/bs_ergb", help="Source BSERGB root")
    parser.add_argument("--dst", default="/work/HAIRDESC/naran/datasets/EDVFI/BSERGB_3_1", help="Destination HighREV-style dataset root")
    parser.add_argument("--m", type=int, default=3, help="3+1 config: number of GT frames per endpoint blur")
    parser.add_argument("--n", type=int, default=1, help="3+1 config: skipped/interpolated frames between blur inputs")
    parser.add_argument("--blur-exposure-frames", type=int, default=3, help="Number of sharp frames averaged into each synthetic BSERGB blur")
    parser.add_argument("--train-split", default="3_TRAINING", help="BSERGB split used for output train/")
    parser.add_argument("--test-split", default="4_TEST", help="BSERGB split used for output test/")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated blur/event files")
    parser.add_argument("--compress-events", action="store_true", help="Write compressed event npz files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.dst)
    if not src_root.is_dir():
        raise FileNotFoundError(f"Source dataset root does not exist: {src_root}")

    summary = []
    summary.append(("train",) + create_split(
        src_root, dst_root, args.train_split, "train",
        args.m, args.n, args.blur_exposure_frames,
        args.overwrite, args.compress_events,
    ))
    summary.append(("test",) + create_split(
        src_root, dst_root, args.test_split, "test",
        args.m, args.n, args.blur_exposure_frames,
        args.overwrite, args.compress_events,
    ))

    for split, sequences, blurs, events in summary:
        print(f"Summary {split}: sequences={sequences} generated_blur={blurs} converted_event={events}", flush=True)
    print(f"Done. Dataset: {dst_root}", flush=True)


if __name__ == "__main__":
    main()
