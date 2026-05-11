# -*- coding: utf-8 -*-
"""
extract_video_frames.py

Video Action Dataset Prototype.

Extract frames from a MuJoCo/VLA demo video.

Input:
    media/vla_panda_demo.mp4

Output:
    video_action/frames/vla_panda_demo/frame_000001.jpg
    video_action/frames/vla_panda_demo/frame_000002.jpg
    ...

Run:
    python3 video_action/extract_video_frames.py --video media/vla_panda_demo.mp4 --fps 2
"""

from __future__ import annotations

from pathlib import Path
import argparse
import cv2


def extract_frames(video_path: str, output_root: str = "video_action/frames", fps: float = 2.0) -> Path:
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    video_name = video_path.stem
    output_dir = Path(output_root) / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / source_fps if source_fps > 0 else 0.0

    if source_fps <= 0:
        source_fps = 30.0

    frame_interval = max(1, int(round(source_fps / fps)))

    print("\n--- Video Frame Extraction ---")
    print(f"Video: {video_path}")
    print(f"Source FPS: {source_fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration_s:.2f} s")
    print(f"Target extraction FPS: {fps}")
    print(f"Frame interval: {frame_interval}")
    print(f"Output dir: {output_dir}")

    frame_idx = 0
    saved_idx = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if frame_idx % frame_interval == 0:
            saved_idx += 1
            out_path = output_dir / f"frame_{saved_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)

        frame_idx += 1

    cap.release()

    print(f"\nSaved frames: {saved_idx}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="media/vla_panda_demo.mp4")
    parser.add_argument("--output-root", type=str, default="video_action/frames")
    parser.add_argument("--fps", type=float, default=2.0)

    args = parser.parse_args()

    extract_frames(
        video_path=args.video,
        output_root=args.output_root,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
