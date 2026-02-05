from __future__ import annotations

from pathlib import Path

import typer

import numpy as np

app = typer.Typer(help="Low-light enhancement (GLADNet-like) utilities.")


@app.command("video")
def enhance_video(
    weights: Path = typer.Option(..., exists=True, dir_okay=False),
    input_video: Path = typer.Option(..., exists=True),
    output_video: Path = typer.Option(..., dir_okay=False),
    batch_size: int = typer.Option(16, min=1),
):
    """Enhance a low-light video frame-by-frame."""
    import cv2
    import tensorflow as tf

    from liptype_rebuild.enhance.gladnet import build_gladnet
    from liptype_rebuild.preprocess.video_io import read_video_rgb

    model = build_gladnet()
    model.load_weights(str(weights))

    vf = read_video_rgb(input_video)
    frames = vf.frames_rgb.astype(np.float32) / 255.0

    outs: list[np.ndarray] = []
    for i in range(0, frames.shape[0], batch_size):
        b = frames[i : i + batch_size]
        y = model(b, training=False).numpy()
        y = np.clip(y * 255.0, 0, 255).astype(np.uint8)
        outs.append(y)

    out_frames = np.concatenate(outs, axis=0)

    # Write video (BGR for OpenCV)
    h, w = out_frames.shape[1], out_frames.shape[2]
    fps = vf.fps if vf.fps > 0 else 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (w, h))
    try:
        for fr_rgb in out_frames:
            writer.write(cv2.cvtColor(fr_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    typer.echo(f"Wrote enhanced video to {output_video}")

