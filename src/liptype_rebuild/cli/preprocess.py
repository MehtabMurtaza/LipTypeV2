from __future__ import annotations

from pathlib import Path

import typer

from liptype_rebuild.datasets.grid import SplitSpec, convert_grid_to_tfrecords
from liptype_rebuild.datasets.splits import load_split_config

app = typer.Typer(help="Dataset preprocessing utilities (TFRecords, ROI extraction).")


@app.command("grid-to-tfrecords")
def grid_to_tfrecords(
    input_root: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    output_root: Path = typer.Option(..., file_okay=False, dir_okay=True),
    split_config: Path = typer.Option(None, exists=True, dir_okay=False),
    num_shards: int = typer.Option(64, min=1),
    max_frames: int = typer.Option(75, min=1),
    max_examples: int = typer.Option(0, min=0, help="If >0, stop after N examples."),
    dlib_predictor: Path = typer.Option(
        None, exists=True, dir_okay=False, help="Path to dlib 68-landmark predictor .dat file."
    ),
):
    """Convert GRID-like dataset (mpg + align) to TFRecords.

    Expects input layout:
      - data/sX_processed/*.mpg
      - data/sX_processed/align/*.align

    Writes sharded TFRecords to output_root.
    """
    if split_config is None:
        raise typer.BadParameter("--split-config is required (YAML with val_speakers/train_speakers).")

    sc = load_split_config(split_config)
    split = SplitSpec.from_seen_unseen(sc.train_speakers, sc.val_speakers)

    landmarks_backend = None
    if dlib_predictor is not None:
        from liptype_rebuild.preprocess.landmarks import Dlib68Backend

        landmarks_backend = Dlib68Backend(str(dlib_predictor))

    convert_grid_to_tfrecords(
        input_root=input_root,
        output_root=output_root,
        split=split,
        num_shards=num_shards,
        max_frames=max_frames,
        max_examples=(max_examples if max_examples > 0 else None),
        landmarks_backend=landmarks_backend,
    )
    typer.echo(f"Wrote TFRecords to {output_root}")

