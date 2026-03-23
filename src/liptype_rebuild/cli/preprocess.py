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
    enhance_weights: Path = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Optional: GLADNet enhancer weights to apply before ROI extraction.",
    ),
    enhance_batch_size: int = typer.Option(16, min=1, help="Batch size for enhancer inference."),
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
    # For overlapped_utterances mode we need utterance IDs to pick per-speaker validation samples.
    from liptype_rebuild.datasets.grid import GridLayout

    layout = GridLayout(root=input_root)
    utt_ids = [(spk, vp.stem) for spk, vp, _ap in layout.iter_utterances()]
    split = SplitSpec.from_config(sc, utt_ids)

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
        enhance_weights=enhance_weights,
        enhance_batch_size=enhance_batch_size,
    )
    typer.echo(f"Wrote TFRecords to {output_root}")


@app.command("liptype-test-to-tfrecords")
def liptype_test_to_tfrecords(
    input_root: Path = typer.Option(
        Path("LipType_Test_Dataset"),
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Root containing P1..P12 folders.",
    ),
    readme: Path = typer.Option(
        Path("LipType_Test_Dataset/README.txt"),
        exists=True,
        dir_okay=False,
        help="README.txt with filename->phrase table.",
    ),
    output_root: Path = typer.Option(..., file_okay=False, dir_okay=True),
    include_vers: str = typer.Option(
        "4,7,10",
        help=(
            "Comma-separated silent-speech versions to include. "
            "README defines phrases for ver1,ver4,ver7,ver10. "
            "Use 4,7,10 to approximate 'Day' per the provided run.sh."
        ),
    ),
    num_shards: int = typer.Option(8, min=1),
    max_frames: int = typer.Option(75, min=1),
    max_text_len: int = typer.Option(120, min=1),
    max_examples: int = typer.Option(0, min=0, help="If >0, stop after N examples."),
    dlib_predictor: Path = typer.Option(
        None, exists=True, dir_okay=False, help="Optional: path to dlib 68-landmark predictor .dat file."
    ),
):
    """Convert LipType_Test_Dataset silent-speech videos to TFRecords (test split only).

    Output files:
      - output_root/test-00000-of-XXXXX.tfrecord
      - output_root/meta.json
    """
    from liptype_rebuild.datasets.liptype_test import convert_liptype_test_to_tfrecords

    vers = {int(v.strip()) for v in include_vers.split(",") if v.strip()}
    if not vers:
        raise typer.BadParameter("--include-vers must contain at least one integer version.")

    landmarks_backend = None
    if dlib_predictor is not None:
        from liptype_rebuild.preprocess.landmarks import Dlib68Backend

        landmarks_backend = Dlib68Backend(str(dlib_predictor))

    meta = convert_liptype_test_to_tfrecords(
        input_root=input_root,
        readme_path=readme,
        output_root=output_root,
        include_vers=vers,
        num_shards=num_shards,
        max_frames=max_frames,
        max_text_len=max_text_len,
        landmarks_backend=landmarks_backend,
        max_examples=(max_examples if max_examples > 0 else None),
    )
    typer.echo(f"Wrote test TFRecords to {output_root} (n={meta['test_examples']}).")


@app.command("mouth-mp4-to-tfrecords")
def mouth_mp4_to_tfrecords(
    input_root: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True, help="Root of mouth-only mp4s."),
    readme: Path = typer.Option(..., exists=True, dir_okay=False, help="README.txt with ver/ph -> phrase table."),
    output_root: Path = typer.Option(..., file_okay=False, dir_okay=True),
    include_vers: str = typer.Option("10", help="Comma-separated silent-speech versions to include, e.g. '10' or '1,4'."),
    num_shards: int = typer.Option(8, min=1),
    max_frames: int = typer.Option(75, min=1),
    img_w: int = typer.Option(100, min=1),
    img_h: int = typer.Option(50, min=1),
    img_c: int = typer.Option(3, min=1, max=3),
    max_text_len: int = typer.Option(120, min=1),
    max_examples: int = typer.Option(0, min=0, help="If >0, stop after N examples."),
):
    """Convert mouth-only mp4s (already aligned/cropped) into TFRecords (test split only)."""
    from liptype_rebuild.datasets.mouth_mp4 import convert_mouth_mp4_to_tfrecords

    vers = {int(v.strip()) for v in include_vers.split(",") if v.strip()}
    if not vers:
        raise typer.BadParameter("--include-vers must contain at least one integer version.")

    meta = convert_mouth_mp4_to_tfrecords(
        input_root=input_root,
        readme_path=readme,
        output_root=output_root,
        include_vers=vers,
        num_shards=num_shards,
        max_frames=max_frames,
        out_w=img_w,
        out_h=img_h,
        out_c=img_c,
        max_text_len=max_text_len,
        max_examples=(max_examples if max_examples > 0 else None),
    )
    typer.echo(f"Wrote test TFRecords to {output_root} (n={meta['test_examples']}).")

