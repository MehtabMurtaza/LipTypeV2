from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from liptype_rebuild.datasets.grid import GridLayout, SplitSpec
from liptype_rebuild.datasets.grid_paper_preproc import convert_grid_to_tfrecords_paper_style
from liptype_rebuild.datasets.splits import load_split_config
from liptype_rebuild.preprocess.paper_alignment import PaperPreprocConfig

app = typer.Typer(help="Paper-style preprocessing utilities (dlib+iBug68+Kalman+affine).")


@app.command("grid-to-tfrecords")
def grid_to_tfrecords(
    grid_root: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True, help="GRID root data/s*_processed."),
    output_root: Path = typer.Option(..., file_okay=False, dir_okay=True),
    split_config: Path = typer.Option(..., exists=True, dir_okay=False),
    num_shards: int = typer.Option(64, min=1),
    max_frames: int = typer.Option(75, min=1),
    max_text_len: int = typer.Option(32, min=1),
    max_examples: int = typer.Option(0, min=0, help="If >0, stop after N examples."),
    progress_every: int = typer.Option(200, min=0, help="Print status every N utterances (0 disables)."),
    detector_device: str = typer.Option("cpu", help="iBug predictor device, e.g. cpu or cuda:0."),
    crop_w: int = typer.Option(100, min=1, help="Mouth crop width."),
    crop_h: int = typer.Option(50, min=1, help="Mouth crop height."),
    pad_ratio_x: float = typer.Option(0.19, min=0.0),
    pad_ratio_y: float = typer.Option(0.30, min=0.0),
    target_w: int = typer.Option(256, min=1, help="Affine-normalized target width."),
    target_h: int = typer.Option(256, min=1, help="Affine-normalized target height."),
    stable_points: str = typer.Option(
        "28,33,36,39,42,45,48,54",
        help="Comma-separated 0-based stable landmark indices used for affine fitting.",
    ),
    canonical_68_npy: Optional[Path] = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Optional canonical [68,2] npy in normalized coordinates [0,1].",
    ),
    kalman_process_noise: float = typer.Option(1e-4, min=0.0),
    kalman_measurement_noise: float = typer.Option(5e-3, min=0.0),
):
    """Generate GRID TFRecords with paper-style preprocessing in one pass."""
    sc = load_split_config(split_config)
    layout = GridLayout(root=grid_root)
    utt_ids = [(spk, vp.stem) for spk, vp, _ap in layout.iter_utterances()]
    split = SplitSpec.from_config(sc, utt_ids)

    stable = tuple(int(x.strip()) for x in stable_points.split(",") if x.strip())
    if len(stable) < 3:
        raise typer.BadParameter("--stable-points must include at least 3 landmark indices.")

    cfg = PaperPreprocConfig(
        crop_w=int(crop_w),
        crop_h=int(crop_h),
        pad_ratio_x=float(pad_ratio_x),
        pad_ratio_y=float(pad_ratio_y),
        target_w=int(target_w),
        target_h=int(target_h),
        stable_points=stable,
        canonical_68_npy=canonical_68_npy,
        kalman_process_noise=float(kalman_process_noise),
        kalman_measurement_noise=float(kalman_measurement_noise),
    )

    meta = convert_grid_to_tfrecords_paper_style(
        input_root=grid_root,
        output_root=output_root,
        split=split,
        num_shards=num_shards,
        max_frames=max_frames,
        max_text_len=max_text_len,
        max_examples=(max_examples if max_examples > 0 else None),
        progress_every=progress_every,
        detector_device=detector_device,
        cfg=cfg,
    )
    typer.echo(
        f"Wrote paper-style TFRecords to {output_root} "
        f"(train={meta['train_examples']} val={meta['val_examples']} test={meta['test_examples']} "
        f"skip={meta['skipped_examples']} fail={meta['failed_examples']})."
    )


@app.command("liptype-test-to-tfrecords")
def liptype_test_to_tfrecords(
    input_root: Path = typer.Option(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Root containing P*/data/silent speech folders (e.g., Version10_phrase_dataset).",
    ),
    readme: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
        help="Phrase mapping text (e.g., video_phrase_mapping.txt or README.txt).",
    ),
    output_root: Path = typer.Option(..., file_okay=False, dir_okay=True),
    include_vers: str = typer.Option("10", help="Comma-separated silent-speech versions to include."),
    num_shards: int = typer.Option(64, min=1),
    max_frames: int = typer.Option(75, min=1),
    max_text_len: int = typer.Option(120, min=1),
    max_examples: int = typer.Option(0, min=0, help="If >0, stop after N examples."),
    progress_every: int = typer.Option(200, min=0, help="Print status every N videos (0 disables)."),
    detector_device: str = typer.Option("cpu", help="iBug predictor device, e.g. cpu or cuda:0."),
    crop_w: int = typer.Option(100, min=1, help="Mouth crop width."),
    crop_h: int = typer.Option(50, min=1, help="Mouth crop height."),
    pad_ratio_x: float = typer.Option(0.19, min=0.0),
    pad_ratio_y: float = typer.Option(0.30, min=0.0),
    target_w: int = typer.Option(256, min=1, help="Affine-normalized target width."),
    target_h: int = typer.Option(256, min=1, help="Affine-normalized target height."),
    stable_points: str = typer.Option(
        "28,33,36,39,42,45,48,54",
        help="Comma-separated 0-based stable landmark indices used for affine fitting.",
    ),
    canonical_68_npy: Optional[Path] = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Optional canonical [68,2] npy in normalized coordinates [0,1].",
    ),
    kalman_process_noise: float = typer.Option(1e-4, min=0.0),
    kalman_measurement_noise: float = typer.Option(5e-3, min=0.0),
):
    """Generate test TFRecords for LipType test data with paper-style preprocessing."""
    from liptype_rebuild.datasets.liptype_test import convert_liptype_test_to_tfrecords_paper_style

    vers = {int(v.strip()) for v in include_vers.split(",") if v.strip()}
    if not vers:
        raise typer.BadParameter("--include-vers must contain at least one integer version.")

    stable = tuple(int(x.strip()) for x in stable_points.split(",") if x.strip())
    if len(stable) < 3:
        raise typer.BadParameter("--stable-points must include at least 3 landmark indices.")

    cfg = PaperPreprocConfig(
        crop_w=int(crop_w),
        crop_h=int(crop_h),
        pad_ratio_x=float(pad_ratio_x),
        pad_ratio_y=float(pad_ratio_y),
        target_w=int(target_w),
        target_h=int(target_h),
        stable_points=stable,
        canonical_68_npy=canonical_68_npy,
        kalman_process_noise=float(kalman_process_noise),
        kalman_measurement_noise=float(kalman_measurement_noise),
    )

    meta = convert_liptype_test_to_tfrecords_paper_style(
        input_root=input_root,
        readme_path=readme,
        output_root=output_root,
        include_vers=vers,
        num_shards=num_shards,
        max_frames=max_frames,
        max_text_len=max_text_len,
        max_examples=(max_examples if max_examples > 0 else None),
        progress_every=progress_every,
        detector_device=detector_device,
        cfg=cfg,
    )
    typer.echo(
        f"Wrote paper-style test TFRecords to {output_root} "
        f"(test={meta['test_examples']} fail={meta['failed_examples']})."
    )


@app.command("liptype-test-train-val-to-tfrecords")
def liptype_test_train_val_to_tfrecords(
    input_root: Path = typer.Option(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Root containing P*/data/silent speech folders (e.g., Version10_phrase_dataset).",
    ),
    readme: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
        help="Phrase mapping text (e.g., video_phrase_mapping.txt or README.txt).",
    ),
    output_root: Path = typer.Option(..., file_okay=False, dir_okay=True),
    include_vers: str = typer.Option("10", help="Comma-separated silent-speech versions to include."),
    val_ratio: float = typer.Option(0.2, min=0.001, max=0.999, help="Validation split ratio."),
    split_seed: int = typer.Option(42, help="Random seed for deterministic train/val split."),
    num_shards: int = typer.Option(64, min=1),
    max_frames: int = typer.Option(75, min=1),
    max_text_len: int = typer.Option(120, min=1),
    max_examples: int = typer.Option(0, min=0, help="If >0, stop after N examples."),
    progress_every: int = typer.Option(200, min=0, help="Print status every N videos (0 disables)."),
    detector_device: str = typer.Option("cpu", help="iBug predictor device, e.g. cpu or cuda:0."),
    crop_w: int = typer.Option(100, min=1, help="Mouth crop width."),
    crop_h: int = typer.Option(50, min=1, help="Mouth crop height."),
    pad_ratio_x: float = typer.Option(0.19, min=0.0),
    pad_ratio_y: float = typer.Option(0.30, min=0.0),
    target_w: int = typer.Option(256, min=1, help="Affine-normalized target width."),
    target_h: int = typer.Option(256, min=1, help="Affine-normalized target height."),
    stable_points: str = typer.Option(
        "28,33,36,39,42,45,48,54",
        help="Comma-separated 0-based stable landmark indices used for affine fitting.",
    ),
    canonical_68_npy: Path = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Optional canonical [68,2] npy in normalized coordinates [0,1].",
    ),
    kalman_process_noise: float = typer.Option(1e-4, min=0.0),
    kalman_measurement_noise: float = typer.Option(5e-3, min=0.0),
):
    """Generate train/val TFRecords for LipType test data with paper-style preprocessing."""
    from liptype_rebuild.datasets.liptype_test import convert_liptype_test_to_tfrecords_paper_style_train_val

    vers = {int(v.strip()) for v in include_vers.split(",") if v.strip()}
    if not vers:
        raise typer.BadParameter("--include-vers must contain at least one integer version.")

    stable = tuple(int(x.strip()) for x in stable_points.split(",") if x.strip())
    if len(stable) < 3:
        raise typer.BadParameter("--stable-points must include at least 3 landmark indices.")

    cfg = PaperPreprocConfig(
        crop_w=int(crop_w),
        crop_h=int(crop_h),
        pad_ratio_x=float(pad_ratio_x),
        pad_ratio_y=float(pad_ratio_y),
        target_w=int(target_w),
        target_h=int(target_h),
        stable_points=stable,
        canonical_68_npy=canonical_68_npy,
        kalman_process_noise=float(kalman_process_noise),
        kalman_measurement_noise=float(kalman_measurement_noise),
    )

    meta = convert_liptype_test_to_tfrecords_paper_style_train_val(
        input_root=input_root,
        readme_path=readme,
        output_root=output_root,
        include_vers=vers,
        val_ratio=val_ratio,
        split_seed=split_seed,
        num_shards=num_shards,
        max_frames=max_frames,
        max_text_len=max_text_len,
        max_examples=(max_examples if max_examples > 0 else None),
        progress_every=progress_every,
        detector_device=detector_device,
        cfg=cfg,
    )
    typer.echo(
        f"Wrote paper-style train/val TFRecords to {output_root} "
        f"(train={meta['train_examples']} val={meta['val_examples']} fail={meta['failed_examples']})."
    )

