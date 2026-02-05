from __future__ import annotations

from pathlib import Path

import typer


app = typer.Typer(help="Evaluation utilities.")


@app.command("liptype")
def eval_liptype(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    weights: Path = typer.Option(..., exists=True, dir_okay=False),
    num_batches: int = typer.Option(50, min=1),
):
    """Evaluate LipType WER on the validation TFRecords configured in YAML."""
    import tensorflow as tf

    from liptype_rebuild.datasets.input_pipeline import PipelineConfig, make_dataset
    from liptype_rebuild.datasets.tfrecords import ExampleSpec
    from liptype_rebuild.model.ctc_decode import DecodeConfig, ctc_beam_decode, sparse_to_texts
    from liptype_rebuild.model.liptype import LipTypeConfig, build_models
    from liptype_rebuild.utils.config import load_yaml
    from liptype_rebuild.utils.metrics import RunningAverage, wer

    cfg = load_yaml(config)
    ds_cfg = cfg["dataset"]
    tr_cfg = cfg["train"]
    dec_cfg = cfg.get("decode", {})

    spec = ExampleSpec(
        max_frames=int(ds_cfg["max_frames"]),
        height=int(ds_cfg["img_h"]),
        width=int(ds_cfg["img_w"]),
        channels=int(ds_cfg["img_c"]),
        max_text_len=int(ds_cfg["max_text_len"]),
    )
    pipe_cfg = PipelineConfig(
        shuffle_buffer=int(tr_cfg.get("shuffle_buffer", 2048)),
        batch_size=int(tr_cfg["batch_size"]),
        seed=int(tr_cfg.get("seed", 55)),
        flip_prob=0.0,
        temporal_jitter_prob=0.0,
    )
    model_cfg = LipTypeConfig(
        img_h=spec.height,
        img_w=spec.width,
        img_c=spec.channels,
        frames_n=spec.max_frames,
        max_text_len=spec.max_text_len,
        output_size=28,
    )

    _train, infer = build_models(model_cfg)
    infer.load_weights(str(weights))

    val_ds = make_dataset(ds_cfg["tfrecords_val"], spec, pipe_cfg, training=False)
    decode_cfg = DecodeConfig(beam_width=int(dec_cfg.get("beam_width", 50)))

    from liptype_rebuild.datasets.labels import Charset

    cs = Charset()
    avg = RunningAverage()
    for x, _y in val_ds.take(num_batches):
        probs = infer(x["frames"], training=False)
        inp_len = tf.squeeze(x["input_len"], axis=1)
        sp = ctc_beam_decode(probs, inp_len, decode_cfg)
        hyps = sparse_to_texts(sp, cs)

        labels = x["labels"].numpy()
        label_lens = tf.squeeze(x["label_len"], axis=1).numpy().tolist()
        refs = []
        for lab, ll in zip(labels, label_lens):
            refs.append(cs.labels_to_text([int(z) for z in lab[:ll] if int(z) >= 0]))
        for h, r in zip(hyps, refs):
            avg = avg.add(wer(h, r))

    typer.echo(f"val_wer={avg.mean:.4f} over {avg.n} samples")

