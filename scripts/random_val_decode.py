#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Args:
    config: str
    weights: str
    split: str
    num_samples: int
    seed: int
    beam_width: int


def _parse_args() -> Args:
    p = argparse.ArgumentParser(
        description=(
            "Sample random examples from validation TFRecords, run LipType inference, and print REF/HYP/WER."
        )
    )
    p.add_argument("--config", required=True, help="Path to YAML config (same one used for training/eval).")
    p.add_argument("--weights", required=True, help="Path to Keras weights file (*.weights.h5).")
    p.add_argument(
        "--split",
        choices=["val", "train"],
        default="val",
        help="Which TFRecords split to sample from (val is recommended).",
    )
    p.add_argument("--num-samples", type=int, default=5, help="How many random validation samples to print.")
    p.add_argument("--seed", type=int, default=55, help="Shuffle seed.")
    p.add_argument("--beam-width", type=int, default=50, help="CTC beam width for decoding.")
    a = p.parse_args()
    return Args(
        config=str(a.config),
        weights=str(a.weights),
        split=str(a.split),
        num_samples=int(a.num_samples),
        seed=int(a.seed),
        beam_width=int(a.beam_width),
    )


def main() -> None:
    import tensorflow as tf

    from liptype_rebuild.datasets.labels import Charset
    from liptype_rebuild.datasets.tfrecords import ExampleSpec, parse_example
    from liptype_rebuild.model.ctc_decode import DecodeConfig, ctc_beam_decode, sparse_to_texts
    from liptype_rebuild.model.liptype import LipTypeConfig, build_models
    from liptype_rebuild.utils.config import load_yaml
    from liptype_rebuild.utils.metrics import wer

    args = _parse_args()
    from pathlib import Path

    cfg = load_yaml(Path(args.config))
    ds_cfg = cfg["dataset"]

    spec = ExampleSpec(
        max_frames=int(ds_cfg["max_frames"]),
        height=int(ds_cfg["img_h"]),
        width=int(ds_cfg["img_w"]),
        channels=int(ds_cfg["img_c"]),
        max_text_len=int(ds_cfg["max_text_len"]),
    )

    # Build inference model
    model_cfg = LipTypeConfig(
        img_h=spec.height,
        img_w=spec.width,
        img_c=spec.channels,
        frames_n=spec.max_frames,
        max_text_len=spec.max_text_len,
        output_size=28,
    )
    _train, infer = build_models(model_cfg)
    infer.load_weights(args.weights)

    key = "tfrecords_val" if args.split == "val" else "tfrecords_train"
    tfrecord_glob = ds_cfg[key]
    files = tf.io.gfile.glob(tfrecord_glob)
    if not files:
        raise FileNotFoundError(f"No TFRecord files matched: {tfrecord_glob}")

    # Parse TFRecords directly so we can access utterance_id/speaker_id for the printed output.
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)

    def _parse(x: tf.Tensor) -> dict:
        return parse_example(x, spec)

    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(4096, seed=args.seed, reshuffle_each_iteration=True)
    ds = ds.take(args.num_samples)

    cs = Charset()
    dec = DecodeConfig(beam_width=args.beam_width)

    for i, ex in enumerate(ds):
        frames = ex["frames"]  # [T,H,W,C] float32
        input_len = tf.cast(ex["input_len"], tf.int32)  # scalar

        probs = infer(tf.expand_dims(frames, 0), training=False)  # [1,T,V]
        sp = ctc_beam_decode(probs, tf.expand_dims(input_len, 0), dec)
        hyp = (sparse_to_texts(sp, cs) or [""])[0]

        label = ex["label"].numpy().tolist()
        label_len = int(ex["label_len"].numpy())
        ref = cs.labels_to_text([int(z) for z in label[:label_len] if int(z) >= 0])

        utt = ex["utterance_id"].numpy().decode("utf-8", errors="ignore")
        spk = ex["speaker_id"].numpy().decode("utf-8", errors="ignore")
        w = wer(hyp, ref)

        print(f"\n[{i}] speaker_id={spk} utterance_id={utt} input_len={int(ex['input_len'].numpy())}")
        print("REF:", ref)
        print("HYP:", hyp)
        print("WER:", f"{w:.4f}")


if __name__ == "__main__":
    main()

