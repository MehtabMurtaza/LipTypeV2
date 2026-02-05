from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import tensorflow as tf

from liptype_rebuild.datasets.tfrecords import ExampleSpec, parse_example


@dataclass(frozen=True)
class PipelineConfig:
    shuffle_buffer: int = 2048
    batch_size: int = 8
    num_parallel_calls: int = tf.data.AUTOTUNE
    seed: int = 55

    # augmentation
    flip_prob: float = 0.5
    temporal_jitter_prob: float = 0.05


def _maybe_flip(frames: tf.Tensor, p: float) -> tf.Tensor:
    """frames: [T,H,W,C]"""
    do = tf.less(tf.random.uniform([], 0, 1.0), p)
    return tf.cond(do, lambda: tf.image.flip_left_right(frames), lambda: frames)


def _temporal_jitter(frames: tf.Tensor, input_len: tf.Tensor, p: float) -> tuple[tf.Tensor, tf.Tensor]:
    """Randomly drop/duplicate frames with small probability, then pad/truncate back to T.

    This is a simple approximation of the old code's delete/dup jitter.
    """
    t = tf.shape(frames)[0]
    # choose per-frame op: -1=drop, 0=keep, +1=dup
    r = tf.random.uniform([t], 0, 1.0)
    drop = r < (p / 2.0)
    dup = (r >= (p / 2.0)) & (r < p)

    idx = tf.range(t)
    keep_idx = tf.boolean_mask(idx, ~drop)
    kept = tf.gather(frames, keep_idx)

    # duplicate selected indices (from keep_idx's perspective)
    dup_idx = tf.boolean_mask(idx, dup & ~drop)
    dup_frames = tf.gather(frames, dup_idx)
    jittered = tf.concat([kept, dup_frames], axis=0)

    # If we changed ordering, we don't preserve timing; shuffle slightly to avoid huge bias
    jittered = tf.random.shuffle(jittered)

    # restore to fixed length t
    cur = tf.shape(jittered)[0]
    jittered = jittered[:t]
    pad = tf.maximum(t - tf.shape(jittered)[0], 0)
    jittered = tf.cond(
        pad > 0,
        lambda: tf.concat([jittered, tf.zeros([pad, tf.shape(frames)[1], tf.shape(frames)[2], tf.shape(frames)[3]], frames.dtype)], axis=0),
        lambda: jittered,
    )
    new_len = tf.minimum(tf.cast(cur, tf.int32), input_len)
    new_len = tf.minimum(new_len, t)
    return jittered, new_len


def make_dataset(
    tfrecord_glob: str,
    spec: ExampleSpec,
    cfg: PipelineConfig,
    training: bool,
) -> tf.data.Dataset:
    files = tf.io.gfile.glob(tfrecord_glob)
    if not files:
        raise FileNotFoundError(f"No TFRecord files matched: {tfrecord_glob}")

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(cfg.shuffle_buffer, seed=cfg.seed, reshuffle_each_iteration=True)

    def _parse(x):
        ex = parse_example(x, spec)
        frames = ex["frames"]  # [T,H,W,C] float
        label = ex["label"]  # [L]
        input_len = ex["input_len"]  # scalar
        label_len = ex["label_len"]

        if training:
            frames = _maybe_flip(frames, cfg.flip_prob)
            frames, input_len2 = _temporal_jitter(frames, input_len, cfg.temporal_jitter_prob)
            input_len = input_len2

        # model expects [T,H,W,C]; batch will make [B,T,H,W,C]
        return {
            "frames": frames,
            "labels": label,
            "input_len": tf.expand_dims(tf.cast(input_len, tf.int32), axis=0),
            "label_len": tf.expand_dims(tf.cast(label_len, tf.int32), axis=0),
        }, tf.zeros([1], tf.float32)

    ds = ds.map(_parse, num_parallel_calls=cfg.num_parallel_calls)
    ds = ds.batch(cfg.batch_size, drop_remainder=training)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

