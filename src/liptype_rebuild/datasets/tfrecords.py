from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _bytes_feature(value: bytes):
    import tensorflow as tf

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_list_feature(values: list[int]):
    import tensorflow as tf

    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _int64_feature(value: int):
    import tensorflow as tf

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


@dataclass(frozen=True)
class ExampleSpec:
    max_frames: int = 75
    height: int = 50
    width: int = 100
    channels: int = 3
    max_text_len: int = 32


def make_example(
    frames_uint8: np.ndarray,  # [T,H,W,C]
    label: list[int],
    utterance_id: str,
    speaker_id: str,
    spec: ExampleSpec,
):
    import tensorflow as tf

    assert frames_uint8.ndim == 4
    t, h, w, c = frames_uint8.shape
    if (h, w, c) != (spec.height, spec.width, spec.channels):
        raise ValueError(f"Frame shape mismatch: {(h,w,c)} != {(spec.height,spec.width,spec.channels)}")

    # Pad/truncate frames to max_frames
    t_use = min(t, spec.max_frames)
    frames = frames_uint8[:t_use]
    if t_use < spec.max_frames:
        pad = np.zeros((spec.max_frames - t_use, h, w, c), dtype=np.uint8)
        frames = np.concatenate([frames, pad], axis=0)

    label_use = label[: spec.max_text_len]
    label_len = len(label_use)
    label_padded = label_use + [-1] * (spec.max_text_len - label_len)

    ex = tf.train.Example(
        features=tf.train.Features(
            feature={
                "frames": _bytes_feature(frames.tobytes()),
                "t": _int64_feature(spec.max_frames),
                "h": _int64_feature(h),
                "w": _int64_feature(w),
                "c": _int64_feature(c),
                "label": _int64_list_feature(label_padded),
                "label_len": _int64_feature(label_len),
                "input_len": _int64_feature(t_use),
                "utterance_id": _bytes_feature(utterance_id.encode("utf-8")),
                "speaker_id": _bytes_feature(speaker_id.encode("utf-8")),
            }
        )
    )
    return ex


def parse_example(serialized: bytes, spec: ExampleSpec):
    """Return dict with frames(float32 [T,H,W,C]), label(int32 [L]), lengths, ids."""
    import tensorflow as tf

    feature_spec = {
        "frames": tf.io.FixedLenFeature([], tf.string),
        "t": tf.io.FixedLenFeature([], tf.int64),
        "h": tf.io.FixedLenFeature([], tf.int64),
        "w": tf.io.FixedLenFeature([], tf.int64),
        "c": tf.io.FixedLenFeature([], tf.int64),
        "label": tf.io.FixedLenFeature([spec.max_text_len], tf.int64),
        "label_len": tf.io.FixedLenFeature([], tf.int64),
        "input_len": tf.io.FixedLenFeature([], tf.int64),
        "utterance_id": tf.io.FixedLenFeature([], tf.string),
        "speaker_id": tf.io.FixedLenFeature([], tf.string),
    }
    ex = tf.io.parse_single_example(serialized, feature_spec)
    t = tf.cast(ex["t"], tf.int32)
    h = tf.cast(ex["h"], tf.int32)
    w = tf.cast(ex["w"], tf.int32)
    c = tf.cast(ex["c"], tf.int32)
    frames = tf.io.decode_raw(ex["frames"], tf.uint8)
    frames = tf.reshape(frames, [t, h, w, c])
    frames = tf.cast(frames, tf.float32) / 255.0
    label = tf.cast(ex["label"], tf.int32)
    return {
        "frames": frames,
        "label": label,
        "label_len": tf.cast(ex["label_len"], tf.int32),
        "input_len": tf.cast(ex["input_len"], tf.int32),
        "utterance_id": ex["utterance_id"],
        "speaker_id": ex["speaker_id"],
    }

