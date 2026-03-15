from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from liptype_rebuild.datasets.tfrecords import ExampleSpec, parse_example


@dataclass(frozen=True)
class QualPipelineConfig:
    batch_size: int = 1
    num_parallel_calls: int = tf.data.AUTOTUNE


def _ctc_required_time(labels_1d: tf.Tensor, label_len_scalar: tf.Tensor) -> tf.Tensor:
    """CTC requires enough timesteps for label transitions.

    Minimum required time is label_len + repeats, where repeats counts adjacent equal labels
    (CTC needs an extra blank to separate identical consecutive tokens).
    """
    ll = tf.cast(label_len_scalar, tf.int32)
    seq = labels_1d[:ll]
    repeats = tf.reduce_sum(tf.cast(tf.equal(seq[1:], seq[:-1]), tf.int32))
    return ll + repeats


def make_qual_dataset(
    tfrecord_glob: str,
    spec: ExampleSpec,
    cfg: QualPipelineConfig = QualPipelineConfig(),
) -> tf.data.Dataset:
    """Dataset for qualitative reporting. Keeps ids and lengths."""
    files = tf.io.gfile.glob(tfrecord_glob)
    if not files:
        raise FileNotFoundError(f"No TFRecord files matched: {tfrecord_glob}")

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)

    def _parse(x):
        ex = parse_example(x, spec)
        # keep ids; normalize len shapes
        return {
            "frames": ex["frames"],
            "labels": ex["label"],
            "input_len": tf.expand_dims(tf.cast(ex["input_len"], tf.int32), axis=0),
            "label_len": tf.expand_dims(tf.cast(ex["label_len"], tf.int32), axis=0),
            "utterance_id": ex["utterance_id"],
            "speaker_id": ex["speaker_id"],
        }

    def _is_valid(ex) -> tf.Tensor:
        input_len = tf.squeeze(ex["input_len"], axis=0)
        label_len = tf.squeeze(ex["label_len"], axis=0)
        required = _ctc_required_time(ex["labels"], label_len)
        return tf.logical_and(input_len > 0, tf.logical_and(label_len > 0, input_len >= required))

    ds = ds.map(_parse, num_parallel_calls=cfg.num_parallel_calls)
    ds = ds.filter(_is_valid)
    ds = ds.batch(int(cfg.batch_size), drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

