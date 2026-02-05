from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from liptype_rebuild.datasets.labels import Charset, CTC_BLANK


@dataclass(frozen=True)
class DecodeConfig:
    beam_width: int = 50
    top_paths: int = 1


def _dense_to_sparse(dense: tf.Tensor, lengths: tf.Tensor) -> tf.SparseTensor:
    """dense: [B,T] int32, lengths: [B]"""
    batch = tf.shape(dense)[0]
    max_t = tf.shape(dense)[1]
    mask = tf.sequence_mask(lengths, maxlen=max_t)
    idx = tf.where(mask)
    vals = tf.gather_nd(dense, idx)
    return tf.SparseTensor(indices=idx, values=vals, dense_shape=tf.cast([batch, max_t], tf.int64))


def ctc_beam_decode(
    probs: tf.Tensor,  # [B,T,C] float32 softmax
    input_len: tf.Tensor,  # [B] int32
    cfg: DecodeConfig = DecodeConfig(),
) -> tf.SparseTensor:
    """Beam decode using TF's CTC beam search decoder.

    Returns best path as SparseTensor with values in [0..C-1] excluding blank.
    """
    # Convert to logit-like time-major for decoder: [T,B,C]
    log_probs = tf.math.log(tf.clip_by_value(probs, 1e-8, 1.0))
    time_major = tf.transpose(log_probs, [1, 0, 2])
    decoded, _ = tf.nn.ctc_beam_search_decoder(
        inputs=time_major,
        sequence_length=tf.cast(input_len, tf.int32),
        beam_width=int(cfg.beam_width),
        top_paths=int(cfg.top_paths),
    )
    return decoded[0]


def sparse_to_texts(sp: tf.SparseTensor, charset: Charset | None = None) -> list[str]:
    if charset is None:
        charset = Charset()
    # Convert sparse to list of lists
    sp = tf.sparse.reorder(sp)
    idx = sp.indices.numpy()
    vals = sp.values.numpy().tolist()
    # group by batch
    out: list[list[int]] = []
    cur_b = -1
    cur: list[int] = []
    for (b, _t), v in zip(idx, vals):
        if b != cur_b:
            if cur_b >= 0:
                out.append(cur)
            cur_b = int(b)
            cur = []
        if int(v) != CTC_BLANK:
            cur.append(int(v))
    if cur_b >= 0:
        out.append(cur)
    return [charset.labels_to_text(seq) for seq in out]

