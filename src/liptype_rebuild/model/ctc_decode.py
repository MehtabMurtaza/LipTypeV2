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


def ctc_beam_decode_nbest(
    probs: tf.Tensor,  # [B,T,C] float32 softmax
    input_len: tf.Tensor,  # [B] int32
    cfg: DecodeConfig = DecodeConfig(),
) -> tuple[list[tf.SparseTensor], np.ndarray]:
    """Return top-N beam paths and CTC log-probabilities.

    Returns:
      - decoded_paths: list length N (`cfg.top_paths`) of SparseTensor
      - log_probs: float32 array [B, N], aligned with decoded_paths
    """
    log_probs = tf.math.log(tf.clip_by_value(probs, 1e-8, 1.0))
    time_major = tf.transpose(log_probs, [1, 0, 2])
    decoded, beam_log_probs = tf.nn.ctc_beam_search_decoder(
        inputs=time_major,
        sequence_length=tf.cast(input_len, tf.int32),
        beam_width=int(cfg.beam_width),
        top_paths=int(cfg.top_paths),
    )
    # TensorFlow returns [N, B]. We use [B, N].
    scores = tf.transpose(beam_log_probs, [1, 0]).numpy().astype(np.float32)
    return decoded, scores


def sparse_to_texts(sp: tf.SparseTensor, charset: Charset | None = None, batch_size: int | None = None) -> list[str]:
    if charset is None:
        charset = Charset()

    if batch_size is None:
        try:
            batch_size = int(sp.dense_shape[0])
        except Exception:
            batch_size = None

    # Convert sparse to list of lists
    sp = tf.sparse.reorder(sp)
    idx = sp.indices.numpy()
    vals = sp.values.numpy().tolist()

    if batch_size is not None:
        out: list[list[int]] = [[] for _ in range(int(batch_size))]
    else:
        out = []
    cur_b = -1
    cur: list[int] = []
    for (b, _t), v in zip(idx, vals):
        b = int(b)
        if b != cur_b:
            if cur_b >= 0:
                if batch_size is None:
                    out.append(cur)
                else:
                    out[cur_b] = cur
            cur_b = int(b)
            cur = []
        if int(v) != CTC_BLANK:
            cur.append(int(v))
    if cur_b >= 0 and batch_size is None:
        out.append(cur)

    if batch_size is not None and cur_b >= 0:
        out[cur_b] = cur

    return [charset.labels_to_text(seq) for seq in out]


def nbest_sparse_to_texts(
    decoded_paths: list[tf.SparseTensor],
    charset: Charset | None = None,
    batch_size: int | None = None,
) -> list[list[str]]:
    """Convert top-N SparseTensor paths into per-sample candidate lists.

    Returns:
      candidates_by_sample: list length B, each element list length N of texts.
    """
    if charset is None:
        charset = Charset()
    if not decoded_paths:
        return []

    if batch_size is None:
        try:
            batch_size = int(decoded_paths[0].dense_shape[0])
        except Exception:
            batch_size = 0

    n = len(decoded_paths)
    per_path_texts = [sparse_to_texts(sp, charset=charset, batch_size=batch_size) for sp in decoded_paths]
    out: list[list[str]] = []
    for i in range(int(batch_size)):
        out.append([per_path_texts[j][i] for j in range(n)])
    return out

