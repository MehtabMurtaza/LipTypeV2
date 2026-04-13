from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from liptype_rebuild.postprocess.dda_data import ids_to_sentence, pair_to_matrices, sentence_to_ids


@dataclass(frozen=True)
class DDAConfig:
    hidden: tuple[int, ...] = (128, 64, 32, 64, 128)
    seq_len: int = 28
    vocab_dim: int = 28

    @property
    def input_dim(self) -> int:
        return int(self.seq_len * self.vocab_dim)


def build_dda(cfg: DDAConfig):
    """Build sequence-level DDA MLP (784 -> 128 -> 64 -> 32 -> 64 -> 128 -> 784).

    Input/output represent a flattened [seq_len, vocab_dim] one-hot matrix.
    """
    import tensorflow as tf

    inp = tf.keras.Input(shape=(cfg.input_dim,), name="dda_in")
    x = inp
    for i, h in enumerate(cfg.hidden):
        x = tf.keras.layers.Dense(h, activation="tanh", name=f"dda_dense_{i}")(x)
    out = tf.keras.layers.Dense(cfg.input_dim, activation="sigmoid", name="dda_out")(x)
    return tf.keras.Model(inp, out, name="dda")


def apply_dda_to_text(dda, text: str, cfg: DDAConfig = DDAConfig()) -> str:
    """Apply trained DDA to text and return corrected text.

    This uses fixed-length chunking (seq_len=28 by default) and newline padding,
    then decodes argmax characters back to text.
    """
    ids = sentence_to_ids(text)
    if not ids:
        return ""

    # Reuse pair encoder for chunking; labels are ignored.
    x, _y = pair_to_matrices(text, text, seq_len=cfg.seq_len)
    flat = x.reshape((-1, cfg.input_dim))
    pred = dda(flat, training=False).numpy().reshape((-1, cfg.seq_len, cfg.vocab_dim))
    ids_out: list[int] = []
    for chunk in pred:
        ids_out.extend(np.argmax(chunk, axis=1).tolist())
    return ids_to_sentence(ids_out)


def load_dda(weights_path: str | None = None, cfg: DDAConfig = DDAConfig()):
    model = build_dda(cfg)
    if weights_path:
        model.load_weights(str(weights_path))
    return model

