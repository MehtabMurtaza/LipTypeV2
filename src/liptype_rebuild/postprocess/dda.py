from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DDAConfig:
    hidden: tuple[int, ...] = (128, 64, 32, 64, 128)
    input_dim: int = 28


def build_dda(cfg: DDAConfig):
    """Build the DDA MLP (28 -> 128 -> 64 -> 32 -> 64 -> 128 -> 28).

    This model is applied per time-step to a 28-dim vector.
    """
    import tensorflow as tf

    inp = tf.keras.Input(shape=(cfg.input_dim,), name="dda_in")
    x = inp
    for i, h in enumerate(cfg.hidden):
        x = tf.keras.layers.Dense(h, activation="tanh", name=f"dda_dense_{i}")(x)
    out = tf.keras.layers.Dense(cfg.input_dim, activation="softmax", name="dda_out")(x)
    return tf.keras.Model(inp, out, name="dda")


def apply_dda_to_sequence(dda, probs):
    """Apply DDA to a sequence of character probabilities.

    probs: np.ndarray or tf.Tensor [T,28] or [B,T,28]
    returns same shape.
    """
    import tensorflow as tf

    p = tf.convert_to_tensor(probs, dtype=tf.float32)
    if p.shape.rank == 2:
        return dda(p, training=False)
    if p.shape.rank == 3:
        b, t, c = tf.unstack(tf.shape(p))
        flat = tf.reshape(p, [b * t, c])
        out = dda(flat, training=False)
        return tf.reshape(out, [b, t, c])
    raise ValueError(f"Unexpected probs rank: {p.shape.rank}")

