from __future__ import annotations

import tensorflow as tf


def _conv(x, filters: int, k: int = 3, s: int = 1, act: str | None = "relu", name: str | None = None):
    x = tf.keras.layers.Conv2D(filters, k, strides=s, padding="same", use_bias=True, name=name)(x)
    if act:
        x = tf.keras.layers.Activation(act)(x)
    return x


def build_gladnet(input_shape: tuple[int | None, int | None, int] = (None, None, 3)) -> tf.keras.Model:
    """GLADNet-like low-light enhancement network (paper: 3 down + 3 up blocks).

    Input/Output: float32 image in [0,1], shape [H,W,3].
    """
    inp = tf.keras.Input(shape=input_shape, name="low_light")

    # --- Global illumination estimation (downsample to 96x96) ---
    x = tf.image.resize(inp, (96, 96), method="nearest")

    # Encoder (3 down blocks)
    e1 = _conv(x, 64, 3, 2)  # 48
    e2 = _conv(e1, 64, 3, 2)  # 24
    e3 = _conv(e2, 64, 3, 2)  # 12

    # Decoder (3 up blocks) with skip connections
    d1 = tf.image.resize(e3, (24, 24), method="nearest")
    d1 = _conv(d1, 64, 3, 1)
    d1 = tf.keras.layers.Add()([d1, e2])

    d2 = tf.image.resize(d1, (48, 48), method="nearest")
    d2 = _conv(d2, 64, 3, 1)
    d2 = tf.keras.layers.Add()([d2, e1])

    d3 = tf.image.resize(d2, (96, 96), method="nearest")
    d3 = _conv(d3, 64, 3, 1)

    # Upsample illumination features back to original resolution
    illum = tf.image.resize(d3, tf.shape(inp)[1:3], method="nearest")

    # --- Detail reconstruction ---
    a = tf.keras.layers.Concatenate(axis=-1)([illum, inp])
    a = _conv(a, 128, 3, 1)
    a = _conv(a, 128, 3, 1)
    out = _conv(a, 3, 3, 1, act="sigmoid")  # keep in [0,1]

    return tf.keras.Model(inp, out, name="gladnet_rebuild")

