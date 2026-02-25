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
    # Use bilinear interpolation for stable gradients (also compatible with XLA if enabled).
    x = tf.keras.layers.Resizing(96, 96, interpolation="bilinear", name="resize_in_96")(inp)

    # Encoder (3 down blocks)
    e1 = _conv(x, 64, 3, 2)  # 48
    e2 = _conv(e1, 64, 3, 2)  # 24
    e3 = _conv(e2, 64, 3, 2)  # 12

    # Decoder (3 up blocks) with skip connections
    d1 = tf.keras.layers.Resizing(24, 24, interpolation="bilinear", name="resize_d1_24")(e3)
    d1 = _conv(d1, 64, 3, 1)
    d1 = tf.keras.layers.Add()([d1, e2])

    d2 = tf.keras.layers.Resizing(48, 48, interpolation="bilinear", name="resize_d2_48")(d1)
    d2 = _conv(d2, 64, 3, 1)
    d2 = tf.keras.layers.Add()([d2, e1])

    d3 = tf.keras.layers.Resizing(96, 96, interpolation="bilinear", name="resize_d3_96")(d2)
    d3 = _conv(d3, 64, 3, 1)

    # Upsample illumination features back to original resolution
    illum = tf.keras.layers.Lambda(
        lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3], method="bilinear", antialias=False),
        name="resize_illum_to_input",
    )([d3, inp])

    # --- Detail reconstruction ---
    a = tf.keras.layers.Concatenate(axis=-1)([illum, inp])
    a = _conv(a, 128, 3, 1)
    a = _conv(a, 128, 3, 1)
    out = _conv(a, 3, 3, 1, act="sigmoid")  # keep in [0,1]

    return tf.keras.Model(inp, out, name="gladnet_rebuild")

