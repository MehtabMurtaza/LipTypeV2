from __future__ import annotations

import tensorflow as tf


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, channels: int, reduction: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(max(channels // reduction, 1), activation="relu")
        self.fc2 = tf.keras.layers.Dense(channels, activation="sigmoid")

    def call(self, x, training=None):
        s = self.gap(x)
        s = self.fc1(s)
        s = self.fc2(s)
        s = tf.reshape(s, [-1, 1, 1, self.channels])
        return x * s


def conv_bn_relu(x, filters: int, kernel_size: int | tuple[int, int], strides=1):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


def basic_block(x, filters: int, strides: int = 1, use_se: bool = True):
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False)(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    if use_se:
        x = SEBlock(filters)(x)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation("relu")(x)
    return x


def build_se_resnet34(input_shape: tuple[int, int, int], width_mult: float = 1.0) -> tf.keras.Model:
    """A compact SE-ResNet34 producing a per-frame feature vector.

    This is a faithful-structure replacement for the paper's 34-layer 2D SE-ResNet.
    Output: [B, F] embedding (global average pooled).
    """
    w1 = int(64 * width_mult)
    w2 = int(128 * width_mult)
    w3 = int(256 * width_mult)
    w4 = int(512 * width_mult)

    inp = tf.keras.Input(shape=input_shape)
    x = conv_bn_relu(inp, w1, 7, strides=2)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)

    # ResNet34 block counts: [3,4,6,3]
    for _ in range(3):
        x = basic_block(x, w1, strides=1)
    x = basic_block(x, w2, strides=2)
    for _ in range(3):
        x = basic_block(x, w2, strides=1)
    x = basic_block(x, w3, strides=2)
    for _ in range(5):
        x = basic_block(x, w3, strides=1)
    x = basic_block(x, w4, strides=2)
    for _ in range(2):
        x = basic_block(x, w4, strides=1)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inp, x, name="se_resnet34")

