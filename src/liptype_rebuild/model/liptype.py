from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from liptype_rebuild.model.se_resnet import build_se_resnet34


@dataclass(frozen=True)
class LipTypeConfig:
    img_h: int = 50
    img_w: int = 100
    img_c: int = 3
    frames_n: int = 75
    max_text_len: int = 32
    output_size: int = 28  # 26 letters + space + CTC blank

    conv3d_filters: int = 32
    conv3d_kernel: tuple[int, int, int] = (5, 7, 7)
    conv3d_strides: tuple[int, int, int] = (1, 2, 2)

    se_width_mult: float = 1.0
    gru_units: int = 256
    dropout: float = 0.5


def _ctc_loss_lambda(args):
    y_pred, labels, input_len, label_len = args
    # y_pred: [B,T,C]
    # labels: [B,L] with -1 padding
    # input_len, label_len: [B,1]
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_len, label_len)


def build_models(cfg: LipTypeConfig):
    """Return (training_model, inference_model)."""
    frames = tf.keras.Input(
        shape=(cfg.frames_n, cfg.img_h, cfg.img_w, cfg.img_c),
        dtype=tf.float32,
        name="frames",
    )
    labels = tf.keras.Input(shape=(cfg.max_text_len,), dtype=tf.int32, name="labels")
    input_len = tf.keras.Input(shape=(1,), dtype=tf.int32, name="input_len")
    label_len = tf.keras.Input(shape=(1,), dtype=tf.int32, name="label_len")

    # 1-layer 3D CNN frontend
    x = tf.keras.layers.ZeroPadding3D(padding=(2, 3, 3))(frames)
    x = tf.keras.layers.Conv3D(
        cfg.conv3d_filters,
        cfg.conv3d_kernel,
        strides=cfg.conv3d_strides,
        padding="valid",
        use_bias=False,
        kernel_initializer="he_normal",
        name="conv3d_frontend",
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.SpatialDropout3D(cfg.dropout)(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding="same")(x)

    # TimeDistributed SE-ResNet34 (per frame)
    # x is [B,T,H,W,C]
    channels = x.shape[-1]
    if channels is None:
        raise ValueError("Channel dimension must be known for SE-ResNet.")
    se = build_se_resnet34(input_shape=(None, None, int(channels)), width_mult=cfg.se_width_mult)
    x = tf.keras.layers.TimeDistributed(se, name="td_se_resnet34")(x)

    # Sequence modeling
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(cfg.gru_units, return_sequences=True, kernel_initializer="orthogonal"),
        merge_mode="concat",
    )(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(cfg.gru_units, return_sequences=True, kernel_initializer="orthogonal"),
        merge_mode="concat",
    )(x)
    logits = tf.keras.layers.Dense(cfg.output_size, kernel_initializer="he_normal")(x)
    y_pred = tf.keras.layers.Activation("softmax", name="softmax")(logits)

    loss_out = tf.keras.layers.Lambda(_ctc_loss_lambda, name="ctc")([y_pred, labels, input_len, label_len])

    training_model = tf.keras.Model(
        inputs={"frames": frames, "labels": labels, "input_len": input_len, "label_len": label_len},
        outputs=loss_out,
        name="liptype_training",
    )
    inference_model = tf.keras.Model(inputs=frames, outputs=y_pred, name="liptype_infer")
    return training_model, inference_model

