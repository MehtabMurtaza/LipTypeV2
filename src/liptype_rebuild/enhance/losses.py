from __future__ import annotations

import tensorflow as tf


class MSSSIML1(tf.keras.losses.Loss):
    """MS-SSIM + L1 combined loss (paper uses MSSSIM-L1 with alpha≈0.816)."""

    def __init__(self, alpha: float = 0.816, name: str = "msssim_l1"):
        super().__init__(name=name)
        self.alpha = float(alpha)

    def call(self, y_true, y_pred):
        y_true = tf.clip_by_value(y_true, 0.0, 1.0)
        y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
        ms_ssim = tf.image.ssim_multiscale(y_true, y_pred, max_val=1.0)
        l1 = tf.reduce_mean(tf.abs(y_true - y_pred), axis=[1, 2, 3])
        # loss: minimize (1 - ms_ssim) and l1
        return self.alpha * (1.0 - ms_ssim) + (1.0 - self.alpha) * l1

