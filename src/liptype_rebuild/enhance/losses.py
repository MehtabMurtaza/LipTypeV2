from __future__ import annotations

import tensorflow as tf


class MSSSIML1(tf.keras.losses.Loss):
    """MS-SSIM + L1 combined loss (paper uses MSSSIM-L1 with alpha≈0.816)."""

    def __init__(
        self,
        alpha: float = 0.816,
        warmup_steps: int = 0,
        name: str = "msssim_l1",
    ):
        super().__init__(name=name)
        self.alpha = float(alpha)
        self.warmup_steps = int(warmup_steps)
        # Training-step counter for warmup scheduling. Updated by a callback.
        self.step = tf.Variable(0, trainable=False, dtype=tf.int64, name=f"{name}_step")

    def call(self, y_true, y_pred):
        y_true = tf.clip_by_value(y_true, 0.0, 1.0)
        y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
        # Note: In TF 2.10, `filter_size` must be a Python int (not a Tensor).
        # This loss assumes inputs have been resized to a sufficiently large size.
        ms_ssim = tf.image.ssim_multiscale(y_true, y_pred, max_val=1.0, filter_size=11)
        l1 = tf.reduce_mean(tf.abs(y_true - y_pred), axis=[1, 2, 3])

        # Safety: if MS-SSIM ever becomes non-finite for a batch, ignore it for that batch
        # instead of propagating NaNs into the training loop.
        ms_ssim = tf.where(tf.math.is_finite(ms_ssim), ms_ssim, tf.zeros_like(ms_ssim))
        l1 = tf.where(tf.math.is_finite(l1), l1, tf.zeros_like(l1))
        # Warmup: start with pure L1 (stable), then ramp MS-SSIM weight up to `alpha`.
        if self.warmup_steps > 0:
            w = tf.cast(self.step, tf.float32) / float(self.warmup_steps)
            w = tf.clip_by_value(w, 0.0, 1.0)
            ms_w = w * self.alpha
            l1_w = 1.0 - ms_w
            loss = ms_w * (1.0 - ms_ssim) + l1_w * l1
            return tf.where(tf.math.is_finite(loss), loss, l1)

        # Default (paper): fixed alpha
        loss = self.alpha * (1.0 - ms_ssim) + (1.0 - self.alpha) * l1
        return tf.where(tf.math.is_finite(loss), loss, l1)

