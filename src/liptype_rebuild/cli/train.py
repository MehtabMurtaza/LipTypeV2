from __future__ import annotations

from pathlib import Path
import re

import typer

app = typer.Typer(help="Train models (LipType, enhancement, postprocess).")


@app.command("liptype")
def train_liptype(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    run_dir: Path = typer.Option(..., file_okay=False, dir_okay=True),
    resume_weights: Path = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Optional: path to weights.###.weights.h5 to resume from (warm-start).",
    ),
    resume_checkpoint: Path = typer.Option(
        None,
        help=(
            "Optional: resume from a TensorFlow checkpoint to restore model + optimizer state. "
            "Pass either the checkpoint directory (runs/<run_dir>/ckpt) or a specific prefix like ckpt-000053."
        ),
    ),
    lr_scheduler: bool = typer.Option(
        True,
        "--lr-scheduler",
        help=(
            "If enabled and config has train.lr_schedule, apply a learning-rate scheduler callback. "
            "Disable with --no-lr-scheduler."
        ),
        show_default=True,
    ),
    no_lr_scheduler: bool = typer.Option(
        False,
        "--no-lr-scheduler",
        help="Disable learning-rate scheduler even if config has train.lr_schedule.",
        show_default=True,
    ),
):
    """Train LipType recognizer from a config file."""
    import tensorflow as tf

    from liptype_rebuild.datasets.input_pipeline import PipelineConfig, make_dataset
    from liptype_rebuild.datasets.tfrecords import ExampleSpec
    from liptype_rebuild.model.ctc_decode import DecodeConfig, ctc_beam_decode, sparse_to_texts
    from liptype_rebuild.model.liptype import LipTypeConfig, build_models
    from liptype_rebuild.utils.config import load_yaml
    from liptype_rebuild.utils.metrics import RunningAverage, wer

    cfg = load_yaml(config)

    ds_cfg = cfg["dataset"]
    tr_cfg = cfg["train"]
    aug_cfg = cfg.get("augment", {})
    dec_cfg = cfg.get("decode", {})

    spec = ExampleSpec(
        max_frames=int(ds_cfg["max_frames"]),
        height=int(ds_cfg["img_h"]),
        width=int(ds_cfg["img_w"]),
        channels=int(ds_cfg["img_c"]),
        max_text_len=int(ds_cfg["max_text_len"]),
    )

    pipe_cfg = PipelineConfig(
        shuffle_buffer=int(tr_cfg.get("shuffle_buffer", 2048)),
        batch_size=int(tr_cfg["batch_size"]),
        seed=int(tr_cfg.get("seed", 55)),
        flip_prob=float(aug_cfg.get("horizontal_flip_prob", 0.5)),
        temporal_jitter_prob=float(aug_cfg.get("temporal_jitter_prob", 0.05)),
    )

    model_cfg = LipTypeConfig(
        img_h=spec.height,
        img_w=spec.width,
        img_c=spec.channels,
        frames_n=spec.max_frames,
        max_text_len=spec.max_text_len,
        output_size=28,
    )

    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = make_dataset(ds_cfg["tfrecords_train"], spec, pipe_cfg, training=True)
    val_ds = make_dataset(ds_cfg["tfrecords_val"], spec, pipe_cfg, training=False)

    training_model, inference_model = build_models(model_cfg)
    inference_model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=float(tr_cfg.get("learning_rate", 1e-4)))
    training_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=opt)

    decode_cfg = DecodeConfig(beam_width=int(dec_cfg.get("beam_width", 50)))

    initial_epoch = 0
    # Full checkpointing: model weights + optimizer state + epoch counter.
    ckpt_dir = run_dir / "ckpt"
    epoch_var = tf.Variable(0, trainable=False, dtype=tf.int64, name="epoch")
    ckpt = tf.train.Checkpoint(model=training_model, optimizer=opt, epoch=epoch_var)
    manager = tf.train.CheckpointManager(ckpt, directory=str(ckpt_dir), max_to_keep=5)

    def _restore_checkpoint(path: Path | None) -> bool:
        """Restore from a checkpoint directory or prefix. Returns True if restored.

        Safe behavior:
        - If directory exists but contains no checkpoints, return False (don't error).
        - If prefix/path doesn't exist, return False (don't error).
        """
        if path is None:
            return False

        p = Path(path)
        ckpt_path: str | None = None

        if p.exists() and p.is_dir():
            ckpt_path = tf.train.latest_checkpoint(str(p))
            if not ckpt_path:
                return False
        else:
            # Could be a prefix like ".../ckpt/ckpt-000053" (checkpoint files have suffixes).
            ckpt_path = str(p)

        try:
            status = ckpt.restore(ckpt_path)
            status.expect_partial()
        except Exception as e:
            # Most commonly NotFoundError when the path/prefix doesn't match any checkpoint files.
            typer.echo(f"Could not restore checkpoint from {ckpt_path}: {e}")
            return False

        typer.echo(f"Resumed checkpoint from {ckpt_path} (epoch={int(epoch_var.numpy())}).")
        return True

    restored = False
    if resume_checkpoint is not None:
        restored = _restore_checkpoint(resume_checkpoint)
    if not restored:
        restored = _restore_checkpoint(ckpt_dir)
    if restored:
        initial_epoch = int(epoch_var.numpy())

    if resume_weights is not None:
        # We save weights from `training_model.fit(...)` via ModelCheckpoint(save_weights_only=True),
        # so resuming should load into the training model (it shares weights with inference_model).
        training_model.load_weights(str(resume_weights))
        m = re.search(r"weights\.(\d+)\.weights\.h5$", resume_weights.name)
        if m:
            # Keras epochs are 0-indexed; weights.001 corresponds to epoch=1 completed.
            initial_epoch = int(m.group(1))
        typer.echo(f"Resuming from {resume_weights} (initial_epoch={initial_epoch}).")

    class CheckpointCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # `epoch` is 0-indexed current epoch number; at end of epoch N we set next start to N+1.
            epoch_var.assign(int(epoch) + 1)
            path = manager.save(checkpoint_number=int(epoch_var.numpy()))
            typer.echo(f"Saved checkpoint: {path}")

    class ValWerCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            avg = RunningAverage()
            # evaluate a small number of batches to keep it fast
            for b, (x, _y) in enumerate(val_ds.take(10)):
                probs = inference_model(x["frames"], training=False)
                inp_len = tf.squeeze(x["input_len"], axis=1)
                sp = ctc_beam_decode(probs, inp_len, decode_cfg)
                hyps = sparse_to_texts(sp)
                # ground truth from dense labels (remove -1)
                labels = x["labels"].numpy()
                label_lens = tf.squeeze(x["label_len"], axis=1).numpy().tolist()
                from liptype_rebuild.datasets.labels import Charset

                cs = Charset()
                refs = []
                for lab, ll in zip(labels, label_lens):
                    refs.append(cs.labels_to_text([int(z) for z in lab[:ll] if int(z) >= 0]))
                for h, r in zip(hyps, refs):
                    avg = avg.add(wer(h, r))
            typer.echo(f"[epoch {epoch}] val_wer_estimate={avg.mean:.4f}")

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "weights.{epoch:03d}.weights.h5"),
            save_weights_only=True,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(run_dir / "tb")),
        CheckpointCallback(),
        ValWerCallback(),
    ]

    lr_cfg = tr_cfg.get("lr_schedule") if (lr_scheduler and not no_lr_scheduler) else None
    if lr_cfg:
        sched_type = str(lr_cfg.get("type", "reduce_on_plateau")).strip().lower()
        if sched_type in ("reduce_on_plateau", "plateau", "rop"):
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=str(lr_cfg.get("monitor", "val_loss")),
                    factor=float(lr_cfg.get("factor", 0.5)),
                    patience=int(lr_cfg.get("patience", 4)),
                    verbose=1,
                    mode=str(lr_cfg.get("mode", "auto")),
                    min_delta=float(lr_cfg.get("min_delta", 0.0)),
                    cooldown=int(lr_cfg.get("cooldown", 0)),
                    min_lr=float(lr_cfg.get("min_lr", 1e-6)),
                )
            )
        else:
            raise typer.BadParameter(
                f"Unsupported train.lr_schedule.type={sched_type!r}. "
                "Supported: reduce_on_plateau."
            )

    training_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(tr_cfg.get("epochs", 2)),
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )


@app.command("gladnet")
def train_gladnet(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    run_dir: Path = typer.Option(..., file_okay=False, dir_okay=True),
):
    """Train GLADNet-like enhancer from paired low/normal image folders."""
    import random
    import tensorflow as tf

    from liptype_rebuild.enhance.gladnet import build_gladnet
    from liptype_rebuild.enhance.losses import MSSSIML1
    from liptype_rebuild.utils.config import load_yaml

    cfg = load_yaml(config)
    low_dir = Path(cfg["data"]["low_dir"])
    norm_dir = Path(cfg["data"]["normal_dir"])
    tr = cfg.get("train", {})
    alpha = float(cfg.get("model", {}).get("alpha", 0.816))
    seed = int(tr.get("seed", 55))
    train_count = int(cfg.get("data", {}).get("train_count", 0) or 0)
    val_count = int(cfg.get("data", {}).get("val_count", 0) or 0)
    image_size = cfg.get("data", {}).get("image_size", 256)
    if isinstance(image_size, (list, tuple)):
        if len(image_size) != 2:
            raise typer.BadParameter("data.image_size must be an int or a 2-item list [H, W].")
        resize_h, resize_w = int(image_size[0]), int(image_size[1])
    else:
        resize_h = resize_w = int(image_size)

    run_dir.mkdir(parents=True, exist_ok=True)

    def _list_pngs(d: Path):
        return sorted([p for p in d.glob("*.png")])

    lows = _list_pngs(low_dir)
    norms = _list_pngs(norm_dir)

    # Pair by filename to avoid silent mismatches.
    low_map = {p.name: p for p in lows}
    norm_map = {p.name: p for p in norms}
    common = sorted(set(low_map.keys()) & set(norm_map.keys()))
    missing_low = sorted(set(norm_map.keys()) - set(low_map.keys()))
    missing_norm = sorted(set(low_map.keys()) - set(norm_map.keys()))
    if missing_low or missing_norm:
        raise typer.BadParameter(
            "low_dir and normal_dir must contain matching *.png filenames. "
            f"Missing in Low: {len(missing_low)}; missing in Normal: {len(missing_norm)}."
        )

    pairs = [(low_map[name], norm_map[name]) for name in common]
    rnd = random.Random(seed)
    rnd.shuffle(pairs)

    if train_count > 0 or val_count > 0:
        if train_count <= 0 or val_count <= 0:
            raise typer.BadParameter("If you set train_count/val_count, both must be > 0.")
        if train_count + val_count > len(pairs):
            raise typer.BadParameter(
                f"train_count+val_count={train_count+val_count} exceeds available pairs={len(pairs)}."
            )
        train_pairs = pairs[:train_count]
        val_pairs = pairs[train_count : train_count + val_count]
    else:
        # Backward compatible default: no explicit val split.
        train_pairs = pairs
        val_pairs = []

    def _read_pair(low_path: tf.Tensor, norm_path: tf.Tensor):
        low = tf.io.decode_png(tf.io.read_file(low_path), channels=3)
        norm = tf.io.decode_png(tf.io.read_file(norm_path), channels=3)
        low = tf.image.convert_image_dtype(low, tf.float32)
        norm = tf.image.convert_image_dtype(norm, tf.float32)
        low = tf.image.resize(low, (resize_h, resize_w), method="bilinear", antialias=True)
        norm = tf.image.resize(norm, (resize_h, resize_w), method="bilinear", antialias=True)
        low = tf.clip_by_value(low, 0.0, 1.0)
        norm = tf.clip_by_value(norm, 0.0, 1.0)
        # Important: give tensors a static shape. Without this, TF 2.10 + MS-SSIM can produce NaNs
        # in `model.fit(...)` even when eager single-batch runs look fine.
        low = tf.ensure_shape(low, (resize_h, resize_w, 3))
        norm = tf.ensure_shape(norm, (resize_h, resize_w, 3))
        return low, norm

    def _make_ds(ps: list[tuple[Path, Path]], training: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((list(map(str, [a for a, _b in ps])), list(map(str, [b for _a, b in ps]))))
        if training:
            ds = ds.shuffle(1024, seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(_read_pair, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(int(tr.get("batch_size", 16))).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = _make_ds(train_pairs, training=True)
    val_ds = _make_ds(val_pairs, training=False) if val_pairs else None

    model = build_gladnet()
    opt = tf.keras.optimizers.Adam(learning_rate=float(tr.get("learning_rate", 1e-3)))
    model.compile(optimizer=opt, loss=MSSSIML1(alpha=alpha))

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(run_dir / "weights.{epoch:03d}.weights.h5"),
        save_weights_only=True,
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(tr.get("epochs", 1)),
        callbacks=[
            tf.keras.callbacks.TerminateOnNaN(),
            ckpt,
            tf.keras.callbacks.TensorBoard(str(run_dir / "tb")),
        ],
    )


@app.command("lm")
def train_lm(
    corpus_txt: Path = typer.Option(..., exists=True, dir_okay=False),
    output_json: Path = typer.Option(..., dir_okay=False),
    min_count: int = typer.Option(1, min=1),
):
    """Train a bidirectional trigram language model and save it as JSON."""
    from liptype_rebuild.postprocess.ngram_lm import BiTrigramLM

    lines = corpus_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
    lm = BiTrigramLM.train(lines, min_count=min_count)
    lm.save(output_json)
    typer.echo(f"Saved LM to {output_json}")

