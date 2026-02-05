from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="Train models (LipType, enhancement, postprocess).")


@app.command("liptype")
def train_liptype(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    run_dir: Path = typer.Option(..., file_okay=False, dir_okay=True),
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
        ValWerCallback(),
    ]

    training_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(tr_cfg.get("epochs", 2)),
        callbacks=callbacks,
    )


@app.command("gladnet")
def train_gladnet(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    run_dir: Path = typer.Option(..., file_okay=False, dir_okay=True),
):
    """Train GLADNet-like enhancer from paired low/normal image folders."""
    import os
    import tensorflow as tf

    from liptype_rebuild.enhance.gladnet import build_gladnet
    from liptype_rebuild.enhance.losses import MSSSIML1
    from liptype_rebuild.utils.config import load_yaml

    cfg = load_yaml(config)
    low_dir = Path(cfg["data"]["low_dir"])
    norm_dir = Path(cfg["data"]["normal_dir"])
    tr = cfg.get("train", {})
    alpha = float(cfg.get("model", {}).get("alpha", 0.816))

    run_dir.mkdir(parents=True, exist_ok=True)

    def _list_pngs(d: Path):
        return sorted([p for p in d.glob("*.png")])

    lows = _list_pngs(low_dir)
    norms = _list_pngs(norm_dir)
    if len(lows) != len(norms):
        raise typer.BadParameter("low_dir and normal_dir must contain same number of *.png files.")

    def _read_pair(low_path: tf.Tensor, norm_path: tf.Tensor):
        low = tf.io.decode_png(tf.io.read_file(low_path), channels=3)
        norm = tf.io.decode_png(tf.io.read_file(norm_path), channels=3)
        low = tf.image.convert_image_dtype(low, tf.float32)
        norm = tf.image.convert_image_dtype(norm, tf.float32)
        return low, norm

    ds = tf.data.Dataset.from_tensor_slices((list(map(str, lows)), list(map(str, norms))))
    ds = ds.shuffle(1024, reshuffle_each_iteration=True)
    ds = ds.map(_read_pair, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(int(tr.get("batch_size", 16))).prefetch(tf.data.AUTOTUNE)

    model = build_gladnet()
    opt = tf.keras.optimizers.Adam(learning_rate=float(tr.get("learning_rate", 1e-3)))
    model.compile(optimizer=opt, loss=MSSSIML1(alpha=alpha))

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(run_dir / "weights.{epoch:03d}.weights.h5"),
        save_weights_only=True,
    )
    model.fit(ds, epochs=int(tr.get("epochs", 1)), callbacks=[ckpt, tf.keras.callbacks.TensorBoard(str(run_dir / "tb"))])


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

