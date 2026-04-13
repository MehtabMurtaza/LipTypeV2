from __future__ import annotations

from pathlib import Path

import typer


app = typer.Typer(help="Evaluation utilities.")


@app.command("liptype")
def eval_liptype(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    weights: Path = typer.Option(..., exists=True, dir_okay=False),
    num_batches: int = typer.Option(50, min=1),
    split: str = typer.Option("val", help="Which split to evaluate: train|val|test."),
    lm_path: Path | None = typer.Option(None, exists=True, dir_okay=False, help="Optional KenLM binary path."),
    lm_weight: float | None = typer.Option(None, help="LM interpolation weight for N-best rescoring."),
    length_bonus: float | None = typer.Option(None, help="Length bonus per non-space character."),
    top_paths: int | None = typer.Option(None, min=1, help="Top-N CTC paths for optional LM rescoring."),
):
    """Evaluate LipType WER on TFRecords configured in YAML."""
    import tensorflow as tf

    from liptype_rebuild.datasets.input_pipeline import PipelineConfig, make_dataset
    from liptype_rebuild.datasets.tfrecords import ExampleSpec
    from liptype_rebuild.model.ctc_decode import DecodeConfig, ctc_beam_decode, ctc_beam_decode_nbest, nbest_sparse_to_texts, sparse_to_texts
    from liptype_rebuild.model.kenlm_rescore import KenLMConfig, load_kenlm_model, rescore_nbest
    from liptype_rebuild.model.liptype import LipTypeConfig, build_models
    from liptype_rebuild.utils.config import load_yaml
    from liptype_rebuild.utils.metrics import RunningAverage, wer

    cfg = load_yaml(config)
    ds_cfg = cfg["dataset"]
    tr_cfg = cfg["train"]
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
        flip_prob=0.0,
        temporal_jitter_prob=0.0,
    )
    model_cfg = LipTypeConfig(
        img_h=spec.height,
        img_w=spec.width,
        img_c=spec.channels,
        frames_n=spec.max_frames,
        max_text_len=spec.max_text_len,
        output_size=28,
    )

    _train, infer = build_models(model_cfg)
    infer.load_weights(str(weights))

    split = split.lower().strip()
    if split == "train":
        tfrec = ds_cfg["tfrecords_train"]
    elif split == "test":
        tfrec = ds_cfg["tfrecords_test"]
    else:
        tfrec = ds_cfg["tfrecords_val"]

    val_ds = make_dataset(tfrec, spec, pipe_cfg, training=False)
    lm_path_cfg = lm_path if lm_path is not None else (Path(dec_cfg["lm_path"]) if dec_cfg.get("lm_path") else None)
    use_lm = lm_path_cfg is not None
    top_paths_cfg = int(top_paths if top_paths is not None else dec_cfg.get("top_paths", 10 if use_lm else 1))
    decode_cfg = DecodeConfig(
        beam_width=int(dec_cfg.get("beam_width", 50)),
        top_paths=top_paths_cfg,
    )

    from liptype_rebuild.datasets.labels import Charset

    cs = Charset()
    lm_model = None
    lm_cfg = None
    if use_lm:
        lm_cfg = KenLMConfig(
            lm_path=lm_path_cfg,
            lm_weight=float(lm_weight if lm_weight is not None else dec_cfg.get("lm_weight", 0.3)),
            length_bonus=float(length_bonus if length_bonus is not None else dec_cfg.get("length_bonus", 0.0)),
            top_paths=int(top_paths_cfg),
        )
        lm_model = load_kenlm_model(lm_cfg.lm_path)

    avg = RunningAverage()
    for x, _y in val_ds.take(num_batches):
        probs = infer(x["frames"], training=False)
        batch_size = int(tf.shape(probs)[0].numpy())
        inp_len = tf.squeeze(x["input_len"], axis=1)
        if use_lm:
            decoded_paths, ctc_scores = ctc_beam_decode_nbest(probs, inp_len, decode_cfg)
            nbest = nbest_sparse_to_texts(decoded_paths, cs, batch_size=batch_size)
            hyps = []
            for cand, scores in zip(nbest, ctc_scores):
                best, _best_idx, _combined = rescore_nbest(cand, scores.tolist(), lm_cfg, model=lm_model)
                hyps.append(best)
        else:
            sp = ctc_beam_decode(probs, inp_len, decode_cfg)
            hyps = sparse_to_texts(sp, cs, batch_size=batch_size)

        labels = x["labels"].numpy()
        label_lens = tf.squeeze(x["label_len"], axis=1).numpy().tolist()
        refs = []
        for lab, ll in zip(labels, label_lens):
            refs.append(cs.labels_to_text([int(z) for z in lab[:ll] if int(z) >= 0]))
        for h, r in zip(hyps, refs):
            avg = avg.add(wer(h, r))

    typer.echo(f"{split}_wer={avg.mean:.4f} over {avg.n} samples")


@app.command("repair")
def eval_repair(
    pairs_tsv: Path = typer.Option(..., exists=True, dir_okay=False, help="TSV: <ref>\\t<hyp> per line."),
    repair_lm: Path = typer.Option(..., exists=True, dir_okay=False),
    repair_dict: Path = typer.Option(..., exists=True, dir_okay=False),
    repair_spell_corpus: Path = typer.Option(None, exists=True, dir_okay=False),
    repair_dda_weights: Path = typer.Option(None, exists=True, dir_okay=False),
    tau1_values: str = typer.Option("0.7", help="Comma-separated tau1 values, e.g. 0.5,0.7,0.9"),
    tau2_values: str = typer.Option("2", help="Comma-separated tau2 values, e.g. 1,2,3"),
    max_samples: int = typer.Option(0, min=0, help="If >0, evaluate only first N pairs."),
):
    """Evaluate post-error-correction module on ref/hyp sentence pairs."""
    from liptype_rebuild.postprocess.dda import DDAConfig, apply_dda_to_text, load_dda
    from liptype_rebuild.postprocess.ngram_lm import BiTrigramLM
    from liptype_rebuild.postprocess.repair import RepairConfig, RepairModel
    from liptype_rebuild.postprocess.spell import NorvigSpell
    from liptype_rebuild.utils.metrics import wer

    taus1 = [float(x.strip()) for x in tau1_values.split(",") if x.strip()]
    taus2 = [int(x.strip()) for x in tau2_values.split(",") if x.strip()]
    if not taus1 or not taus2:
        raise typer.BadParameter("tau1_values/tau2_values must not be empty.")

    lines = pairs_tsv.read_text(encoding="utf-8", errors="ignore").splitlines()
    pairs: list[tuple[str, str]] = []
    for ln in lines:
        if "\t" not in ln:
            continue
        ref, hyp = ln.split("\t", 1)
        pairs.append((ref.strip().lower(), hyp.strip().lower()))
        if max_samples > 0 and len(pairs) >= max_samples:
            break
    if not pairs:
        raise typer.BadParameter("No valid pairs found. Expected TSV lines: <ref>\\t<hyp>.")

    lm = BiTrigramLM.load(repair_lm)
    dict_words = repair_dict.read_text(encoding="utf-8", errors="ignore").splitlines()
    spell = NorvigSpell.from_file(str(repair_spell_corpus)) if repair_spell_corpus else None
    dda = load_dda(str(repair_dda_weights), DDAConfig()) if repair_dda_weights else None

    baseline = sum(wer(h, r) for r, h in pairs) / float(len(pairs))
    typer.echo(f"baseline_wer={baseline:.4f} over {len(pairs)} samples")

    best = (10.0, None, None)  # (wer, tau1, tau2)
    for t1 in taus1:
        for t2 in taus2:
            model = RepairModel(
                lm=lm,
                dictionary_words=dict_words,
                spell=spell,
                cfg=RepairConfig(tau1=float(t1), tau2=int(t2)),
            )
            total = 0.0
            for ref, hyp in pairs:
                x = hyp
                if dda is not None:
                    x = apply_dda_to_text(dda, x, DDAConfig())
                rep = model.repair_sentence(x)
                total += wer(rep, ref)
            w = total / float(len(pairs))
            typer.echo(f"tau1={t1:.3f} tau2={t2} repaired_wer={w:.4f}")
            if w < best[0]:
                best = (w, t1, t2)

    typer.echo(f"best_repaired_wer={best[0]:.4f} at tau1={best[1]:.3f}, tau2={int(best[2])}")

