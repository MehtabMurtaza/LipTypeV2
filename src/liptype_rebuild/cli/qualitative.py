from __future__ import annotations

from pathlib import Path
import json

import typer

app = typer.Typer(help="Qualitative evaluation utilities (dump refs vs hyps).")


@app.command("liptype")
def qualitative_liptype(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    weights: Path = typer.Option(..., exists=True, dir_okay=False),
    split: str = typer.Option("test", help="Which split to run: train|val|test."),
    out_jsonl: Path = typer.Option(..., dir_okay=False, help="Write per-sample results as JSONL."),
    max_samples: int = typer.Option(200, min=1),
    print_n: int = typer.Option(20, min=0, help="Print first N samples to stdout."),
    beam_width: int = typer.Option(200, min=1),
    batch_size: int = typer.Option(1, min=1),
    lm_path: Path | None = typer.Option(None, exists=True, dir_okay=False, help="Optional KenLM binary path."),
    lm_weight: float | None = typer.Option(None, help="LM interpolation weight for N-best rescoring."),
    length_bonus: float | None = typer.Option(None, help="Length bonus per non-space character."),
    top_paths: int | None = typer.Option(None, min=1, help="Top-N CTC paths for optional LM rescoring."),
):
    """Dump hypothesis vs reference for TFRecords, with per-sample WER."""
    import tensorflow as tf

    from liptype_rebuild.datasets.labels import Charset
    from liptype_rebuild.datasets.qualitative_pipeline import QualPipelineConfig, make_qual_dataset
    from liptype_rebuild.datasets.tfrecords import ExampleSpec
    from liptype_rebuild.model.ctc_decode import DecodeConfig, ctc_beam_decode, ctc_beam_decode_nbest, nbest_sparse_to_texts, sparse_to_texts
    from liptype_rebuild.model.kenlm_rescore import KenLMConfig, load_kenlm_model, rescore_nbest
    from liptype_rebuild.model.liptype import LipTypeConfig, build_models
    from liptype_rebuild.utils.config import load_yaml
    from liptype_rebuild.utils.metrics import wer

    cfg = load_yaml(config)
    ds_cfg = cfg["dataset"]
    dec_cfg = cfg.get("decode", {})

    spec = ExampleSpec(
        max_frames=int(ds_cfg["max_frames"]),
        height=int(ds_cfg["img_h"]),
        width=int(ds_cfg["img_w"]),
        channels=int(ds_cfg["img_c"]),
        max_text_len=int(ds_cfg["max_text_len"]),
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

    split_l = split.lower().strip()
    if split_l == "train":
        tfrec = ds_cfg["tfrecords_train"]
    elif split_l == "val":
        tfrec = ds_cfg["tfrecords_val"]
    else:
        tfrec = ds_cfg["tfrecords_test"]

    ds = make_qual_dataset(tfrec, spec, QualPipelineConfig(batch_size=batch_size))
    lm_path_cfg = lm_path if lm_path is not None else (Path(dec_cfg["lm_path"]) if dec_cfg.get("lm_path") else None)
    use_lm = lm_path_cfg is not None
    top_paths_cfg = int(top_paths if top_paths is not None else dec_cfg.get("top_paths", 10 if use_lm else 1))
    decode_cfg = DecodeConfig(beam_width=int(beam_width), top_paths=top_paths_cfg)
    cs = Charset()
    lm_model = None
    lm_cfg = None
    if use_lm:
        lm_cfg = KenLMConfig(
            lm_path=lm_path_cfg,
            lm_weight=float(lm_weight if lm_weight is not None else dec_cfg.get("lm_weight", 0.3)),
            length_bonus=float(length_bonus if length_bonus is not None else dec_cfg.get("length_bonus", 0.0)),
            top_paths=top_paths_cfg,
        )
        lm_model = load_kenlm_model(lm_cfg.lm_path)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    printed = 0
    total_wer = 0.0

    with out_jsonl.open("w", encoding="utf-8") as f:
        for batch in ds:
            if n >= max_samples:
                break

            probs = infer(batch["frames"], training=False)
            batch_n = int(tf.shape(probs)[0].numpy())
            inp_len = tf.squeeze(batch["input_len"], axis=1)
            if use_lm:
                decoded_paths, ctc_scores = ctc_beam_decode_nbest(probs, inp_len, decode_cfg)
                nbest = nbest_sparse_to_texts(decoded_paths, cs, batch_size=batch_n)
                hyps = []
                for cand, scores in zip(nbest, ctc_scores):
                    best, _best_idx, _combined = rescore_nbest(cand, scores.tolist(), lm_cfg, model=lm_model)
                    hyps.append(best)
            else:
                sp = ctc_beam_decode(probs, inp_len, decode_cfg)
                hyps = sparse_to_texts(sp, cs, batch_size=batch_n)

            labels = batch["labels"].numpy()
            label_lens = tf.squeeze(batch["label_len"], axis=1).numpy().tolist()
            utt_ids = [u.decode("utf-8", errors="ignore") for u in batch["utterance_id"].numpy().tolist()]
            spk_ids = [s.decode("utf-8", errors="ignore") for s in batch["speaker_id"].numpy().tolist()]
            inp_lens = inp_len.numpy().tolist()

            for i in range(len(hyps)):
                if n >= max_samples:
                    break
                ref = cs.labels_to_text([int(z) for z in labels[i][: label_lens[i]] if int(z) >= 0])
                hyp = hyps[i]
                w = float(wer(hyp, ref))
                rec = {
                    "idx": n,
                    "speaker_id": spk_ids[i],
                    "utterance_id": utt_ids[i],
                    "input_len": int(inp_lens[i]),
                    "ref": ref,
                    "hyp": hyp,
                    "wer": w,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if printed < print_n:
                    typer.echo(f"[{n}] {spk_ids[i]} {utt_ids[i]} len={int(inp_lens[i])} wer={w:.3f}")
                    typer.echo(f"  REF: {ref}")
                    typer.echo(f"  HYP: {hyp}")
                    printed += 1

                total_wer += w
                n += 1

    avg = total_wer / n if n else 0.0
    typer.echo(f"Wrote {n} samples to {out_jsonl}")
    typer.echo(f"avg_wer={avg:.4f}")

