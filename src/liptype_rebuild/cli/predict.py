from __future__ import annotations

from pathlib import Path

import typer

import json
import numpy as np

app = typer.Typer(help="Run inference (LipType + optional repair model).")


@app.command("liptype")
def predict_liptype(
    weights: Path = typer.Option(..., exists=True, dir_okay=False),
    video: Path = typer.Option(..., exists=True),
    beam_width: int = typer.Option(50, min=1),
    output_json: Path = typer.Option(None, dir_okay=False),
    repair_lm: Path = typer.Option(None, exists=True, dir_okay=False, help="Path to saved BiTrigramLM json."),
    repair_dict: Path = typer.Option(None, exists=True, dir_okay=False, help="Dictionary word list (one per line)."),
    repair_spell_corpus: Path = typer.Option(None, exists=True, dir_okay=False, help="Optional corpus for Norvig spell."),
    dlib_predictor: Path = typer.Option(None, exists=True, dir_okay=False, help="Path to dlib 68-landmark predictor .dat file."),
):
    """Predict text from a video (or frames directory)."""
    import tensorflow as tf

    from liptype_rebuild.model.ctc_decode import DecodeConfig, ctc_beam_decode, sparse_to_texts
    from liptype_rebuild.model.liptype import LipTypeConfig, build_models
    from liptype_rebuild.preprocess.landmarks import default_landmarks_backend, Dlib68Backend
    from liptype_rebuild.preprocess.mouth_roi import MouthRoiConfig, crop_video_mouth_rois
    from liptype_rebuild.preprocess.video_io import read_video_rgb

    cfg = LipTypeConfig()
    _training_model, infer = build_models(cfg)
    infer.load_weights(str(weights))

    vf = read_video_rgb(video)
    backend = Dlib68Backend(str(dlib_predictor)) if dlib_predictor is not None else default_landmarks_backend()
    lms = [backend.detect(frame) for frame in vf.frames_rgb]
    rois_u8 = crop_video_mouth_rois(vf.frames_rgb, lms, cfg=MouthRoiConfig(width=cfg.img_w, height=cfg.img_h))

    # pad/truncate to cfg.frames_n
    t = rois_u8.shape[0]
    t_use = min(t, cfg.frames_n)
    frames = rois_u8[:t_use].astype(np.float32) / 255.0
    if t_use < cfg.frames_n:
        pad = np.zeros((cfg.frames_n - t_use, cfg.img_h, cfg.img_w, cfg.img_c), dtype=np.float32)
        frames = np.concatenate([frames, pad], axis=0)

    x = np.expand_dims(frames, axis=0)  # [1,T,H,W,C]
    probs = infer(x, training=False)
    inp_len = tf.constant([t_use], dtype=tf.int32)
    sp = ctc_beam_decode(probs, inp_len, DecodeConfig(beam_width=beam_width))
    texts = sparse_to_texts(sp)
    text = texts[0] if texts else ""

    if repair_lm is not None and repair_dict is not None:
        from liptype_rebuild.postprocess.ngram_lm import BiTrigramLM
        from liptype_rebuild.postprocess.repair import RepairModel
        from liptype_rebuild.postprocess.spell import NorvigSpell

        lm = BiTrigramLM.load(repair_lm)
        dict_words = repair_dict.read_text(encoding="utf-8").splitlines()
        spell = NorvigSpell.from_file(str(repair_spell_corpus)) if repair_spell_corpus else None
        repair = RepairModel(lm=lm, dictionary_words=dict_words, spell=spell)
        text = repair.repair_sentence(text)

    result = {"text": text, "input_len": int(t_use), "video": str(video)}
    typer.echo(text)
    if output_json is not None:
        output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

