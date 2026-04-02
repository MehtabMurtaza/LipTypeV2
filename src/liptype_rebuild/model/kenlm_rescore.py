from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


@dataclass(frozen=True)
class KenLMConfig:
    lm_path: Path
    lm_weight: float = 0.3
    length_bonus: float = 0.0
    top_paths: int = 10


_LM_CACHE: dict[str, object] = {}


def normalize_for_charset(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^a-z ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def char_tokenize_for_lm(text: str) -> list[str]:
    """Character-level tokenization compatible with LipType charset.

    We keep letters as-is and encode spaces as a dedicated `<sp>` token so
    boundaries survive tokenization when passed through KenLM's whitespace
    tokenizer.
    """
    t = normalize_for_charset(text)
    out: list[str] = []
    for ch in t:
        if ch == " ":
            out.append("<sp>")
        else:
            out.append(ch)
    return out


def _to_lm_line(text: str) -> str:
    toks = char_tokenize_for_lm(text)
    return " ".join(toks)


def load_kenlm_model(lm_path: str | Path):
    p = str(Path(lm_path).resolve())
    if p in _LM_CACHE:
        return _LM_CACHE[p]
    try:
        import kenlm  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "KenLM python bindings are not available. Install package `kenlm` in this environment."
        ) from e

    model = kenlm.Model(p)  # type: ignore[attr-defined]
    _LM_CACHE[p] = model
    return model


def score_text_kenlm(model, text: str) -> float:
    line = _to_lm_line(text)
    if not line:
        return 0.0
    # kenlm.Model.score returns log10 probability. Higher is better.
    return float(model.score(line, bos=True, eos=True))


def rescore_nbest(
    candidates: list[str],
    ctc_logprobs: list[float] | np.ndarray,
    cfg: KenLMConfig,
    *,
    model=None,
) -> tuple[str, int, np.ndarray]:
    """Return (best_text, best_index, combined_scores)."""
    if len(candidates) == 0:
        return "", 0, np.zeros((0,), dtype=np.float32)

    if model is None:
        model = load_kenlm_model(cfg.lm_path)

    ctc = np.asarray(ctc_logprobs, dtype=np.float32)
    if ctc.ndim != 1:
        ctc = ctc.reshape(-1)
    if ctc.shape[0] < len(candidates):
        pad = np.full((len(candidates) - ctc.shape[0],), -1e9, dtype=np.float32)
        ctc = np.concatenate([ctc, pad], axis=0)
    elif ctc.shape[0] > len(candidates):
        ctc = ctc[: len(candidates)]

    lm_scores = np.array([score_text_kenlm(model, s) for s in candidates], dtype=np.float32)
    lengths = np.array([len(normalize_for_charset(s).replace(" ", "")) for s in candidates], dtype=np.float32)
    combined = ctc + float(cfg.lm_weight) * lm_scores + float(cfg.length_bonus) * lengths
    best_idx = int(np.argmax(combined))
    return candidates[best_idx], best_idx, combined

