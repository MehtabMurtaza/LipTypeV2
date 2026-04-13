from __future__ import annotations

from dataclasses import dataclass
import random
import re

import numpy as np


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _delete_char(w: str, rng: random.Random) -> str:
    if len(w) <= 1:
        return w
    i = rng.randrange(len(w))
    return w[:i] + w[i + 1 :]


def _transpose_char(w: str, rng: random.Random) -> str:
    if len(w) <= 1:
        return w
    i = rng.randrange(len(w) - 1)
    return w[:i] + w[i + 1] + w[i] + w[i + 2 :]


def _replace_char(w: str, rng: random.Random) -> str:
    if len(w) == 0:
        return w
    i = rng.randrange(len(w))
    c = chr(ord("a") + rng.randrange(26))
    return w[:i] + c + w[i + 1 :]


def _insert_char(w: str, rng: random.Random) -> str:
    i = rng.randrange(len(w) + 1)
    c = chr(ord("a") + rng.randrange(26))
    return w[:i] + c + w[i:]


def inject_one_error_per_word(sentence: str, rng: random.Random) -> str:
    """Inject one character-level error per word using paper-style operations.

    Operation cycle per word index:
      0 -> delete, 1 -> transpose, 2 -> replace, 3 -> insert
    """
    words = [w for w in sentence.split(" ") if w]
    out: list[str] = []
    for i, w in enumerate(words):
        op = i % 4
        if op == 0:
            out.append(_delete_char(w, rng))
        elif op == 1:
            out.append(_transpose_char(w, rng))
        elif op == 2:
            out.append(_replace_char(w, rng))
        else:
            out.append(_insert_char(w, rng))
    return " ".join(out)


@dataclass(frozen=True)
class DDADatasetConfig:
    seq_len: int = 28
    max_sentences: int = 200_000
    train_sentences: int = 100_000
    seed: int = 55


def make_noisy_clean_sentence_pairs(lines: list[str], cfg: DDADatasetConfig) -> tuple[list[str], list[str]]:
    norm = [normalize_text(x) for x in lines]
    norm = [x for x in norm if x]
    if not norm:
        return [], []

    rng = random.Random(cfg.seed)
    rng.shuffle(norm)
    use = norm[: min(cfg.max_sentences, len(norm))]

    n_train = min(cfg.train_sentences, len(use))
    clean = use[:n_train]
    noisy = [inject_one_error_per_word(s, rng) for s in clean]
    return noisy, clean


def train_test_split_pairs(
    noisy: list[str], clean: list[str], test_ratio: float = 0.2
) -> tuple[list[str], list[str], list[str], list[str]]:
    n = min(len(noisy), len(clean))
    n_test = int(round(n * test_ratio))
    n_test = max(1, min(n_test, n - 1)) if n > 1 else 0
    n_train = n - n_test
    return noisy[:n_train], clean[:n_train], noisy[n_train:n], clean[n_train:n]


def _char_to_idx(ch: str) -> int:
    # 0-25 letters, 26 space, 27 newline
    if "a" <= ch <= "z":
        return ord(ch) - ord("a")
    if ch == " ":
        return 26
    return 27


def _idx_to_char(idx: int) -> str:
    if 0 <= idx <= 25:
        return chr(idx + ord("a"))
    if idx == 26:
        return " "
    return "\n"


def sentence_to_ids(s: str) -> list[int]:
    # End with newline token so decoder learns sentence termination.
    return [_char_to_idx(ch) for ch in s] + [27]


def ids_to_sentence(ids: list[int]) -> str:
    chars = [_idx_to_char(int(i)) for i in ids]
    txt = "".join(chars)
    # Trim at first newline token.
    if "\n" in txt:
        txt = txt.split("\n", 1)[0]
    return re.sub(r"\s+", " ", txt).strip()


def _chunk_ids(ids: list[int], seq_len: int) -> list[list[int]]:
    if not ids:
        return [[27] * seq_len]
    chunks: list[list[int]] = []
    for i in range(0, len(ids), seq_len):
        c = ids[i : i + seq_len]
        if len(c) < seq_len:
            c = c + [27] * (seq_len - len(c))
        chunks.append(c)
    return chunks


def pair_to_matrices(noisy_text: str, clean_text: str, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert one sentence pair into chunked one-hot matrices.

    Returns:
      x: [N, seq_len, 28], y: [N, seq_len, 28]
    """
    n_ids = sentence_to_ids(noisy_text)
    c_ids = sentence_to_ids(clean_text)
    n_chunks = _chunk_ids(n_ids, seq_len)
    c_chunks = _chunk_ids(c_ids, seq_len)

    n = max(len(n_chunks), len(c_chunks))
    while len(n_chunks) < n:
        n_chunks.append([27] * seq_len)
    while len(c_chunks) < n:
        c_chunks.append([27] * seq_len)

    x = np.zeros((n, seq_len, 28), dtype=np.float32)
    y = np.zeros((n, seq_len, 28), dtype=np.float32)
    for i in range(n):
        for t in range(seq_len):
            x[i, t, n_chunks[i][t]] = 1.0
            y[i, t, c_chunks[i][t]] = 1.0
    return x, y


def build_dda_arrays(noisy: list[str], clean: list[str], seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for n, c in zip(noisy, clean):
        x, y = pair_to_matrices(n, c, seq_len=seq_len)
        xs.append(x)
        ys.append(y)
    if not xs:
        return np.zeros((0, seq_len * 28), dtype=np.float32), np.zeros((0, seq_len * 28), dtype=np.float32)
    x_all = np.concatenate(xs, axis=0).reshape((-1, seq_len * 28))
    y_all = np.concatenate(ys, axis=0).reshape((-1, seq_len * 28))
    return x_all, y_all

