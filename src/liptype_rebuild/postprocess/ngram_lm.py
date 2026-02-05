from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import json
import re
from collections import Counter, defaultdict


def _words(line: str) -> list[str]:
    # Keep only a-z and spaces
    line = line.lower()
    line = re.sub(r"[^a-z\\s]", " ", line)
    return [w for w in line.split() if w]


@dataclass(frozen=True)
class BiTrigramLM:
    """Count-based bidirectional trigram LM (simple smoothed MLE)."""

    forward: dict[str, dict[str, float]]
    backward: dict[str, dict[str, float]]
    unigram: dict[str, float]

    @staticmethod
    def train(lines: Iterable[str], min_count: int = 1) -> "BiTrigramLM":
        f_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        b_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        uni = Counter()

        for line in lines:
            toks = ["<s>"] + _words(line) + ["</s>"]
            for w in toks:
                uni[w] += 1
            # forward trigrams
            for i in range(len(toks) - 2):
                ctx = (toks[i], toks[i + 1])
                nxt = toks[i + 2]
                f_counts[ctx][nxt] += 1
            # backward trigrams (reverse conditioning)
            for i in range(2, len(toks)):
                ctx = (toks[i], toks[i - 1])  # (current, previous) predicts previous-previous
                prevprev = toks[i - 2]
                b_counts[ctx][prevprev] += 1

        # prune unigram vocab
        vocab = {w for w, c in uni.items() if c >= min_count}
        total_uni = float(sum(uni[w] for w in vocab))
        unigram = {w: (uni[w] / total_uni) for w in vocab}

        def _normalize(counts: Counter[str]) -> dict[str, float]:
            total = float(sum(counts.values()))
            return {w: c / total for w, c in counts.items() if w in vocab}

        forward = {f"{a}\t{b}": _normalize(cnt) for (a, b), cnt in f_counts.items()}
        backward = {f"{a}\t{b}": _normalize(cnt) for (a, b), cnt in b_counts.items()}
        return BiTrigramLM(forward=forward, backward=backward, unigram=unigram)

    def save(self, path: Path) -> None:
        obj = {"forward": self.forward, "backward": self.backward, "unigram": self.unigram}
        path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "BiTrigramLM":
        obj = json.loads(path.read_text(encoding="utf-8"))
        return BiTrigramLM(forward=obj["forward"], backward=obj["backward"], unigram=obj["unigram"])

    def _p_unigram(self, w: str) -> float:
        return float(self.unigram.get(w, 1e-9))

    def p_forward(self, w1: str, w2: str, w3: str) -> float:
        key = f"{w1}\t{w2}"
        dist = self.forward.get(key)
        if not dist:
            return self._p_unigram(w3)
        return float(dist.get(w3, 1e-9))

    def p_backward(self, w3: str, w2: str, w1: str) -> float:
        key = f"{w3}\t{w2}"
        dist = self.backward.get(key)
        if not dist:
            return self._p_unigram(w1)
        return float(dist.get(w1, 1e-9))

    def p_combined_word(self, prev: str, cur: str, nxt: str) -> float:
        # combined probability proxy for the middle word given neighbors
        return self.p_forward(prev, cur, nxt) * self.p_backward(nxt, cur, prev)

