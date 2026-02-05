from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from liptype_rebuild.postprocess.bktree import BKTree
from liptype_rebuild.postprocess.ngram_lm import BiTrigramLM
from liptype_rebuild.postprocess.spell import NorvigSpell
from liptype_rebuild.utils.levenshtein import levenshtein


@dataclass(frozen=True)
class RepairConfig:
    tau1: float = 0.7
    tau2: int = 2


def _lev(a: str, b: str) -> int:
    return int(levenshtein(a, b))


class RepairModel:
    def __init__(
        self,
        lm: BiTrigramLM,
        dictionary_words: Iterable[str],
        spell: NorvigSpell | None = None,
        cfg: RepairConfig = RepairConfig(),
    ):
        self.lm = lm
        self.cfg = cfg
        self.spell = spell
        self.bk = BKTree(_lev).build({w.lower() for w in dictionary_words if w})

    def _score_sentence(self, words: list[str]) -> float:
        # product of combined word probabilities (log-space would be better; keep simple)
        if not words:
            return 0.0
        toks = ["<s>"] + [w.lower() for w in words] + ["</s>"]
        s = 1.0
        for i in range(1, len(toks) - 1):
            s *= self.lm.p_combined_word(toks[i - 1], toks[i], toks[i + 1])
        return s

    def repair_sentence(self, text: str) -> str:
        if self.spell is not None:
            text = self.spell.sentence(text)
        words = text.strip().split()
        if not words:
            return ""

        toks = ["<s>"] + [w.lower() for w in words] + ["</s>"]
        repaired = words[:]

        for i in range(1, len(toks) - 1):
            prev, cur, nxt = toks[i - 1], toks[i], toks[i + 1]
            p = self.lm.p_combined_word(prev, cur, nxt)
            if p >= self.cfg.tau1:
                continue

            # candidates within edit distance tau2
            candidates = [cur]
            candidates += [w for (w, d) in self.bk.query(cur, self.cfg.tau2)]
            # choose candidate maximizing sentence score when substituted
            best = repaired[i - 1]
            best_score = -1.0
            for cand in candidates[:50]:
                tmp = repaired[:]
                tmp[i - 1] = cand
                sc = self._score_sentence(tmp)
                if sc > best_score:
                    best_score = sc
                    best = cand
            repaired[i - 1] = best

        return " ".join(repaired)

