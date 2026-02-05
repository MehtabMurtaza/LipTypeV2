from __future__ import annotations

from dataclasses import dataclass

from liptype_rebuild.utils.levenshtein import levenshtein


def wer(hyp: str, ref: str) -> float:
    """Word error rate using Levenshtein distance on word tokens."""
    hyp_words = hyp.strip().split()
    ref_words = ref.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return levenshtein(ref_words, hyp_words) / float(len(ref_words))


@dataclass(frozen=True)
class RunningAverage:
    total: float = 0.0
    n: int = 0

    def add(self, x: float) -> "RunningAverage":
        return RunningAverage(total=self.total + float(x), n=self.n + 1)

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0

