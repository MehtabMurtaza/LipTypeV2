from __future__ import annotations

import re
import string
from collections import Counter


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


def _untokenize(words: list[str]) -> str:
    text = " ".join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace(". . .", "...")
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r" ([.,:;?!%]+)$", r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace("can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()


class NorvigSpell:
    """Norvig-style spell corrector.

    Dictionary is built from a text corpus (one big string).
    """

    def __init__(self, corpus_text: str):
        self.dictionary = Counter(list(string.punctuation) + self._words(corpus_text))

    @classmethod
    def from_file(cls, path: str) -> "NorvigSpell":
        return cls(open(path, "r", encoding="utf-8").read())

    def _words(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def P(self, word: str, N: int | None = None) -> float:
        if N is None:
            N = sum(self.dictionary.values())
        return self.dictionary[word] / N if N else 0.0

    def correction(self, word: str) -> str:
        return max(self.candidates(word), key=self.P)

    def candidates(self, word: str):
        return (
            self.known([word])
            or self.known(self.edits1(word))
            or self.known(self.edits2(word))
            or [word]
        )

    def known(self, words):
        return set(w for w in words if w in self.dictionary)

    def edits1(self, word: str):
        letters = "abcdefghijklmnopqrstuvwxyz"
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word: str):
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def sentence(self, sentence: str) -> str:
        tokens = _tokenize(sentence)
        corrected = [self.correction(tok) if tok.isalpha() else tok for tok in tokens]
        return _untokenize(corrected)

