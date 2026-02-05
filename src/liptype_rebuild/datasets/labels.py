from __future__ import annotations

from dataclasses import dataclass


CTC_BLANK = 27
SPACE = 26


@dataclass(frozen=True)
class Charset:
    """Character set consistent with LipNet/LipType GRID setup.

    Indices:
    - a-z -> 0-25
    - space -> 26
    - CTC blank -> 27 (not part of labels)
    """

    blank: int = CTC_BLANK
    space: int = SPACE

    def text_to_labels(self, text: str) -> list[int]:
        labels: list[int] = []
        for ch in text.lower():
            if "a" <= ch <= "z":
                labels.append(ord(ch) - ord("a"))
            elif ch == " ":
                labels.append(self.space)
            # silently drop other chars
        return labels

    def labels_to_text(self, labels: list[int]) -> str:
        out: list[str] = []
        for idx in labels:
            if 0 <= idx <= 25:
                out.append(chr(idx + ord("a")))
            elif idx == self.space:
                out.append(" ")
        return "".join(out)

