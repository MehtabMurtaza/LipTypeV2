from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlignItem:
    start_s: float
    end_s: float
    token: str


def parse_align_file(path: str) -> list[AlignItem]:
    """Parse GRID-style .align file.

    Format per line: `start_ms end_ms token`
    """
    items: list[AlignItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            start_ms, end_ms, token = parts[0], parts[1], parts[2]
            try:
                start_s = int(start_ms) / 1000.0
                end_s = int(end_ms) / 1000.0
            except ValueError:
                continue
            items.append(AlignItem(start_s=start_s, end_s=end_s, token=token))
    return items


def align_to_sentence(items: list[AlignItem], drop_tokens: set[str] | None = None) -> str:
    if drop_tokens is None:
        drop_tokens = {"sp", "sil"}
    words = [it.token for it in items if it.token not in drop_tokens]
    return " ".join(words)

