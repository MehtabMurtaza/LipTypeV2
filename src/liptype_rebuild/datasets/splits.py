from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


@dataclass(frozen=True)
class SplitConfig:
    train_speakers: set[str]
    val_speakers: set[str]


def load_split_config(path: Path) -> SplitConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    train = set(map(str, data.get("train_speakers", [])))
    val = set(map(str, data.get("val_speakers", [])))
    return SplitConfig(train_speakers=train, val_speakers=val)


def split_from_all_speakers(all_speakers: Iterable[str], val_speakers: Iterable[str]) -> SplitConfig:
    all_speakers = set(all_speakers)
    val = set(val_speakers)
    train = all_speakers - val
    return SplitConfig(train_speakers=train, val_speakers=val)

