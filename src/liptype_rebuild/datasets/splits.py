from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import yaml


@dataclass(frozen=True)
class SplitConfig:
    mode: Literal["speaker_holdout", "overlapped_utterances"]

    # Speaker-level splits (speaker_holdout)
    train_speakers: set[str]
    val_speakers: set[str]

    # Optional explicit test speakers (either mode)
    test_speakers: set[str]

    # Speakers to ignore completely (e.g. missing/corrupt speaker folders)
    exclude_speakers: set[str]

    # Overlapped split params (overlapped_utterances)
    val_utterances_per_speaker: int
    seed: int


def load_split_config(path: Path) -> SplitConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    mode = str(data.get("mode", "")).strip() or None

    train = set(map(str, data.get("train_speakers", [])))
    val = set(map(str, data.get("val_speakers", [])))
    test = set(map(str, data.get("test_speakers", [])))
    exclude = set(map(str, data.get("exclude_speakers", [])))

    if mode is None:
        # Backward compatible legacy behavior: treat as speaker-holdout split.
        return SplitConfig(
            mode="speaker_holdout",
            train_speakers=train,
            val_speakers=val,
            test_speakers=test,
            exclude_speakers=exclude,
            val_utterances_per_speaker=0,
            seed=int(data.get("seed", 55)),
        )

    if mode not in {"speaker_holdout", "overlapped_utterances"}:
        raise ValueError(f"Unknown split mode: {mode!r}")

    if mode == "overlapped_utterances":
        vps = int(data.get("val_utterances_per_speaker", 255))
        if vps <= 0:
            raise ValueError("val_utterances_per_speaker must be > 0 for overlapped_utterances mode.")
        sd = int(data.get("seed", 55))
        return SplitConfig(
            mode="overlapped_utterances",
            train_speakers=train,
            val_speakers=val,
            test_speakers=test,
            exclude_speakers=exclude,
            val_utterances_per_speaker=vps,
            seed=sd,
        )

    # speaker_holdout
    return SplitConfig(
        mode="speaker_holdout",
        train_speakers=train,
        val_speakers=val,
        test_speakers=test,
        exclude_speakers=exclude,
        val_utterances_per_speaker=0,
        seed=int(data.get("seed", 55)),
    )


def split_from_all_speakers(all_speakers: Iterable[str], val_speakers: Iterable[str]) -> SplitConfig:
    all_speakers = set(all_speakers)
    val = set(val_speakers)
    train = all_speakers - val
    return SplitConfig(train_speakers=train, val_speakers=val)


def make_overlapped_val_set(
    utterances: Iterable[tuple[str, str]],
    *,
    test_speakers: set[str],
    exclude_speakers: set[str],
    val_utterances_per_speaker: int,
    seed: int,
) -> set[tuple[str, str]]:
    """Create a deterministic per-speaker validation set (overlapped speakers).

    Inputs:
      utterances: iterable of (speaker_id, utterance_id)

    Returns:
      set of (speaker_id, utterance_id) assigned to validation.
      Speakers in test_speakers/exclude_speakers are never included.
    """
    import random

    by_spk: dict[str, list[str]] = {}
    for spk, utt in utterances:
        if spk in test_speakers or spk in exclude_speakers:
            continue
        by_spk.setdefault(spk, []).append(str(utt))

    val_set: set[tuple[str, str]] = set()
    for spk, utts in by_spk.items():
        utts = sorted(utts)
        rng = random.Random(f"{seed}:{spk}")
        k = min(int(val_utterances_per_speaker), len(utts))
        chosen = set(rng.sample(utts, k=k)) if k > 0 else set()
        for u in chosen:
            val_set.add((spk, u))
    return val_set

