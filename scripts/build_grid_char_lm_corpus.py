from __future__ import annotations

from pathlib import Path
import argparse

from liptype_rebuild.datasets.align import align_to_sentence, parse_align_file
from liptype_rebuild.datasets.grid import GridLayout
from liptype_rebuild.model.kenlm_rescore import char_tokenize_for_lm, normalize_for_charset


def build_corpus(grid_root: Path, out_path: Path, dedupe: bool = False) -> dict[str, int]:
    layout = GridLayout(root=grid_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_written = 0
    seen: set[str] = set()
    with out_path.open("w", encoding="utf-8") as f:
        for _speaker_id, _video_path, align_path in layout.iter_utterances():
            n_total += 1
            sent = align_to_sentence(parse_align_file(str(align_path)))
            sent = normalize_for_charset(sent)
            if not sent:
                continue
            line = " ".join(char_tokenize_for_lm(sent))
            if dedupe:
                if line in seen:
                    continue
                seen.add(line)
            f.write(line + "\n")
            n_written += 1

    return {"utterances_total": n_total, "lines_written": n_written, "deduped": int(dedupe)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Build character-level KenLM corpus from GRID align files.")
    ap.add_argument("--grid-root", type=Path, required=True, help="GRID processed root (contains s*_processed).")
    ap.add_argument("--out", type=Path, required=True, help="Output corpus text path.")
    ap.add_argument("--dedupe", action="store_true", help="Keep unique lines only.")
    args = ap.parse_args()

    stats = build_corpus(args.grid_root, args.out, dedupe=bool(args.dedupe))
    print(
        "[grid_char_lm_corpus] "
        f"total={stats['utterances_total']} written={stats['lines_written']} "
        f"dedupe={bool(stats['deduped'])} out={args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

