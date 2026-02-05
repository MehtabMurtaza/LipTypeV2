from __future__ import annotations


def levenshtein(a: list[str] | str, b: list[str] | str) -> int:
    """Pure-Python Levenshtein distance.

    Supports either strings or token lists.
    """
    if a == b:
        return 0
    # treat strings as sequence of characters
    a_seq = list(a) if isinstance(a, str) else a
    b_seq = list(b) if isinstance(b, str) else b
    if not a_seq:
        return len(b_seq)
    if not b_seq:
        return len(a_seq)

    # DP with O(min(n,m)) space
    if len(a_seq) < len(b_seq):
        a_seq, b_seq = b_seq, a_seq
    prev = list(range(len(b_seq) + 1))
    for i, ca in enumerate(a_seq, start=1):
        cur = [i]
        for j, cb in enumerate(b_seq, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]

