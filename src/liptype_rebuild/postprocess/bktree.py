from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


DistanceFn = Callable[[str, str], int]


@dataclass
class BKNode:
    term: str
    children: dict[int, "BKNode"]


class BKTree:
    def __init__(self, dist: DistanceFn):
        self.dist = dist
        self.root: BKNode | None = None

    def add(self, term: str) -> None:
        if self.root is None:
            self.root = BKNode(term=term, children={})
            return
        node = self.root
        while True:
            d = self.dist(term, node.term)
            nxt = node.children.get(d)
            if nxt is None:
                node.children[d] = BKNode(term=term, children={})
                return
            node = nxt

    def build(self, terms: Iterable[str]) -> "BKTree":
        for t in terms:
            self.add(t)
        return self

    def query(self, term: str, max_dist: int) -> list[tuple[str, int]]:
        if self.root is None:
            return []
        out: list[tuple[str, int]] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            d = self.dist(term, node.term)
            if d <= max_dist:
                out.append((node.term, d))
            lo = d - max_dist
            hi = d + max_dist
            for cd, child in node.children.items():
                if lo <= cd <= hi:
                    stack.append(child)
        out.sort(key=lambda x: x[1])
        return out

