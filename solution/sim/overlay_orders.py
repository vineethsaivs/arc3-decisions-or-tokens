"""Overlay orders an overlay test checks: its own overlay paired with each other one in both
orders, plus every overlay in name order and reversed. Full permutations grow factorially (seven
overlays are 5,040 bundle copies, about 17 GB) and filled the disk on 2026-10-04."""
from __future__ import annotations


def orders(names, own: str) -> list[tuple[str, ...]]:
    names = sorted(names)
    assert own in names, (own, names)
    others = [n for n in names if n != own]
    out = [(own,)] + [pair for o in others for pair in ((own, o), (o, own))]
    if len(names) > 2:
        out += [tuple(names), tuple(reversed(names))]
    return out


if __name__ == "__main__":
    got = orders(["a", "b", "c"], "b")
    assert got == [("b",), ("b", "a"), ("a", "b"), ("b", "c"), ("c", "b"), ("a", "b", "c"), ("c", "b", "a")], got
    assert len(orders([f"o{i}" for i in range(7)], "o3")) == 1 + 12 + 2
    print("PASS")
