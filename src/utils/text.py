from __future__ import annotations

import re
from typing import Iterable, List


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_to_chunks(text: str, max_len: int = 700) -> List[str]:
    # Simple sentence-ish split, then re-pack to chunks
    parts = re.split(r"(?<=[.!?…])\s+|\n+", text)
    parts = [p.strip() for p in parts if p and p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    total = 0
    for p in parts:
        if total + len(p) + 1 > max_len:
            if buf:
                chunks.append(" ".join(buf).strip())
                buf, total = [], 0
        buf.append(p)
        total += len(p) + 1
    if buf:
        chunks.append(" ".join(buf).strip())
    return chunks


def to_lines(texts: Iterable[str]) -> str:
    return "\n".join(t.strip() for t in texts if t and t.strip()) 