from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

from rapidfuzz import fuzz  # type: ignore


@dataclass
class Elective:
    name: str
    score: float


_HEADING_KEYWORDS = [
    "учебный план",
    "наименование модулей",
    "блок",
    "обязательн",
    "пул выборных",
    "аттестац",
    "практик",
    "индивидуальная профессиональная подготовка",
]


def _is_heading(name: str) -> bool:
    nl = (name or "").strip().lower()
    if not nl:
        return True
    if any(k in nl for k in _HEADING_KEYWORDS):
        return True
    if nl.endswith("семестр") or " семестр" in nl:
        # e.g., "Пул выборных дисциплин. 1 семестр"
        return True
    # Very long section-like lines with lots of spaces but without parentheses/commas may be headings
    if len(nl) > 80 and ("(" not in nl and "," not in nl):
        return True
    return False


def _extract_course_names(program: Dict) -> List[str]:
    courses = program.get("curriculum", [])
    names: List[str] = []
    for c in courses:
        nm = (c.get("name") or "").strip()
        if not nm or _is_heading(nm):
            continue
        names.append(nm)
    return names


def recommend_electives(background: str, program: Dict, top_k: int = 5) -> List[Elective]:
    names = _extract_course_names(program)
    if not names:
        return []

    query = (background or "").lower().strip()
    scored: List[Elective] = []
    for name in names:
        s = max(
            fuzz.partial_ratio(query, name.lower()),
            fuzz.token_set_ratio(query, name.lower()),
        ) / 100.0
        scored.append(Elective(name=name, score=s))

    scored.sort(key=lambda e: e.score, reverse=True)
    return scored[:top_k]


def score_program(background: str, program: Dict, top_k: int = 5) -> Tuple[float, List[Elective]]:
    """Return an aggregate score and top-k electives for a program."""
    recs = recommend_electives(background, program, top_k=top_k)
    if not recs:
        return 0.0, []
    # Aggregate: average of top_k scores
    agg = sum(e.score for e in recs) / len(recs)
    return agg, recs 