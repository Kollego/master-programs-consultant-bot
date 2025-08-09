from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Set

import numpy as np
from rapidfuzz import fuzz  # type: ignore
from sentence_transformers import SentenceTransformer
import faiss  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class Doc:
    id: str
    text: str
    meta: dict


class Retriever:
    def __init__(self, index_dir: str, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self.docs: List[Doc] = self._load_docs()
        self.texts = [d.text for d in self.docs]
        self._init_index()

    def _load_docs(self) -> List[Doc]:
        lines = Path(self.index_dir, "docs.jsonl").read_text(encoding="utf-8").splitlines()
        docs: List[Doc] = []
        for line in lines:
            d = ast.literal_eval(line)
            docs.append(Doc(id=d["id"], text=d["text"], meta=d.get("meta", {})))
        return docs

    def _init_index(self) -> None:
        index_path = self.index_dir / "faiss.index"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Build it with src/pipeline/index.py.")
        self.model = SentenceTransformer(self.model_name)
        self.index = faiss.read_index(str(index_path))

    def search(self, query: str, k: int = 5) -> List[Tuple[Doc, float]]:
        qv = self.model.encode([query], normalize_embeddings=True)
        sims, idxs = self.index.search(qv.astype(np.float32), k)
        return [(self.docs[i], float(sims[0, j])) for j, i in enumerate(idxs[0])]


ALLOWED_KEYWORDS = [
    "магистратура", "учебный план", "дисциплины", "курсы", "семестр", "предметы",
    "профиль", "поступление", "программа", "AI", "ИИ", "AI Product", "аспирантура",
    "стоимость", "язык обучения", "контакты", "факультет", "срок обучения", "цена",
]

PROGRAM_ALIASES = [
    "ИИ", "искусственный интеллект", "AI", "AI Product", "проектирование AI-продуктов",
]


def is_relevant(question: str, retrieved: List[Tuple[Doc, float]]) -> bool:
    q = question.lower()
    key_hit = any(kw in q for kw in [k.lower() for k in ALLOWED_KEYWORDS])
    alias_hit = fuzz.partial_ratio(" ".join(PROGRAM_ALIASES).lower(), q) > 70
    retr_hit = any(p[1] > 0.15 for p in retrieved)
    return key_hit or alias_hit or retr_hit


def _extract_program_info(docs: List[Doc]) -> Tuple[Optional[str], Optional[str]]:
    for d in docs:
        mt = d.meta or {}
        title = mt.get("program_title")
        url = mt.get("program_url")
        if title or url:
            return title, url
    return None, None


def _collect_course_names(docs: List[Doc], limit: int = 5) -> List[str]:
    names: List[str] = []
    for d in docs:
        section = (d.meta or {}).get("section", "")
        if section != "course":
            continue
        txt = (d.text or "").strip()
        if not txt or len(txt) < 2:
            continue
        if any(h in txt.lower() for h in [
            "учебный план", "наименование модулей", "блок 1", "обязательные дисциплины", "пул выборных",
        ]):
            continue
        if txt not in names:
            names.append(txt)
        if len(names) >= limit:
            break
    return names


def _looks_like_teacher_question(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in [
        "кто вед", "кто препода", "преподавател", "лектор", "преподы", "кто читает",
    ])


def _looks_like_cost_question(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in [
        "сколько стоит", "стоимост", "цена", "сколько в год", "платн", "руб", "₽",
    ])


def _find_facts(docs: List[Doc]) -> List[str]:
    facts: List[str] = []
    for d in docs:
        if (d.meta or {}).get("section") == "facts" and d.text:
            facts.append(d.text)
    return facts


def build_answer(question: str, retrieved: List[Tuple[Doc, float]]) -> str:
    if not retrieved:
        return "К сожалению, я не нашёл ответа в материалах программ. Уточните вопрос."

    docs = [d for (d, _s) in retrieved if d and d.text]
    prog_title, prog_url = _extract_program_info(docs)

    if _looks_like_cost_question(question):
        facts = _find_facts(docs)
        fact_line = next((f for f in facts if any(w in f.lower() for w in ["стоимость", "цена", "руб", "₽"])), None)
        if fact_line:
            value = fact_line.split(';')[0].replace('Стоимость:', '').strip()
            if prog_title and prog_url:
                return f"Стоимость обучения по программе «{prog_title}»: {value}. Подробнее — {prog_url}."
            if prog_title:
                return f"Стоимость обучения по программе «{prog_title}»: {value}."
            return fact_line

    if _looks_like_teacher_question(question):
        if not any("препода" in (d.text or "").lower() or "лектор" in (d.text or "").lower() for d in docs):
            base = "В открытых материалах программы нет фиксированного списка преподавателей — он может меняться по семестрам."
            if prog_title and prog_url:
                return f"{base} Актуальную информацию обычно публикуют на странице программы «{prog_title}»: {prog_url}."
            if prog_title:
                return f"{base} Попробуйте уточнить у учебного офиса программы «{prog_title}»."
            return base + " Посмотрите страницу программы или уточните вопрос."

    course_names = _collect_course_names(docs, limit=5)
    if course_names:
        listed = ", ".join(course_names)
        if prog_title and prog_url:
            return (
                f"По учебному плану программы «{prog_title}» вам могут быть полезны дисциплины: {listed}. "
                f"Полный список и описание — на странице программы: {prog_url}."
            )
        if prog_title:
            return f"В учебном плане «{prog_title}» встречаются курсы: {listed}."
        return f"Среди подходящих курсов: {listed}."

    lines: List[str] = []
    seen: Set[str] = set()
    for doc, _score in retrieved[:3]:
        snippet = (doc.text or "").strip()
        if not snippet or snippet in seen:
            continue
        seen.add(snippet)
        lines.append(snippet)

    if not lines:
        return "Не удалось сформировать ответ по материалам. Уточните вопрос."

    body = " ".join(lines[:2]) if len(lines) >= 2 else lines[0]
    if prog_title and prog_url:
        return f"По материалам программы «{prog_title}»: {body}. Подробнее — {prog_url}."
    if prog_title:
        return f"По материалам «{prog_title}»: {body}."
    return body 