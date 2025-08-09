from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup

from src.scraping.fetch import fetch_html
from src.utils.logging_config import setup_logging
from src.utils.text import normalize_whitespace
from src.utils.storage import write_json
from src.utils.pdf import extract_text_from_pdf

logger = logging.getLogger(__name__)


@dataclass
class Course:
    name: str
    semester: Optional[str] = None
    type: Optional[str] = None  # core | elective | module | other


@dataclass
class Program:
    url: str
    title: str
    code: Optional[str]
    degree: Optional[str]
    language: Optional[str]
    duration: Optional[str]
    faculty: Optional[str]
    tuition: Optional[str]
    description: str
    curriculum: List[Course]
    contacts: Dict[str, str]


def _extract_text(el) -> str:
    if el is None:
        return ""
    text = el.get_text(" ", strip=True)
    return normalize_whitespace(text)


def _first_match(patterns: List[re.Pattern[str]], text: str) -> Optional[str]:
    for pat in patterns:
        m = pat.search(text)
        if m:
            # pick the last group if any capture groups exist, else whole match
            if m.lastindex:
                return normalize_whitespace(m.group(m.lastindex))
            return normalize_whitespace(m.group(0))
    return None


def parse_program(url: str) -> Program:
    html = fetch_html(url)
    soup = BeautifulSoup(html, "lxml")

    # Prefer robust data from Next.js __NEXT_DATA__ when present
    next_data = soup.find("script", id="__NEXT_DATA__")
    api = None
    if next_data:
        try:
            import json
            nd = json.loads(next_data.text)
            api = (nd.get("props", {}) or {}).get("pageProps", {}).get("apiProgram")
        except Exception:
            api = None

    # Title
    title_tag = soup.find(["h1", "h2"])
    title = _extract_text(title_tag) or (api.get("title") if api else None) or "Программа магистратуры"

    # Heuristics for fields
    code = None
    degree = None
    language = None
    duration = None
    faculty = None
    tuition = None

    page_text = _extract_text(soup)

    if api:
        code = api.get("direction_code") or code
        degree = api.get("degree") or degree
        language = api.get("language") or language
        # duration is not explicit; keep heuristic below
        faculties = api.get("faculties") or []
        if faculties and isinstance(faculties, list):
            faculty = faculties[0].get("name") or faculty
        # Common name guesses for tuition fields
        for key in [
            "tuition_fee", "tuition", "price", "cost", "tuitionRUR", "tuition_rub", "tuitionRub",
        ]:
            val = api.get(key)
            if val:
                tuition = str(val)
                break

    # Description: take the first prominent text block under title
    desc = ""
    if title_tag:
        # find next sizable paragraph or section
        next_section = title_tag.find_next(["p", "div", "section"])
        if next_section:
            desc = _extract_text(next_section)[:2000]
    if not desc:
        # fallback to meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        desc = meta_desc["content"][:2000] if meta_desc and meta_desc.get("content") else ""

    # Curriculum extraction
    curriculum: List[Course] = []

    def add_course(name: str, semester: Optional[str] = None, ctype: Optional[str] = None):
        clean = normalize_whitespace(name)
        if not clean:
            return
        curriculum.append(Course(name=clean, semester=semester, type=ctype))

    # 1) Try to get structured disciplines from apiProgram if present
    if api and isinstance(api.get("disciplines"), list) and api["disciplines"]:
        for item in api["disciplines"]:
            disc = (item or {}).get("discipline", {})
            nm = disc.get("name") or disc.get("title") or ""
            sem = (item or {}).get("semester")
            ctype = None
            add_course(nm, semester=str(sem) if sem else None, ctype=ctype)

    # 2) If empty, extract from academic_plan PDF (plain text fallback)
    if not curriculum and api and api.get("academic_plan"):
        try:
            pdf_url = api["academic_plan"]
            pdf_text = extract_text_from_pdf(pdf_url)
            # Heuristic: lines that look like course names (exclude headers, page numbers)
            for line in pdf_text.splitlines():
                t = line.strip()
                if len(t) < 4:
                    continue
                if any(s in t.lower() for s in ["страница", "page", "итмо", "университет", "семестр:"]):
                    continue
                # likely a course if starts with letter and contains lowercase Cyrillic/Latin
                if (t[0].isalpha()) and any(ch.islower() for ch in t):
                    add_course(t)
        except Exception as e:
            logger.warning("Failed to parse academic_plan PDF: %s", e)

    # 3) As last resort, use HTML heuristics
    if not curriculum:
        headings = soup.find_all(["h2", "h3", "h4"], string=re.compile(r"(Учебн|План|Дисциплин|Модул|Выбор|Курс)", re.IGNORECASE))
        seen = set()
        for h in headings:
            sec = h.find_parent(["section", "div"]) or h
            for li in sec.find_all("li"):
                txt = _extract_text(li)
                if txt and (txt, None) not in seen and len(txt) > 3:
                    seen.add((txt, None))
                    ctype = "elective" if re.search(r"выбор", txt, re.IGNORECASE) else None
                    add_course(txt, ctype=ctype)
            for row in sec.find_all("tr"):
                cells = [_extract_text(td) for td in row.find_all(["td", "th"])]
                if len(cells) >= 1:
                    name = cells[0]
                    if name and (name, None) not in seen and len(name) > 3:
                        seen.add((name, None))
                        add_course(name)

    # Contacts: try to find phone/email links
    contacts: Dict[str, str] = {}
    email_link = soup.find("a", href=re.compile(r"mailto:", re.IGNORECASE))
    phone_link = soup.find("a", href=re.compile(r"tel:", re.IGNORECASE))
    if email_link:
        contacts["email"] = _extract_text(email_link)
    if phone_link:
        contacts["phone"] = _extract_text(phone_link)

    # Heuristics for tuition, duration, language, if missing
    fulltext = page_text

    if not tuition:
        tuition_patterns = [
            re.compile(r"(?:стоимост[ьи]|цена)[^\n\r\.:]{0,60}?([\d\s]{3,}(?:[.,]\d{3})?\s*(?:₽|руб\.?|рублей|RUB))", re.IGNORECASE),
            re.compile(r"([\d\s]{3,}(?:[.,]\d{3})?\s*(?:₽|руб\.?|рублей|RUB))\s*(?:в год|за год|/год)", re.IGNORECASE),
        ]
        tuition = _first_match(tuition_patterns, fulltext)

    if not duration:
        duration_patterns = [
            re.compile(r"(?:срок\s*обучения|продолжительность)[^\n\r\.:]{0,40}?(\d+(?:[.,]\d+)?\s*г(?:ода|одов|\.)?)", re.IGNORECASE),
            re.compile(r"(\d+(?:[.,]\d+)?\s*г(?:ода|одов|\.)?)\s*(?:обучения)?", re.IGNORECASE),
        ]
        duration = _first_match(duration_patterns, fulltext)

    if not language:
        lang_patterns = [
            re.compile(r"(?:язык\s*обучения|language)[^\n\r\.:]{0,40}?(русский|английский|русский и английский|bilingual|english|russian)", re.IGNORECASE),
        ]
        language = _first_match(lang_patterns, fulltext)

    if not faculty:
        fac_patterns = [
            re.compile(r"(?:факультет|меганаправление|школа)[^\n\r\.:]{0,60}?([A-Za-zА-Яа-я \-«»\"]{6,60})", re.IGNORECASE),
        ]
        fac = _first_match(fac_patterns, fulltext)
        faculty = fac or faculty

    program = Program(
        url=url,
        title=title,
        code=code,
        degree=degree,
        language=language,
        duration=duration,
        faculty=faculty,
        tuition=tuition,
        description=desc,
        curriculum=curriculum,
        contacts=contacts,
    )
    logger.info("Parsed program: %s | %d courses", program.title, len(curriculum))
    return program


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Parse ITMO program pages")
    parser.add_argument("--urls", nargs="+", required=True, help="Program URLs")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    programs: List[Dict[str, Any]] = []
    for url in args.urls:
        try:
            p = parse_program(url)
            item = asdict(p)
            # Convert dataclasses in curriculum
            item["curriculum"] = [asdict(c) for c in p.curriculum]
            programs.append(item)
        except Exception as e:
            logger.exception("Failed to parse %s: %s", url, e)

    write_json(programs, args.out)
    logger.info("Saved %d programs to %s", len(programs), args.out)


if __name__ == "__main__":
    main() 