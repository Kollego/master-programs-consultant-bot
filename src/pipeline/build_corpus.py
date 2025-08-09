from __future__ import annotations

import argparse
import logging
from typing import Dict, List

from src.utils.logging_config import setup_logging
from src.utils.storage import read_json, write_jsonl
from src.utils.text import split_to_chunks

logger = logging.getLogger(__name__)


def build_documents(programs: List[Dict]) -> List[Dict]:
    docs: List[Dict] = []
    for prog in programs:
        base_meta = {
            "program_title": prog.get("title"),
            "program_url": prog.get("url"),
            "program_code": prog.get("code"),
        }
        # Description
        desc = prog.get("description") or ""
        for idx, chunk in enumerate(split_to_chunks(desc, max_len=800)):
            docs.append({
                "id": f"{prog.get('title')}:desc:{idx}",
                "text": chunk,
                "meta": {**base_meta, "section": "description"},
            })
        # Facts
        facts: List[str] = []
        if prog.get("tuition"):
            facts.append(f"Стоимость: {prog['tuition']}")
        if prog.get("duration"):
            facts.append(f"Срок обучения: {prog['duration']}")
        if prog.get("language"):
            facts.append(f"Язык обучения: {prog['language']}")
        if prog.get("faculty"):
            facts.append(f"Факультет: {prog['faculty']}")
        if facts:
            docs.append({
                "id": f"{prog.get('title')}:facts",
                "text": "; ".join(facts),
                "meta": {**base_meta, "section": "facts"},
            })
        # Curriculum as individual entries
        for c in prog.get("curriculum", []):
            name = c.get("name") or ""
            if not name:
                continue
            txt = name
            if c.get("type"):
                txt += f" | тип: {c['type']}"
            if c.get("semester"):
                txt += f" | семестр: {c['semester']}"
            docs.append({
                "id": f"{prog.get('title')}:course:{name}",
                "text": txt,
                "meta": {**base_meta, "section": "course"},
            })
        # Contacts
        contacts = prog.get("contacts") or {}
        if contacts:
            ctext = "; ".join(f"{k}: {v}" for k, v in contacts.items())
            docs.append({
                "id": f"{prog.get('title')}:contacts",
                "text": ctext,
                "meta": {**base_meta, "section": "contacts"},
            })
    return docs


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Build RAG corpus from parsed programs")
    parser.add_argument("--in", dest="inp", required=True, help="Input programs.json")
    parser.add_argument("--out", dest="out", required=True, help="Output corpus.jsonl")
    args = parser.parse_args()

    programs = read_json(args.inp)
    docs = build_documents(programs)
    write_jsonl(docs, args.out)
    logger.info("Wrote %d documents to %s", len(docs), args.out)


if __name__ == "__main__":
    main() 