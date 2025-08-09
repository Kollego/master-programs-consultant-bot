from __future__ import annotations

import io
import logging
from typing import Optional

import requests
from pdfminer.high_level import extract_text as pdf_extract_text

logger = logging.getLogger(__name__)


def extract_text_from_pdf(url_or_path: str, timeout_sec: int = 30) -> str:
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        resp = requests.get(url_or_path, timeout=timeout_sec)
        resp.raise_for_status()
        bio = io.BytesIO(resp.content)
        text = pdf_extract_text(bio) or ""
        return text
    else:
        return pdf_extract_text(url_or_path) or "" 