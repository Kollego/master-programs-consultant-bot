from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.utils.logging_config import setup_logging
from src.utils.storage import read_jsonl, ensure_dir

from sentence_transformers import SentenceTransformer
import faiss  # type: ignore

logger = logging.getLogger(__name__)


class FaissIndex:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.vectors = None

    def build(self, texts: List[str]) -> None:
        vectors = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors.astype(np.float32))
        self.index = index
        self.vectors = vectors

    def search(self, queries: List[str], k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        qv = self.model.encode(queries, normalize_embeddings=True)
        sims, idxs = self.index.search(qv.astype(np.float32), k)
        return sims, idxs

    def save(self, d: str) -> None:
        dpath = Path(d)
        ensure_dir(dpath)
        faiss.write_index(self.index, str(dpath / "faiss.index"))
        (dpath / "model.txt").write_text(self.model._first_module()._get_name(), encoding="utf-8")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Build vector index for RAG")
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    docs = read_jsonl(args.inp)
    texts = [d["text"] for d in docs]

    logger.info("Building FAISS index with model %s", args.model)
    idx = FaissIndex(model_name=args.model)
    idx.build(texts)
    idx.save(args.index_dir)
    Path(args.index_dir, "docs.jsonl").write_text("\n".join([str(d) for d in docs]), encoding="utf-8")


if __name__ == "__main__":
    main() 