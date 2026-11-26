"""RAGインデックス作成：data/rag_docs の添加物文書 → e5 Embedding → FAISS。

例: python scripts/build_index.py
"""
import argparse
import json

import _bootstrap  # noqa: F401
from foodrag.config import load_config
from foodrag.rag import build_index

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    cfg = load_config(ap.parse_args().config)
    man = build_index(cfg)
    print(json.dumps({k: man[k] for k in ("version", "n_documents", "n_chunks", "embedding_model")},
                     ensure_ascii=False, indent=2))
