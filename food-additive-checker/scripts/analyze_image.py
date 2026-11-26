"""コマンドラインで1枚のラベル画像を判定する（動作確認用）。

例: python scripts/analyze_image.py samples/test_label.png
"""
import argparse
import json

import _bootstrap  # noqa: F401
from PIL import Image

from foodrag.config import load_config
from foodrag.models import SarashinaModel, resolve_adapter
from foodrag.ocr import read_label
from foodrag.pipeline import analyze_text
from foodrag.rag import Retriever, build_index, docs_changed_since_index, load_catalog

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--config")
    ap.add_argument("--no-model", action="store_true", help="Sarashina を使わず参考文書から説明を作る")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if docs_changed_since_index(cfg):
        build_index(cfg)
    retriever = Retriever(cfg)
    model = None if a.no_model else SarashinaModel(
        cfg["model"]["base_model"], resolve_adapter(cfg["model"]["adapter"], cfg["model"]["adapters_dir"]),
        cfg["model"]["device"], cfg["model"]["dtype"])
    text, engine = read_label(Image.open(a.image), cfg["ocr"])
    res = analyze_text(text, engine, model, retriever, load_catalog(cfg), cfg)
    d = res.to_dict()
    for k in ("references",):
        d[k] = [{"n": r["n"], "title": r["title"]} for r in d[k]]
    print(json.dumps(d, ensure_ascii=False, indent=2))
