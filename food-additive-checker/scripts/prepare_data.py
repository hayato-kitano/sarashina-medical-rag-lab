"""LoRA学習データの作成（data/processed/<data_version>/train.jsonl, validation.jsonl）。

例: python scripts/prepare_data.py --n 2000
"""
import argparse
import json

import _bootstrap  # noqa: F401
from foodrag.config import load_config
from foodrag.data_prep import prepare

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--n", type=int)
    a = ap.parse_args()
    print(json.dumps(prepare(load_config(a.config), a.n), ensure_ascii=False, indent=2))
