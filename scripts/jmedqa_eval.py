"""JMedQA（医師国家試験・画像不要のテキスト選択問題）で学習前後の正答率を比較する。

例:
  python scripts/jmedqa_eval.py --prepare            # データ取得・画像問題除外
  python scripts/jmedqa_eval.py --limit 300          # 学習前・学習後で評価
"""
import argparse
import json

import _bootstrap  # noqa: F401
import pandas as pd
from medrag.config import load_config, resolve
from medrag.data_prep import read_jsonl
from medrag.evaluation import prepare_jmedqa, run_jmedqa
from medrag.models import ComparisonModel, resolve_adapter
from medrag.versioning import now_id, write_json

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)
    path = cfg["evaluation"]["jmedqa_file"]
    if args.prepare:
        print(json.dumps(prepare_jmedqa(path), ensure_ascii=False, indent=2))
        raise SystemExit
    items = read_jsonl(resolve(path))[: args.limit]
    adapter = resolve_adapter(cfg["model"]["adapter"], cfg["model"]["adapters_dir"])
    model = ComparisonModel(cfg["model"]["base_model"], adapter, cfg["model"]["device"], cfg["model"]["dtype"])
    frames = [run_jmedqa(model, cfg, items, use_lora=False,
                         progress=lambda p, m: print(f"\rbase {p:6.1%}", end=""))]
    if adapter:
        frames.append(run_jmedqa(model, cfg, items, use_lora=True,
                                 progress=lambda p, m: print(f"\rlora {p:6.1%}", end="")))
    df = pd.concat(frames)
    out = resolve(cfg["evaluation"]["runs_dir"]) / now_id("jmedqa-")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results.csv", index=False, encoding="utf-8-sig")
    summary = df.groupby("model")["correct"].agg(["count", "mean"]).rename(columns={"mean": "accuracy"})
    write_json(out / "manifest.json", {"versions": model.version_info(), "n_items": len(items),
                                       "summary": summary.reset_index().to_dict("records")})
    print("\n", summary.to_string(), f"\n保存先: {out}")
