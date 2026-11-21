"""評価質問の一括実行（4モード A〜D）。結果は outputs/eval_runs/<id>/results.csv。

例:
  python scripts/run_eval.py                       # 全150問 × A,B,C,D
  python scripts/run_eval.py --modes A B --limit 20
  python scripts/run_eval.py --adapter lora-20261121-xxxx
"""
import argparse

import _bootstrap  # noqa: F401
from medrag.config import load_config
from medrag.evaluation import aggregate, load_questions, load_run, run_batch
from medrag.models import ComparisonModel, resolve_adapter
from medrag.rag import Retriever, index_exists

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--modes", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--adapter")
    ap.add_argument("--name", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    adapter = resolve_adapter(args.adapter or cfg["model"]["adapter"], cfg["model"]["adapters_dir"])
    if adapter is None and any(m in ("B", "D") for m in args.modes):
        raise SystemExit("LoRAアダプターが見つかりません。先に scripts/train_lora.py を実行してください。")
    model = ComparisonModel(cfg["model"]["base_model"], adapter, cfg["model"]["device"], cfg["model"]["dtype"])
    retriever = Retriever(cfg) if index_exists(cfg) else None
    if retriever is None and any(m in ("C", "D") for m in args.modes):
        raise SystemExit("RAGインデックスがありません。先に scripts/build_index.py を実行してください。")
    qs = load_questions(cfg["evaluation"]["questions_file"])
    if args.limit:
        qs = qs.head(args.limit)
    out = run_batch(qs, args.modes, model, retriever, cfg,
                    progress=lambda p, m: print(f"\r{p:6.1%} {m}", end="", flush=True), name=args.name)
    print(f"\n保存先: {out}")
    print(aggregate(load_run(out)).iloc[:, :8].to_string())
