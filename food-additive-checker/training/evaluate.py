"""学習済みモデルの評価：添加物の検出漏れ・出典の正しさ・根拠カバー率。

  python -m training.evaluate --val training/data/val.jsonl --limit 50
"""
from __future__ import annotations

import argparse
import json

from src import generator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="training/data/val.jsonl")
    ap.add_argument("--limit", type=int, default=50)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.val, encoding="utf-8")][: a.limit]
    tot = {"recall": 0.0, "invalid": 0, "coverage": 0.0}
    for r in rows:
        msgs = r["messages"][:-1]
        # プロンプト中の参考資料から [D番号]→名称 を復元
        user = msgs[1]["content"]
        refs = {}
        for line in user.splitlines():
            if line.startswith("[D"):
                k, rest = line[1:].split("]", 1)
                refs[k] = {"name": rest.split("｜")[0].strip(), "body": rest}
        out = generator.generate(msgs)
        chk = generator.verify_citations(out, refs)
        names = [refs[k]["name"] for k in refs]
        exp = [n for n in names[: len(r["expected_ids"])]]
        tot["recall"] += sum(n in out for n in exp) / max(1, len(exp)) if exp else 1.0
        tot["invalid"] += len(chk["invalid_refs"])
        tot["coverage"] += chk["coverage"]
    n = max(1, len(rows))
    print(json.dumps({
        "samples": len(rows),
        "additive_recall": round(tot["recall"] / n, 3),
        "invalid_citations_per_answer": round(tot["invalid"] / n, 3),
        "grounded_line_ratio": round(tot["coverage"] / n, 3),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
