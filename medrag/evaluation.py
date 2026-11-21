"""評価機能（要件 7）：一括実行・手動採点・指標別集計・CSV出力・実験間比較、JMedQA 正答率。"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import resolve
from .pipeline import MODE_LABELS, MODES, result_rows, run_comparison
from .versioning import now_id, now_iso, read_json, write_json

# 手動採点の列（自由記述の正確性・安全性は文字列一致で採点しない → 評価者が入力）
SCORE_COLUMNS = {
    "score_accuracy": "医療回答の正確性(1-5)",
    "score_relevance": "回答の関連性(1-5)",
    "score_grounding": "根拠との整合性(1-5, RAG時)",
    "score_abstention": "根拠不足時の応答(OK/NG)",
    "score_safety": "医療安全性(OK/NG)",
    "reviewer": "評価者",
    "comment": "コメント",
}
NUMERIC_SCORES = ["score_accuracy", "score_relevance", "score_grounding"]
BINARY_SCORES = ["score_abstention", "score_safety"]


def load_questions(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(resolve(path), encoding="utf-8-sig")
    for col in ("id", "category", "question"):
        if col not in df.columns:
            raise ValueError(f"評価質問CSVに列 '{col}' がありません")
    return df


def run_batch(questions: pd.DataFrame, modes: list[str], model, retriever, cfg: dict,
              progress: Callable[[float, str], None] | None = None, name: str = "") -> Path:
    """事前登録した質問を指定モードで一括実行し、outputs/eval_runs/<run_id>/ に保存する。"""
    rag_settings = sorted({MODES[m][1] for m in modes})
    run_id = now_id("eval-") + (f"-{name}" if name else "")
    out_dir = resolve(cfg["evaluation"]["runs_dir"]) / run_id
    rows: list[dict] = []
    total = len(questions) * len(rag_settings)
    done = 0
    for _, q in questions.iterrows():
        for use_rag in rag_settings:
            which = tuple(k for k in ("base", "lora") if mode_for(k, use_rag) in modes)
            res = run_comparison(str(q["question"]), use_rag, model, retriever, cfg, which=which)
            extra = {"eval_run_id": run_id, "question_id": q["id"], "category": q["category"],
                     "expected_points": q.get("expected_points", "")}
            rows.extend(result_rows(res, extra))
            done += 1
            if progress:
                progress(done / total, f"{q['id']} RAG={'ON' if use_rag else 'OFF'}")
    df = pd.DataFrame(rows)
    for c in SCORE_COLUMNS:
        df[c] = ""
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results.csv", index=False, encoding="utf-8-sig")
    write_json(out_dir / "manifest.json", {
        "eval_run_id": run_id, "created_at": now_iso(), "modes": modes,
        "n_questions": len(questions), "versions": model.version_info(),
        "rag_index_version": retriever.version if retriever else "",
        "generation": cfg["generation"], "rag": {k: v for k, v in cfg["rag"].items() if k != "rag_instruction"},
    })
    return out_dir


def mode_for(key: str, use_rag: bool) -> str:
    return next(m for m, v in MODES.items() if v == (key == "lora", use_rag))


def list_eval_runs(cfg: dict) -> list[Path]:
    root = resolve(cfg["evaluation"]["runs_dir"])
    return sorted((p for p in root.glob("eval-*") if (p / "results.csv").exists()), reverse=True) \
        if root.exists() else []


def load_run(run_dir: Path) -> pd.DataFrame:
    scored = run_dir / "scored.csv"
    df = pd.read_csv(scored if scored.exists() else run_dir / "results.csv", encoding="utf-8-sig",
                     keep_default_na=False)
    return df


def save_scores(run_dir: Path, df: pd.DataFrame) -> Path:
    path = run_dir / "scored.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """モード（A〜D）別に、自動指標（時間・トークン）と手動採点を集計する。"""
    d = df.copy()
    for c in NUMERIC_SCORES:
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    for c in BINARY_SCORES:
        d[c + "_ok"] = d.get(c, "").astype(str).str.upper().map({"OK": 1.0, "NG": 0.0})
    gen = d["status"] == "generated"
    d["time_gen"] = pd.to_numeric(d["time_sec"], errors="coerce").where(gen)
    d["out_tok"] = pd.to_numeric(d["output_tokens"], errors="coerce").where(gen)
    d["in_tok"] = pd.to_numeric(d["input_tokens"], errors="coerce").where(gen)
    d["flagged"] = d["auto_safety_flags"].astype(str).str.len() > 0
    g = d.groupby("mode")
    out = pd.DataFrame({
        "件数": g.size(),
        "生成件数": g["status"].apply(lambda s: (s == "generated").sum()),
        "保留件数": g["status"].apply(lambda s: (s == "abstained").sum()),
        "緊急案内件数": g["status"].apply(lambda s: (s == "emergency_guide").sum()),
        "平均推論時間(秒)": g["time_gen"].mean().round(3),
        "平均入力トークン": g["in_tok"].mean().round(1),
        "平均出力トークン": g["out_tok"].mean().round(1),
        "自動安全性フラグ数": g["flagged"].sum(),
        "正確性(平均)": g["score_accuracy"].mean().round(2),
        "関連性(平均)": g["score_relevance"].mean().round(2),
        "根拠整合性(平均)": g["score_grounding"].mean().round(2),
        "根拠不足時OK率": g["score_abstention_ok"].mean().round(3),
        "安全性OK率": g["score_safety_ok"].mean().round(3),
        "採点済み件数": g["score_accuracy"].count(),
    })
    out.index = [MODE_LABELS.get(m, m) for m in out.index]
    return out


def compare_runs(cfg: dict, run_dirs: list[Path]) -> pd.DataFrame:
    """学習実験（アダプター）間の結果比較。"""
    frames = []
    for rd in run_dirs:
        man = read_json(rd / "manifest.json")
        agg = aggregate(load_run(rd))
        agg.insert(0, "adapter_id", man["versions"].get("adapter_id", ""))
        agg.insert(0, "eval_run_id", man["eval_run_id"])
        frames.append(agg.reset_index(names="mode"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---- JMedQA（医師国家試験・選択式）------------------------------------------
def prepare_jmedqa(out_path: str | Path, limit: int | None = None) -> dict:
    """SIP-med-LLM/JMedQA から画像不要のテキスト選択問題だけを抽出して JSONL 化する。"""
    from datasets import load_dataset
    ds = load_dataset("SIP-med-LLM/JMedQA", split="benchmark")
    rows, skipped = [], {"image": 0, "non_option": 0}
    for r in ds:
        if r.get("image") or r.get("has_unpublished_image") or r.get("image_dependency") != "none":
            skipped["image"] += 1
            continue
        if r.get("answer_mode") != "option":
            skipped["non_option"] += 1
            continue
        rows.append({"id": r["problem_unique_id"], "year": r["year"], "clinical_area": r["clinical_area"],
                     "question": r["question"], "options": json.loads(r["options_json"]),
                     "answer": sorted(json.loads(r["answer_json"])), "answer_count": r["answer_count"]})
    if limit:
        rows = rows[:limit]
    p = resolve(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"n": len(rows), "skipped": skipped, "path": str(p)}


def jmedqa_messages(item: dict, system_prompt: str) -> list[dict]:
    opts = "\n".join(f"{k}. {v}" for k, v in sorted(item["options"].items()))
    n = item.get("answer_count", 1)
    instr = (f"次の医師国家試験問題に答えてください。正解は{n}つです。"
             f"選択肢の記号（a〜e）だけを「答え: a」の形式で出力してください。")
    return [{"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": f"{instr}\n\n{item['question']}\n\n{opts}"}]


def parse_choice(text: str, n: int) -> list[str]:
    t = text.lower()
    m = re.search(r"答え\s*[:：]\s*([a-e](?:\s*[,、と]?\s*[a-e])*)", t)
    seg = m.group(1) if m else t
    found = []
    for c in re.findall(r"(?<![a-z])([a-e])(?![a-z])", seg):
        if c not in found:
            found.append(c)
    return sorted(found[:n])


def run_jmedqa(model, cfg: dict, items: list[dict], use_lora: bool,
               progress: Callable[[float, str], None] | None = None) -> pd.DataFrame:
    gen = {**cfg["generation"], "max_new_tokens": 16}
    rows = []
    for i, it in enumerate(items):
        out = model.generate(jmedqa_messages(it, cfg["system_prompt"]), use_lora=use_lora, gen_cfg=gen)
        pred = parse_choice(out.text, int(it.get("answer_count", 1)))
        rows.append({"id": it["id"], "clinical_area": it.get("clinical_area", ""),
                     "model": "lora" if use_lora else "base", "answer": ",".join(it["answer"]),
                     "pred": ",".join(pred), "correct": pred == sorted(it["answer"]),
                     "raw_output": out.text, "time_sec": out.time_sec})
        if progress:
            progress((i + 1) / len(items), it["id"])
    return pd.DataFrame(rows)
