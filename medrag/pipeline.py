"""質問 → 安全性チェック → 検索（1回だけ）→ 学習前/学習後で生成 の比較パイプライン。

公平性のため：
- システムプロンプト・生成パラメーターは両モデル共通
- RAG ON の場合、検索は1回だけ行い、同じ検索結果（同じプロンプト）を両モデルへ渡す
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .safety import detect_dangerous_advice, detect_emergency, emergency_message
from .versioning import now_iso, short_hash

# 比較モード（要件 6.2）: mode -> (use_lora, use_rag)
MODES: dict[str, tuple[bool, bool]] = {
    "A": (False, False),
    "B": (True, False),
    "C": (False, True),
    "D": (True, True),
}
MODE_LABELS = {
    "A": "A：学習前",
    "B": "B：学習後",
    "C": "C：学習前＋RAG",
    "D": "D：学習後＋RAG",
}


def mode_of(use_lora: bool, use_rag: bool) -> str:
    return next(m for m, v in MODES.items() if v == (use_lora, use_rag))


ABSTAIN_MESSAGE = (
    "登録されている参考文書の中に、この質問に答えるための十分な根拠が見つかりませんでした。"
    "確かな情報源（医療機関・公的機関の資料）を確認するか、医師・薬剤師に相談してください。"
)


@dataclass
class Answer:
    mode: str
    model_label: str
    text: str
    time_sec: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    generated: bool = True             # False = 保留・緊急案内などでモデルを呼ばなかった
    status: str = "generated"          # generated / abstained / emergency_guide / error
    safety_flags: str = ""


@dataclass
class ComparisonResult:
    question: str
    use_rag: bool
    answers: dict[str, Answer]         # "base" / "lora"
    retrieved: list[dict] = field(default_factory=list)
    emergency: str = ""
    insufficient_evidence: bool = False
    prompt_hash: str = ""
    timestamp: str = field(default_factory=now_iso)
    versions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def format_context(chunks: list[dict]) -> str:
    lines = ["【参考文書】"]
    for i, c in enumerate(chunks, 1):
        meta = [f"出典: {c['title']}"]
        if c.get("version"):
            meta.append(f"版: {c['version']}")
        if c.get("updated"):
            meta.append(f"更新日: {c['updated']}")
        lines.append(f"[{i}] " + "／".join(meta))
        lines.append(c["text"].strip())
        lines.append("")
    return "\n".join(lines)


def build_messages(question: str, system_prompt: str, chunks: list[dict] | None,
                   rag_instruction: str = "") -> list[dict]:
    if chunks:
        user = f"{rag_instruction.strip()}\n\n{format_context(chunks)}\n【質問】\n{question.strip()}"
    else:
        user = question.strip()
    return [{"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user}]


def run_comparison(question: str, use_rag: bool, model, retriever, cfg: dict,
                   which: tuple[str, ...] = ("base", "lora"),
                   emergency_override: bool | None = None) -> ComparisonResult:
    """1つの質問について、学習前（base）と学習後（lora）の回答を同条件で生成する。"""
    gen_cfg = cfg["generation"]
    rag_cfg = cfg["rag"]
    labels = {"base": "学習前 Sarashina", "lora": "学習後 Sarashina（LoRA）"}
    versions = {**model.version_info(),
                "rag_index_version": retriever.version if (use_rag and retriever) else "",
                "system_prompt_hash": short_hash(cfg["system_prompt"]),
                "generation_params": json.dumps(gen_cfg, sort_keys=True)}

    def blank(key: str, text: str, status: str) -> Answer:
        return Answer(mode=mode_of(key == "lora", use_rag), model_label=labels[key], text=text,
                      generated=False, status=status)

    # 1) 緊急性チェック
    emg = detect_emergency(question)
    action = cfg["safety"]["emergency_action"]
    if emergency_override is not None:
        action = "guide_and_generate" if emergency_override else "guide_only"
    emergency_text = emergency_message(emg) if emg.flagged else ""
    if emg.flagged and action == "guide_only":
        return ComparisonResult(question, use_rag,
                                {k: blank(k, emergency_text, "emergency_guide") for k in which},
                                emergency=emergency_text, versions=versions)

    # 2) 検索（1回だけ実行し、両モデルで共有）
    chunks: list[dict] = []
    insufficient = False
    if use_rag:
        if retriever is None:
            raise RuntimeError("RAGインデックスがありません。文書を登録してインデックスを作成してください。")
        chunks = [c for c in retriever.search(question, int(rag_cfg["top_k"]))]
        good = [c for c in chunks if c["score"] >= float(rag_cfg["min_score"])]
        insufficient = not good
        if insufficient and rag_cfg.get("insufficient_action", "hold") == "hold":
            return ComparisonResult(question, use_rag,
                                    {k: blank(k, ABSTAIN_MESSAGE, "abstained") for k in which},
                                    retrieved=chunks, emergency=emergency_text,
                                    insufficient_evidence=True, versions=versions)
        chunks = good or chunks

    # 3) 同一プロンプトで生成
    messages = build_messages(question, cfg["system_prompt"], chunks or None,
                              rag_cfg.get("rag_instruction", ""))
    answers = {}
    for key in which:
        out = model.generate(messages, use_lora=(key == "lora"), gen_cfg=gen_cfg)
        flags = detect_dangerous_advice(out.text)
        answers[key] = Answer(mode=mode_of(key == "lora", use_rag), model_label=labels[key],
                              text=out.text, time_sec=out.time_sec, input_tokens=out.input_tokens,
                              output_tokens=out.output_tokens, safety_flags=flags.label)
    return ComparisonResult(question, use_rag, answers, retrieved=chunks, emergency=emergency_text,
                            insufficient_evidence=insufficient,
                            prompt_hash=short_hash(json.dumps(messages, ensure_ascii=False)),
                            versions=versions)


def result_rows(res: ComparisonResult, extra: dict | None = None) -> list[dict]:
    """CSV出力用に1回答1行へ展開する。"""
    refs = " | ".join(f"[{i}] {c['title']} ({c.get('source_url','')}; 版:{c.get('version','')}; "
                      f"更新:{c.get('updated','')}; score={c['score']:.3f})"
                      for i, c in enumerate(res.retrieved, 1))
    rows = []
    for key, a in res.answers.items():
        rows.append({
            **(extra or {}),
            "timestamp": res.timestamp,
            "question": res.question,
            "mode": a.mode,
            "mode_label": MODE_LABELS[a.mode],
            "model": key,
            "lora": key == "lora",
            "rag": res.use_rag,
            "answer": a.text,
            "status": a.status,
            "time_sec": a.time_sec,
            "input_tokens": a.input_tokens,
            "output_tokens": a.output_tokens,
            "auto_safety_flags": a.safety_flags,
            "insufficient_evidence": res.insufficient_evidence,
            "emergency_detected": bool(res.emergency),
            "references": refs,
            "prompt_hash": res.prompt_hash,
            **{f"ver_{k}": v for k, v in res.versions.items()},
        })
    return rows
