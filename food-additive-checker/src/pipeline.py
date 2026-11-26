"""画像 → OCR → 添加物抽出 → RAG検索 → 根拠付き回答"""
from __future__ import annotations

from PIL import Image

from . import config, generator, label_parser, vision
from .rag import get_retriever

DISCLAIMER = (
    "この結果は公開情報（IARC・各国機関）に基づく一般的な情報で、医学的な診断ではありません。"
    "IARC分類は「がんを起こす証拠の強さ」であり、通常の摂取量で病気になるかどうかを示すものではありません。"
)


def analyze_text(label_text: str) -> dict:
    parsed = label_parser.parse_label(label_text)
    retriever = get_retriever()
    hits = retriever.lookup(parsed["additives"], fallback_text=parsed["section"])
    if config.LLM_MODE == "full":
        gen = generator.answer(parsed["additives"], hits, retriever.guides())
    else:
        gen = generator.summarize(parsed["additives"], hits, retriever.guides())

    key_of = {v["id"]: k for k, v in gen["refs"].items()}
    findings = [
        {
            "ref": key_of[h["doc"].id],
            "label_text": h["matched"],
            "name": h["doc"].data["name"],
            "risk_level": h["doc"].data["risk_level"],
            "iarc_group": h["doc"].data["iarc_group"],
            "summary": h["doc"].data["summary"],
            "match_method": h["method"],
            "category": h["doc"].data["category"],
            "evidence": h["doc"].data["evidence"],
            "japan_status": h["doc"].data["japan_status"],
            "advice": h["doc"].data["advice"],
        }
        for h in hits
    ]
    sources = {
        k: {"name": v["name"], "evidence": v["evidence"], "sources": v["sources"]}
        for k, v in gen["refs"].items()
    }
    return {
        "ocr_text": label_text,
        "ingredients_section": parsed["section"],
        "additives": parsed["additives"],
        "findings": findings,
        "explanation": gen["text"],
        "sources": sources,
        "verification": gen["verification"],
        "fallback_used": gen["fallback"],
        "llm_mode": config.LLM_MODE,
        "llm_error": gen.get("error"),
        "retrieval_backend": retriever.backend,
        "disclaimer": DISCLAIMER,
    }


def analyze_image(image: Image.Image) -> dict:
    return analyze_text(vision.read_label(image))
