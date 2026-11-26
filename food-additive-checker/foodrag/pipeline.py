"""食品ラベルの文字 → 添加物の抽出 → RAG（文書照合・検索）→ Sarashina で根拠付き説明 → 根拠チェック。

sarashina-medical-rag-lab の medrag/pipeline.py の流れ（検索は1回・同じ参考文書をモデルへ渡す・
出典番号 [1] を付けて回答・根拠不足なら保留）に合わせています。
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .label import parse_label
from .rag import AdditiveMatcher, lexical_overlap
from .safety import DISCLAIMER
from .text import is_readable
from .versioning import now_iso, short_hash

RISK_LABEL = {"high": "高", "medium": "中", "low": "低", "info": "-"}
SECTION_HEADS = {"概要", "発がん性に関する根拠", "日本での扱い", "気をつけること", "詳しい説明", "日本での状況", "注意"}


def summary_line(text: str) -> str:
    """文書の本文から「概要」の1文目（見出し行を除いた最初の行）を取り出す。"""
    for line in text.splitlines():
        s = re.sub(r"^#+\s*", "", line).strip()
        if s and s not in SECTION_HEADS:
            return s
    return ""
_CITE = re.compile(r"\[(\d+)\]")


@dataclass
class Finding:
    ref: int
    doc_id: str
    name: str
    label_text: str          # ラベル上の表記（読めない文字列の場合は空）
    risk_level: str
    iarc_group: str
    category: str
    summary: str
    method: str


@dataclass
class AnalysisResult:
    ocr_text: str
    ocr_engine: str
    ingredients_section: str
    additives: list[str]
    findings: list[Finding]
    references: list[dict]
    answer: str
    answer_source: str        # sarashina / sarashina+資料 / 資料
    raw_output: str = ""
    coverage: float = 0.0
    generation_mode: str = ""
    time_sec: float = 0.0
    error: str = ""
    disclaimer: str = DISCLAIMER
    prompt_hash: str = ""
    timestamp: str = field(default_factory=now_iso)
    versions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---- 参考文書 -----------------------------------------------------------------
def collect_references(hits: list[dict], catalog: dict, retriever) -> list[dict]:
    """見つかった添加物の文書（＋解説文書）を、インデックスのチャンク単位で集める。"""
    doc_ids = [h["doc_id"] for h in hits] + ["G01"] + ([] if hits else ["G02"])
    refs = []
    for n, doc_id in enumerate(dict.fromkeys(doc_ids), 1):
        m = catalog.get(doc_id)
        if not m:
            continue
        chunks = [c["text"] for c in (retriever.chunks if retriever else []) if c.get("doc_id") == doc_id]
        text = "\n".join(chunks) if chunks else re.sub(r"^#+\s*", "", m["body"], flags=re.M).strip()
        refs.append({"n": n, "doc_id": doc_id, "title": m["title"], "text": text,
                     "risk_level": m.get("risk_level", ""), "iarc_group": m.get("iarc_group", ""),
                     "source_url": m.get("source_url", ""), "sources": m.get("sources") or [],
                     "version": m.get("version", ""), "updated": m.get("updated", ""),
                     "license": m.get("license", ""), "doc_path": m.get("path", "")})
    return refs


def format_context(refs: list[dict], short: bool = False) -> str:
    lines = ["【参考文書】"]
    for r in refs:
        head = f"[{r['n']}] 出典: {r['title']}"
        if r.get("iarc_group") and r["iarc_group"] != "-":
            head += f"／IARC分類: {r['iarc_group']}／注意度: {RISK_LABEL.get(r['risk_level'], '-')}"
        lines.append(head)
        text = r["text"]
        lines.append(summary_line(text) if short else text)
        lines.append("")
    return "\n".join(lines)


def build_messages(additives: list[str], refs: list[dict], cfg: dict, short: bool) -> list[dict]:
    add = "、".join(additives) if additives else "（読み取れませんでした）"
    q = ("この食品の添加物について、がんとの関連の観点から3〜5文で要点をまとめてください。" if short else
         "この食品の添加物に、がんとの関連が指摘されているものはありますか？"
         "該当する添加物ごとに注意度・IARC分類・理由・日本での扱いを説明し、最後にまとめを書いてください。")
    user = (f"{cfg['rag']['rag_instruction'].strip()}\n\n{format_context(refs, short)}\n"
            f"【食品ラベルの添加物】\n{add}\n\n【質問】\n{q}")
    return [{"role": "system", "content": cfg["system_prompt"].strip()},
            {"role": "user", "content": user}]


# ---- 根拠チェック ---------------------------------------------------------------
def ground_sentences(raw: str, refs: list[dict], min_support: float, auto_cite_min: float) -> list[str]:
    """モデル出力を文に分け、参考文書と内容が一致する文だけを残す（出典番号の補完つき）。"""
    by_n = {r["n"]: r for r in refs}
    raw = _CITE.sub(lambda m: m.group(0) if int(m.group(1)) in by_n else "", raw or "")
    parts = [p.strip() for p in re.split(r"(?<=[。．！？])|\n", raw) if p and p.strip()]
    sents: list[str] = []
    for p in parts:  # 「…です。[1]」の出典番号を前の文に戻す
        if sents and re.fullmatch(r"(\[\d+\])+", p):
            sents[-1] += p
        else:
            sents.append(p)
    kept, seen = [], set()
    for snt in sents:
        body = _CITE.sub("", snt).strip().lstrip("-・* ").strip()
        if len(body) < 8 or body in seen or not re.search(r"[。．！？]$", body):
            continue  # 短すぎる・重複・途中で切れた文は使わない
        keys = [int(k) for k in _CITE.findall(snt) if int(k) in by_n]
        if keys:
            sup = lexical_overlap(body, " ".join(by_n[k]["title"] + by_n[k]["text"] for k in keys))
        else:
            scored = [(lexical_overlap(body, r["title"] + r["text"]), r["n"]) for r in refs]
            sup, best = max(scored) if scored else (0.0, None)
            keys = [best] if best is not None and sup >= auto_cite_min else []
        if keys and sup >= min_support:
            seen.add(body)
            kept.append(body + "".join(f"[{k}]" for k in dict.fromkeys(keys)))
    return kept


def coverage_of(text: str, refs: list[dict]) -> float:
    by_n = {r["n"]: r for r in refs}
    lines = [l for l in text.splitlines() if l.strip() and not l.startswith("#")]
    ok = 0
    for l in lines:
        keys = [int(k) for k in _CITE.findall(l) if int(k) in by_n]
        if keys and lexical_overlap(_CITE.sub("", l), " ".join(by_n[k]["title"] + by_n[k]["text"]
                                                               for k in keys)) >= 0.5:
            ok += 1
    return round(ok / max(1, len(lines)), 2)


# ---- 資料からの説明（モデルを使わない／根拠不足時） ----------------------------------
def template_answer(hits: list[dict], refs: list[dict]) -> str:
    n_of = {r["doc_id"]: r["n"] for r in refs}
    g = f"[{n_of['G01']}]" if "G01" in n_of else ""
    if not hits:
        g2 = f"[{n_of['G02']}]" if "G02" in n_of else ""
        return ("読み取れた添加物の中に、参考文書でがんとの関連が指摘されているものは見つかりませんでした。" + g + "\n"
                "「香料」「乳化剤」などの一括名で表示される添加物は個別の物質が分からないため、判定の対象外です。" + g2)
    out = [f"がんとの関連について情報がある添加物が{len(hits)}件見つかりました。{g}"]
    for lv, word in (("high", "IARCでグループ2A以上に関連づけられており、摂取頻度に注意が必要です"),
                     ("medium", "グループ2Bに分類されているか、海外で規制上の論点がある添加物です"),
                     ("low", "現時点でヒトへの発がん性を示す確立した証拠はありません")):
        hs = [h for h in hits if h["doc"]["risk_level"] == lv]
        if hs:
            names = "、".join(h["doc"]["title"] for h in hs)
            out.append(f"{names}は{word}。" + "".join(f"[{n_of[h['doc_id']]}]" for h in hs))
    out.append(f"IARC分類は有害性の証拠の強さを示すもので、通常の摂取量でのリスクの大きさではありません。{g}")
    return "\n".join(out)


def _gen_mode(cfg: dict, model) -> str:
    mode = cfg["generation"].get("mode", "auto")
    if model is None:
        return "off"
    if mode == "auto":
        return "full" if getattr(model, "device", "cpu") == "cuda" else "summary"
    return mode


# ---- 全体 -----------------------------------------------------------------------
def analyze_text(text: str, ocr_engine: str, model, retriever, catalog: dict, cfg: dict) -> AnalysisResult:
    parsed = parse_label(text)
    matcher = AdditiveMatcher(catalog, retriever, cfg)
    hits = matcher.lookup(parsed["additives"], fallback_text=parsed["section"])
    refs = collect_references(hits, catalog, retriever)
    n_of = {r["doc_id"]: r["n"] for r in refs}

    findings = [Finding(ref=n_of[h["doc_id"]], doc_id=h["doc_id"], name=h["doc"]["title"],
                        label_text=h["label_text"] if is_readable(h["label_text"]) else "",
                        risk_level=h["doc"]["risk_level"], iarc_group=h["doc"]["iarc_group"],
                        category=h["doc"]["category"],
                        summary=summary_line(h["doc"]["body"]),
                        method=h["method"]) for h in hits]

    mode = _gen_mode(cfg, model)
    gcfg, grd = cfg["generation"], cfg["grounding"]
    messages = build_messages(parsed["additives"], refs, cfg, short=(mode != "full"))
    answer, source, raw, err, elapsed = "", "資料", "", "", 0.0
    if mode != "off":
        try:
            max_new = gcfg["max_new_tokens_full"] if mode == "full" else gcfg["max_new_tokens_summary"]
            out = model.generate(messages, gcfg, max_new)
            raw, elapsed = out.text, out.time_sec
            kept = ground_sentences(raw, refs, float(grd["min_support"]), float(grd["auto_cite_min"]))
            if kept:
                source = "sarashina"
                joined = "".join(kept)
                for h in hits:  # 注意度「高」に触れていなければ資料の一文で補う
                    n = n_of[h["doc_id"]]
                    if h["doc"]["risk_level"] == "high" and f"[{n}]" not in joined:
                        f = next(x for x in findings if x.doc_id == h["doc_id"])
                        kept.insert(0, f"（資料より）{f.name}：{f.summary}[{n}]")
                        source = "sarashina+資料"
                answer = "\n".join(kept)
        except Exception as e:  # noqa: BLE001  メモリ不足などでもアプリを止めない
            err = f"{type(e).__name__}: {e}"
    if not answer:
        answer, source = template_answer(hits, refs), "資料"

    versions = {**(model.version_info() if model is not None else {}),
                "rag_index_version": getattr(retriever, "version", ""),
                "system_prompt_hash": short_hash(cfg["system_prompt"])}
    return AnalysisResult(
        ocr_text=text, ocr_engine=ocr_engine, ingredients_section=parsed["section"],
        additives=parsed["additives"], findings=findings, references=refs, answer=answer,
        answer_source=source, raw_output=raw, coverage=coverage_of(answer, refs), generation_mode=mode,
        time_sec=elapsed, error=err, prompt_hash=short_hash(json.dumps(messages, ensure_ascii=False)),
        versions=versions)
