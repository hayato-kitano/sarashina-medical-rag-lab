"""LoRA学習データの作成（データ版を manifest.json で追跡）。

ナレッジベース（data/rag_docs）から架空の食品ラベルを作り、アプリ本番と同じ
「ラベル解析 → 添加物照合 → 参考文書の収集 → プロンプト作成」を通したうえで、
参考文書だけを根拠に [番号] で出典を示す模範回答を教師データにする。
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from .config import resolve
from .label import parse_label
from .pipeline import RISK_LABEL, build_messages, collect_references, template_answer
from .rag import AdditiveMatcher, load_catalog
from .versioning import now_iso, write_json

BASES = [
    ["豚肉", "豚脂肪", "食塩", "香辛料", "ポークエキス"], ["小麦粉", "砂糖", "植物油脂", "鶏卵", "全粉乳"],
    ["果糖ぶどう糖液糖", "果汁(りんご)", "食塩"], ["米", "しょうゆ", "砂糖", "みりん"],
    ["鶏肉", "でん粉", "しょうゆ", "砂糖", "食塩"], ["魚肉すり身", "でん粉", "食塩", "砂糖"],
    ["砂糖", "水あめ", "ココアバター", "全粉乳", "カカオマス"], ["大根", "漬け原材料(食塩、醸造酢)"],
    ["じゃがいも", "植物油", "食塩"], ["オレンジ(アメリカ)"],
]
SAFE = ["調味料(アミノ酸等)", "乳化剤", "香料", "pH調整剤", "酸化防止剤(ビタミンC)", "増粘多糖類",
        "リン酸塩(Na)", "カゼインNa", "ソルビトール", "酸味料", "膨張剤", "クエン酸", "グリシン", "酵素",
        "炭酸カルシウム", "トレハロース", "セルロース", "乳酸Ca"]
USAGE = {"発色剤": "発色剤", "着色": "着色料", "甘味料": "甘味料", "酸化防止剤": "酸化防止剤", "保存料": "保存料",
         "防かび剤": "防かび剤", "殺菌料": "殺菌料", "増粘": "増粘剤"}
TAIL = ["内容量 200g", "賞味期限 枠外に記載", "保存方法 直射日光を避けて保存", "製造者 サンプル食品株式会社"]


def full_answer(hits: list[dict], refs: list[dict]) -> str:
    """GPU向け（添加物ごとの説明）の模範回答。"""
    n_of = {r["doc_id"]: r["n"] for r in refs}
    if not hits:
        return template_answer(hits, refs)
    out = ["## 判定結果", template_answer(hits, refs).split("\n")[0], ""]
    for i, h in enumerate(hits, 1):
        d, n = h["doc"], n_of[h["doc_id"]]
        sec, cur = {}, None
        for line in d["body"].splitlines():
            if line.startswith("#"):
                cur = line.lstrip("# ").strip()
            elif line.strip() and cur:
                sec[cur] = (sec.get(cur, "") + line.strip())
        out += [f"### {i}. {d['title']}（注意度：{RISK_LABEL[d['risk_level']]}）",
                f"- IARC分類：{d['iarc_group']}。[{n}]",
                f"- 理由：{sec.get('発がん性に関する根拠', '')}[{n}]",
                f"- 日本での扱い：{sec.get('日本での扱い', '')}[{n}]", ""]
    out.append("## まとめ")
    out += template_answer(hits, refs).split("\n")[1:]
    return "\n".join(out)


def _surface(rng, doc) -> str:
    al = [a for a in doc.get("aliases") or [] if any(ord(c) > 0x3000 for c in a) or a in ("BHA", "BHT", "OPP")]
    return rng.choice(al or [doc["title"]])


def make_label(rng: random.Random, docs: list[dict]) -> str:
    items = rng.sample(SAFE, rng.randint(1, 4))
    for d in rng.sample(docs, rng.choices([0, 1, 2, 3, 4], weights=[2, 4, 4, 2, 1])[0]):
        name = _surface(rng, d)
        use = next((v for k, v in USAGE.items() if k in d["category"]), "")
        items.append(f"{use}({name})" if use and rng.random() < 0.7 else name)
    rng.shuffle(items)
    sep = rng.choice(["/", "／", "、添加物 "])
    lines = [f"名称 {rng.choice(['ソーセージ', '菓子', '清涼飲料水', '弁当', '漬物', 'かまぼこ'])}",
             f"{rng.choice(['原材料名', '原材料名:', '原材料'])} {'、'.join(rng.choice(BASES))}{sep}{'、'.join(items)}"]
    return "\n".join(lines + rng.sample(TAIL, rng.randint(1, len(TAIL))))


def prepare(cfg: dict, n: int | None = None) -> dict:
    dcfg = cfg["data"]
    rng = random.Random(int(dcfg["seed"]))
    catalog = load_catalog(cfg)
    matcher = AdditiveMatcher(catalog, None, cfg)
    docs = list(matcher.catalog.values())
    rows = []
    for i in range(int(n or dcfg["n_samples"])):
        label = make_label(rng, docs)
        parsed = parse_label(label)
        hits = matcher.lookup(parsed["additives"], fallback_text=parsed["section"])
        refs = collect_references(hits, catalog, None)
        short = rng.random() < 0.5   # CPU向け要約 / GPU向け詳細 の両方を学習
        messages = build_messages(parsed["additives"], refs, cfg, short=short)
        target = template_answer(hits, refs) if short else full_answer(hits, refs)
        rows.append({"id": f"s{i:05d}", "source": "synthetic",
                     "messages": messages + [{"role": "assistant", "content": target}]})
    rng.shuffle(rows)
    n_val = max(20, int(len(rows) * float(dcfg["split"]["validation"])))
    blob = json.dumps(rows, ensure_ascii=False).encode()
    version = "data-" + hashlib.sha256(blob).hexdigest()[:10]
    out = resolve(dcfg["processed_dir"]) / version
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("validation", rows[:n_val]), ("train", rows[n_val:])):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"data_version": version, "created_at": now_iso(), "n_train": len(rows) - n_val,
                "n_validation": n_val, "sources": ["synthetic labels from data/rag_docs"],
                "seed": dcfg["seed"]}
    write_json(out / "manifest.json", manifest)
    (resolve(dcfg["processed_dir"]) / "LATEST").write_text(version, encoding="utf-8")
    return manifest


def resolve_data_version(cfg: dict, version: str = "latest") -> Path:
    root = resolve(cfg["data"]["processed_dir"])
    if version == "latest":
        version = (root / "LATEST").read_text(encoding="utf-8").strip()
    return root / version
