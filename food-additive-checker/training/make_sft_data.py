"""LoRA学習用データの生成。

ナレッジベースから架空の食品ラベルを大量に作り、アプリ本番と同じ
「ラベル解析 → RAG検索 → プロンプト作成」を通したうえで、
資料だけを根拠に [D番号] で出典を示す模範回答を教師データにする。

  python -m training.make_sft_data --n 2000
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("EMBED_BACKEND", "bm25")  # データ生成は辞書+BM25で十分

from src import generator, label_parser  # noqa: E402
from src.rag import Retriever  # noqa: E402

BASES = [
    ["豚肉", "豚脂肪", "食塩", "香辛料", "ポークエキス"],
    ["小麦粉", "砂糖", "植物油脂", "鶏卵", "全粉乳"],
    ["果糖ぶどう糖液糖", "果汁(りんご)", "食塩"],
    ["米", "しょうゆ", "砂糖", "みりん"],
    ["鶏肉", "でん粉", "しょうゆ", "砂糖", "食塩", "香辛料"],
    ["魚肉すり身", "でん粉", "食塩", "砂糖", "みりん"],
    ["砂糖", "水あめ", "ココアバター", "全粉乳", "カカオマス"],
    ["大根", "漬け原材料(食塩、醸造酢)"],
    ["じゃがいも", "植物油", "食塩"],
    ["オレンジ(アメリカ)"],
]
SAFE_ADDITIVES = [
    "調味料(アミノ酸等)", "乳化剤", "香料", "pH調整剤", "酸化防止剤(ビタミンC)", "酸化防止剤(ビタミンE)",
    "増粘多糖類", "リン酸塩(Na)", "カゼインNa", "ソルビトール", "酸味料", "膨張剤", "クエン酸",
    "グリシン", "酵素", "ビタミンB1", "炭酸カルシウム", "トレハロース", "セルロース", "乳酸Ca",
]
USAGE = {
    "発色剤": "発色剤", "着色": "着色料", "甘味料": "甘味料", "酸化防止剤": "酸化防止剤",
    "保存料": "保存料", "防かび剤": "防かび剤", "殺菌料": "殺菌料", "増粘": "増粘剤", "小麦粉処理剤": "",
}
TAIL = ["内容量 200g", "賞味期限 枠外に記載", "保存方法 直射日光を避けて保存", "製造者 サンプル食品株式会社"]


def usage_of(cat: str) -> str:
    for k, v in USAGE.items():
        if k in cat:
            return v
    return ""


def surface_names(doc) -> list[str]:
    """ラベルに書かれうる日本語表記（英字のみの別名は除外）"""
    al = [a for a in doc.data["aliases"] if any(ord(c) > 0x3000 for c in a) or a in ("BHA", "BHT", "OPP")]
    return al or [doc.data["name"]]


def make_label(rng: random.Random, docs) -> str:
    items = rng.sample(SAFE_ADDITIVES, rng.randint(1, 4))
    k = rng.choices([0, 1, 2, 3, 4], weights=[2, 4, 4, 2, 1])[0]
    for d in rng.sample(docs, k):
        name = rng.choice(surface_names(d))
        use = usage_of(d.data["category"])
        items.append(f"{use}({name})" if use and rng.random() < 0.7 else name)
    rng.shuffle(items)
    sep = rng.choice(["/", "／", "、添加物 "])
    base = "、".join(rng.choice(BASES))
    wrap = rng.choice([("(", ")"), ("（", "）")])
    adds = "、".join(i.replace("(", wrap[0]).replace(")", wrap[1]) for i in items)
    lines = [f"名称 {rng.choice(['ソーセージ', '菓子', '清涼飲料水', '弁当', '漬物', 'かまぼこ', 'チョコレート'])}",
             f"{rng.choice(['原材料名', '原材料名:', '原材料'])} {base}{sep}{adds}"]
    lines += rng.sample(TAIL, rng.randint(1, len(TAIL)))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="training/data")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    retr = Retriever()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for _ in range(a.n):
        label = make_label(rng, retr.additive_docs)
        parsed = label_parser.parse_label(label)
        hits = retr.lookup(parsed["additives"], fallback_text=parsed["section"])
        messages, refs = generator.build_messages(parsed["additives"], hits, retr.guides())
        messages.append({"role": "assistant", "content": generator.template_answer(hits, refs)})
        rows.append({"messages": messages, "label": label, "expected_ids": [h["doc"].id for h in hits]})

    n_val = max(20, len(rows) // 20)
    for name, part in (("val", rows[:n_val]), ("train", rows[n_val:])):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(part)} 件 → {out / (name + '.jsonl')}")


if __name__ == "__main__":
    main()
