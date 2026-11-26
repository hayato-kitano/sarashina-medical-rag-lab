"""添加物ナレッジベースの検索（RAG）。

1. 別名辞書による完全一致（最長一致優先）… 表示名のゆれ（亜硝酸Na / 亜硝酸ナトリウム）を確実に拾う
2. Sarashina-Embedding によるベクトル検索 … 辞書にない表記・OCR誤読を補う
   （モデルが使えない環境では文字n-gram BM25で代替）
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from . import config


@dataclass
class Doc:
    id: str
    data: dict

    @property
    def text(self) -> str:
        d = self.data
        return (
            f"{d['name']}（{d['category']}）IARC分類: {d['iarc_group']}。"
            f"{d['summary']} {d['evidence']} {d['japan_status']}"
        )


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).lower().replace(" ", "")


# 用途名・一括名（具体的な物質を表さないので意味検索にかけない）
USAGE_NAMES = {_n for _n in (
    "調味料", "酸化防止剤", "保存料", "着色料", "発色剤", "甘味料", "香料", "乳化剤", "ph調整剤", "酸味料",
    "増粘剤", "安定剤", "ゲル化剤", "増粘多糖類", "膨張剤", "かんすい", "苦味料", "光沢剤", "ガムベース",
    "イーストフード", "酵素", "軟化剤", "豆腐用凝固剤", "防かび剤", "漂白剤", "ビタミンc", "ビタミンe",
    "アミノ酸等", "核酸等", "na", "k", "ca",
)}


def _surface(item: str, alias: str) -> str:
    """照合した別名を、ラベル上の表記のまま取り出す。"""
    i = _norm(item).find(alias)
    return item.replace(" ", "")[i:i + len(alias)] if i >= 0 else item


def load_kb(path=config.KB_PATH) -> list[Doc]:
    with open(path, encoding="utf-8") as f:
        return [Doc(d["id"], d) for d in (json.loads(l) for l in f if l.strip())]


# ---------------------------------------------------------------- 辞書照合
class AliasMatcher:
    def __init__(self, docs: list[Doc]):
        pairs = []
        for d in docs:
            for a in d.data.get("aliases", []):
                a = _norm(a)
                if len(a) >= 2:
                    pairs.append((a, d.id))
        self.pairs = sorted(pairs, key=lambda p: -len(p[0]))  # 長い別名から照合

    def find(self, text: str) -> list[tuple[str, str]]:
        """[(doc_id, 照合した表記)]。一度照合した部分は伏せ字にして二重照合を防ぐ。"""
        buf = _norm(text)
        hits = []
        for alias, doc_id in self.pairs:
            start = 0
            while (i := buf.find(alias, start)) != -1:
                hits.append((i, doc_id, alias))
                buf = buf[:i] + "\0" * len(alias) + buf[i + len(alias):]
                start = i + len(alias)
        hits.sort()
        seen, out = set(), []
        for _, doc_id, alias in hits:
            if doc_id not in seen:
                seen.add(doc_id)
                out.append((doc_id, alias))
        return out


# ---------------------------------------------------------------- ベクトル検索
class BM25:
    """文字2-gram BM25（モデル不要のフォールバック）"""

    def __init__(self, texts: list[str], k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.toks = [self._tok(t) for t in texts]
        self.avg = sum(map(len, self.toks)) / max(1, len(self.toks))
        df = Counter(g for t in self.toks for g in set(t))
        n = len(self.toks)
        self.idf = {g: math.log(1 + (n - c + 0.5) / (c + 0.5)) for g, c in df.items()}
        self.tf = [Counter(t) for t in self.toks]

    @staticmethod
    def _tok(s: str) -> list[str]:
        s = _norm(s)
        return [s[i:i + 2] for i in range(len(s) - 1)] or [s]

    def scores(self, q: str) -> list[float]:
        qt = self._tok(q)
        out = []
        for tf, toks in zip(self.tf, self.toks):
            s = 0.0
            for g in qt:
                if g in tf:
                    f = tf[g]
                    s += self.idf[g] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(toks) / self.avg))
            out.append(s)
        return out


class SarashinaEmbedder:
    """sbintuitions/sarashina-embedding-v1-1b（sentence-transformers形式）"""

    def __init__(self, docs: list[Doc]):
        from sentence_transformers import SentenceTransformer
        import numpy as np

        self.np = np
        self.model = SentenceTransformer(config.EMBED_MODEL, trust_remote_code=True)
        # 名称・別名を重視した短い文書で埋め込む（添加物名の照合が目的のため）
        self.texts = [f"{d.data['name']} " + " ".join(d.data.get("aliases", [])) + f" {d.data['category']}" for d in docs]
        self.mat = self._enc(self.texts)

    def _enc(self, texts):
        v = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return v

    def scores(self, q: str) -> list[float]:
        v = self._enc([q])[0]
        return (self.mat @ v).tolist()


class Retriever:
    def __init__(self, docs: list[Doc] | None = None):
        self.docs = docs or load_kb()
        self.by_id = {d.id: d for d in self.docs}
        self.additive_docs = [d for d in self.docs if d.data["type"] == "additive"]
        self.alias = AliasMatcher(self.additive_docs)
        self.backend = "bm25"
        self.threshold = config.BM25_THRESHOLD
        self.vec = None
        if config.EMBED_BACKEND == "sarashina" and not config.MOCK:
            try:
                self.vec = SarashinaEmbedder(self.additive_docs)
                self.backend, self.threshold = "sarashina-embedding", config.SEMANTIC_THRESHOLD
            except Exception as e:  # モデル取得不可などの場合
                print(f"[rag] Sarashina-Embedding を読み込めないため BM25 を使用します: {e}")
        if self.vec is None:
            self.vec = BM25([f"{d.data['name']} " + " ".join(d.data["aliases"]) for d in self.additive_docs])

    def semantic(self, query: str) -> tuple[Doc, float] | None:
        sc = self.vec.scores(query)
        i = max(range(len(sc)), key=sc.__getitem__)
        return (self.additive_docs[i], sc[i]) if sc[i] >= self.threshold else None

    def lookup(self, additive_items: list[str], fallback_text: str = "") -> list[dict]:
        """添加物リストに該当するKB文書を返す。"""
        from .label_parser import expand

        found: dict[str, dict] = {}
        for item in additive_items or [fallback_text]:
            hits = self.alias.find(item)
            for doc_id, alias in hits:
                found.setdefault(doc_id, {"doc": self.by_id[doc_id], "matched": _surface(item, alias), "method": "辞書一致"})
            if hits or not additive_items:
                continue
            # 辞書で見つからない表記は、物質名（括弧内）を意味検索にかける。用途名は対象外。
            names = expand(item)
            cands = names[2:] if len(names) > 1 else names
            for cand in cands:
                if len(_norm(cand)) < 3 or _norm(cand) in USAGE_NAMES:
                    continue
                r = self.semantic(cand)
                if r and r[0].id not in found:
                    found[r[0].id] = {"doc": r[0], "matched": item, "method": f"意味検索({self.backend}:{r[1]:.2f})"}
        order = {"high": 0, "medium": 1, "low": 2}
        return sorted(found.values(), key=lambda h: (order.get(h["doc"].data["risk_level"], 3), h["doc"].id))

    def guides(self) -> list[Doc]:
        return [d for d in self.docs if d.data["type"] == "guide"]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()
