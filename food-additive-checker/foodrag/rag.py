"""RAG：添加物文書の読み込み → 分割 → Embedding（multilingual-e5-small）→ FAISS 検索。

sarashina-medical-rag-lab の medrag/rag.py をベースに、添加物用のメタデータ
（id / aliases / risk_level / iarc_group / category / sources）と、ラベル表記からの添加物照合
（AdditiveMatcher：別名の最長一致 → e5 意味検索）を追加しています。

文書メタデータ（参照元文書名・URL・版・更新日・ライセンス）は
- Markdown / TXT: 先頭の YAML front matter（--- で囲む）
- PDF: 同じ場所の `<ファイル名>.meta.yaml`
で指定する。メタデータが無い文書も読み込めるが、画面に「出典情報なし」と表示される。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from .config import resolve
from .versioning import hash_files, now_iso, read_json, write_json

SUPPORTED = {".md", ".markdown", ".txt", ".pdf"}
META_KEYS = ("title", "source_url", "version", "updated", "license", "publisher")
EXTRA_KEYS = ("id", "type", "category", "risk_level", "iarc_group", "aliases", "sources")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_path: str
    title: str
    source_url: str
    version: str
    updated: str
    license: str
    page: int | None
    text: str


# ---- 読み込み ---------------------------------------------------------------
def _split_front_matter(text: str) -> tuple[dict, str]:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, flags=re.S)
    if not m:
        return {}, text
    meta = yaml.safe_load(m.group(1)) or {}
    return meta, text[m.end():]


def load_document(path: Path) -> tuple[dict, list[tuple[int | None, str]]]:
    """(メタデータ, [(ページ番号, 本文), ...]) を返す。"""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = [(i + 1, (p.extract_text() or "")) for i, p in enumerate(reader.pages)]
        side = path.with_name(path.name + ".meta.yaml")
        meta = yaml.safe_load(side.read_text(encoding="utf-8")) if side.exists() else {}
    else:
        meta, body = _split_front_matter(path.read_text(encoding="utf-8"))
        pages = [(None, body)]
    extra = {k: meta.get(k) for k in EXTRA_KEYS if k in meta}
    meta = {k: str(meta.get(k, "") or "") for k in META_KEYS}
    meta["title"] = meta["title"] or path.stem
    meta.update(extra)
    meta.setdefault("id", path.stem)
    return meta, pages


def iter_documents(docs_dir: Path) -> list[Path]:
    return sorted(p for p in docs_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)


# ---- 分割 -------------------------------------------------------------------
_SENT_END = re.compile(r"(?<=[。！？!?])|\n")


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """日本語の文境界を優先して chunk_size 文字程度に分割する（overlap 文字の重なり）。"""
    text = re.sub(r"[ \t　]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text).strip()
    sents = [s for s in _SENT_END.split(text) if s and s.strip()]
    chunks, cur = [], ""
    for s in sents:
        while len(s) > chunk_size:  # 極端に長い文は強制分割
            head, s = s[:chunk_size], s[chunk_size - overlap:]
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(head)
        if len(cur) + len(s) > chunk_size and cur:
            chunks.append(cur)
            cur = cur[-overlap:] if overlap else ""
        cur += s
    if cur.strip():
        chunks.append(cur)
    return [c.strip() for c in chunks if len(c.strip()) >= 10]


def build_chunks(docs_dir: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    out: list[Chunk] = []
    for path in iter_documents(docs_dir):
        meta, pages = load_document(path)
        rel = str(path.relative_to(docs_dir))
        for page, body in pages:
            # Markdown 見出しは本文として残しつつ、記号を除去
            body = re.sub(r"^#+\s*", "", body, flags=re.M)
            for j, piece in enumerate(split_text(body, chunk_size, overlap)):
                cid = hashlib.sha1(f"{rel}:{page}:{j}:{piece}".encode()).hexdigest()[:12]
                out.append(Chunk(cid, str(meta["id"]), rel, meta["title"], meta["source_url"],
                                 meta["version"], meta["updated"], meta["license"], page, piece))
    return out


# ---- Embedding --------------------------------------------------------------
class E5Embedder:
    """intfloat/multilingual-e5-* 用。query:/passage: 接頭辞 + mean pooling + L2正規化。"""

    def __init__(self, name: str, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.name = name
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()

    def _encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        torch = self.torch
        vecs = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                b = self.tok(texts[i:i + batch_size], padding=True, truncation=True,
                             max_length=512, return_tensors="pt").to(self.device)
                h = self.model(**b).last_hidden_state
                mask = b["attention_mask"].unsqueeze(-1).float()
                v = (h * mask).sum(1) / mask.sum(1)
                vecs.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
        return np.concatenate(vecs).astype("float32") if vecs else np.zeros((0, 1), "float32")

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([f"query: {text}"])


class HashingEmbedder:
    """【テスト専用】ネットワーク不要の文字n-gramハッシュEmbedding。本番評価には使わない。"""

    def __init__(self, dim: int = 2048):
        self.name = "hashing"
        self.dim = dim

    def _vec(self, t: str) -> np.ndarray:
        v = np.zeros(self.dim, "float32")
        t = re.sub(r"\s+", "", t)
        for n in (1, 2, 3):
            for i in range(len(t) - n + 1):
                v[int(hashlib.md5(t[i:i + n].encode()).hexdigest(), 16) % self.dim] += 1.0
        nrm = np.linalg.norm(v)
        return v / nrm if nrm else v

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), "float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)[None, :]


def make_embedder(name: str, device: str = "cpu"):
    return HashingEmbedder() if name == "hashing" else E5Embedder(name, device)


# ---- インデックス ---------------------------------------------------------------
def build_index(cfg: dict, embedder=None) -> dict:
    """文書ディレクトリから FAISS インデックスを作成し、版情報（manifest）を保存する。"""
    import faiss
    rcfg = cfg["rag"]
    docs_dir, index_dir = resolve(rcfg["docs_dir"]), resolve(rcfg["index_dir"])
    files = iter_documents(docs_dir)
    if not files:
        raise FileNotFoundError(f"文書がありません: {docs_dir}")
    chunks = build_chunks(docs_dir, int(rcfg["chunk_size"]), int(rcfg["chunk_overlap"]))
    embedder = embedder or make_embedder(rcfg["embedding_model"])
    vecs = embedder.embed_passages([c.text for c in chunks])
    index = faiss.IndexFlatIP(vecs.shape[1])  # 正規化済みベクトルの内積 = コサイン類似度
    index.add(vecs)

    docs_hash = hash_files(files, root=docs_dir)
    params = f"{rcfg['embedding_model']}|{rcfg['chunk_size']}|{rcfg['chunk_overlap']}"
    version = "rag-" + hashlib.sha256((docs_hash + params).encode()).hexdigest()[:10]
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    docs_info = []
    for p in files:
        meta, _ = load_document(p)
        docs_info.append({"path": str(p.relative_to(docs_dir)), **meta,
                          "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:12],
                          "n_chunks": sum(c.doc_path == str(p.relative_to(docs_dir)) for c in chunks)})
    manifest = {"version": version, "created_at": now_iso(), "embedding_model": rcfg["embedding_model"],
                "chunk_size": rcfg["chunk_size"], "chunk_overlap": rcfg["chunk_overlap"],
                "n_documents": len(files), "n_chunks": len(chunks), "documents": docs_info}
    write_json(index_dir / "manifest.json", manifest)
    return manifest


class Retriever:
    def __init__(self, cfg: dict, embedder=None):
        import faiss
        rcfg = cfg["rag"]
        index_dir = resolve(rcfg["index_dir"])
        self.manifest = read_json(index_dir / "manifest.json")
        if self.manifest["embedding_model"] != rcfg["embedding_model"]:
            raise ValueError("インデックス作成時と異なるEmbeddingモデルが設定されています。再作成してください。")
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        with open(index_dir / "chunks.jsonl", encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        self.embedder = embedder or make_embedder(rcfg["embedding_model"])

    @property
    def version(self) -> str:
        return self.manifest["version"]

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        q = self.embedder.embed_query(query)
        scores, idx = self.index.search(q, min(top_k, len(self.chunks)))
        return [{**self.chunks[i], "score": float(s)} for s, i in zip(scores[0], idx[0]) if i >= 0]


def index_exists(cfg: dict) -> bool:
    d = resolve(cfg["rag"]["index_dir"])
    return (d / "index.faiss").exists() and (d / "manifest.json").exists()


def docs_changed_since_index(cfg: dict) -> bool:
    """文書を追加・更新したがインデックスが古いままになっていないか。"""
    if not index_exists(cfg):
        return True
    docs_dir = resolve(cfg["rag"]["docs_dir"])
    man = read_json(resolve(cfg["rag"]["index_dir"]) / "manifest.json")
    current = {str(p.relative_to(docs_dir)): hashlib.sha256(p.read_bytes()).hexdigest()[:12]
               for p in iter_documents(docs_dir)}
    indexed = {d["path"]: d["sha256"] for d in man["documents"]}
    return current != indexed


# ---- 添加物の照合 -------------------------------------------------------------
def load_catalog(cfg: dict) -> dict[str, dict]:
    """文書のメタデータ一覧（doc_id → メタデータ）。"""
    docs_dir = resolve(cfg["rag"]["docs_dir"])
    out = {}
    for p in iter_documents(docs_dir):
        meta, pages = load_document(p)
        meta["path"] = str(p.relative_to(docs_dir))
        meta["body"] = "\n".join(b for _, b in pages)
        out[str(meta["id"])] = meta
    return out


def _norm(s: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFKC", s or "").lower().replace(" ", "")


def _bigrams(s: str) -> set[str]:
    s = _norm(s)
    return {s[i:i + 2] for i in range(len(s) - 1)} or ({s} if s else set())


def lexical_overlap(query: str, target: str) -> float:
    q = _bigrams(query)
    return len(q & _bigrams(target)) / max(1, len(q))


# 用途名・一括名（具体的な物質を表さないので意味検索にかけない）
USAGE_NAMES = {_norm(x) for x in (
    "調味料", "酸化防止剤", "保存料", "着色料", "発色剤", "甘味料", "香料", "乳化剤", "pH調整剤", "酸味料",
    "増粘剤", "安定剤", "ゲル化剤", "増粘多糖類", "膨張剤", "かんすい", "苦味料", "光沢剤", "ガムベース",
    "イーストフード", "酵素", "軟化剤", "豆腐用凝固剤", "防かび剤", "漂白剤", "ビタミンC", "ビタミンE",
    "アミノ酸等", "核酸等", "Na", "K", "Ca",
)}


class AdditiveMatcher:
    """ラベルの添加物表記 → 文書。別名の最長一致を優先し、見つからない表記だけ意味検索する。"""

    def __init__(self, catalog: dict[str, dict], retriever: "Retriever | None", cfg: dict):
        self.catalog = {k: v for k, v in catalog.items() if v.get("type") == "additive"}
        self.retriever = retriever
        self.min_score = float(cfg["rag"]["min_score"])
        self.lexical_min = float(cfg["rag"]["lexical_min"])
        pairs = []
        for doc_id, m in self.catalog.items():
            for a in m.get("aliases") or []:
                a = _norm(a)
                if len(a) >= 2:
                    pairs.append((a, doc_id))
        self.pairs = sorted(pairs, key=lambda p: -len(p[0]))

    def find_aliases(self, text: str) -> list[tuple[str, str]]:
        buf, hits = _norm(text), []
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

    def fuzzy(self, query: str, min_overlap: float = 0.75) -> tuple[str, float] | None:
        """OCRの1〜2文字の誤読（例: ナトリウ厶）を、別名との文字の重なりで拾う。"""
        best = None
        for alias, doc_id in self.pairs:
            if len(alias) < 4:
                continue
            ov = min(lexical_overlap(query, alias), lexical_overlap(alias, query))
            if ov >= min_overlap and (best is None or ov > best[1]):
                best = (doc_id, ov)
        return best

    def semantic(self, query: str) -> tuple[str, float] | None:
        if self.retriever is None:
            return None
        for c in self.retriever.search(query, top_k=3):
            doc = self.catalog.get(c["doc_id"])
            if not doc or c["score"] < self.min_score:
                continue
            names = " ".join([doc["title"], *(doc.get("aliases") or [])])
            if lexical_overlap(query, names) >= self.lexical_min:
                return c["doc_id"], c["score"]
        return None

    def lookup(self, additives: list[str], fallback_text: str = "") -> list[dict]:
        from .label import expand
        found: dict[str, dict] = {}
        for item in additives or [fallback_text]:
            hits = self.find_aliases(item)
            for doc_id, alias in hits:
                found.setdefault(doc_id, {"doc_id": doc_id, "label_text": _surface(item, alias),
                                          "method": "辞書一致", "score": 1.0})
            if hits or not additives:
                continue
            names = expand(item)
            for cand in (names[2:] if len(names) > 1 else names):
                if len(_norm(cand)) < 3 or _norm(cand) in USAGE_NAMES:
                    continue
                r, method = self.fuzzy(cand), "表記ゆれ一致"
                if not r:
                    r, method = self.semantic(cand), "意味検索"
                if r and r[0] not in found:
                    found[r[0]] = {"doc_id": r[0], "label_text": cand, "method": method, "score": r[1]}
        order = {"high": 0, "medium": 1, "low": 2}
        hits = []
        for h in found.values():
            h["doc"] = self.catalog[h["doc_id"]]
            hits.append(h)
        return sorted(hits, key=lambda h: (order.get(h["doc"]["risk_level"], 3), h["doc_id"]))


def _surface(item: str, alias: str) -> str:
    i = _norm(item).find(alias)
    return item.replace(" ", "")[i:i + len(alias)] if i >= 0 else item
