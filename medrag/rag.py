"""RAG：文書読み込み → 分割 → Embedding（multilingual-e5-small）→ FAISS 検索。

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


@dataclass
class Chunk:
    chunk_id: str
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
    meta = {k: str(meta.get(k, "") or "") for k in META_KEYS}
    meta["title"] = meta["title"] or path.stem
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
                out.append(Chunk(cid, rel, meta["title"], meta["source_url"], meta["version"],
                                 meta["updated"], meta["license"], page, piece))
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


def save_uploaded(docs_dir: Path, filename: str, data: bytes, meta: dict) -> Path:
    """Streamlit からアップロードされた文書をメタデータ付きで保存する。"""
    docs_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.\-ぁ-んァ-ヶ一-龥ー]", "_", Path(filename).name)
    path = docs_dir / safe
    meta = {k: v for k, v in meta.items() if v}
    if path.suffix.lower() == ".pdf":
        path.write_bytes(data)
        (path.with_name(path.name + ".meta.yaml")).write_text(
            yaml.safe_dump(meta, allow_unicode=True), encoding="utf-8")
    else:
        text = data.decode("utf-8", errors="replace")
        _, body = _split_front_matter(text)
        fm = yaml.safe_dump(meta, allow_unicode=True).strip()
        path.write_text(f"---\n{fm}\n---\n{body}", encoding="utf-8")
    return path
