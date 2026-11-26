"""データ・モデル・RAG文書のバージョン追跡（MVP完成条件：バージョンを追跡できる）。"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Iterable


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def short_hash(text: str, n: int = 10) -> str:
    return sha256_text(text)[:n]


def hash_files(paths: Iterable[Path], root: Path | None = None) -> str:
    """ファイル群の内容（と相対パス）からハッシュを計算する。"""
    h = hashlib.sha256()
    for p in sorted(Path(x) for x in paths):
        rel = str(p.relative_to(root)) if root else p.name
        h.update(rel.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def now_id(prefix: str = "") -> str:
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}{ts}" if prefix else ts


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def library_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for name in ["torch", "transformers", "peft", "trl", "datasets", "faiss"]:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception:  # noqa: BLE001
            out[name] = "not installed"
    return out
