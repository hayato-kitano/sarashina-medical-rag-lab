"""設定ファイルの読み込みとプロジェクト内パスの解決。"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """config.yaml を読み込む。環境変数 MEDRAG_CONFIG でも差し替え可能。"""
    path = Path(path or os.environ.get("MEDRAG_CONFIG", DEFAULT_CONFIG))
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "_base" in cfg:  # 別の設定ファイルを継承して一部だけ上書きする
        base = load_config((path.parent / cfg.pop("_base")).resolve())
        base.pop("_config_path", None)
        cfg = deep_update(base, cfg)
    if overrides:
        cfg = deep_update(cfg, overrides)
    cfg["_config_path"] = str(path)
    return cfg


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def resolve(p: str | os.PathLike) -> Path:
    """相対パスはプロジェクトルート基準で解決する。"""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def model_ref(name: str) -> str:
    """ローカルに存在するパスならプロジェクト基準の絶対パス、それ以外は Hugging Face のモデルIDとして扱う。"""
    p = resolve(name)
    return str(p) if p.exists() else name
