"""Sarashina2.2-0.5B-Instruct（＋任意のLoRAアダプター）の読み込みと生成。

sarashina-medical-rag-lab の medrag/models.py をベースにしています。
アダプターは models/adapters/LATEST が指すものを読み込み、無ければベースモデルのみで動作します。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from .config import model_ref, resolve
from .versioning import read_json


@dataclass
class GenerationOutput:
    text: str
    time_sec: float
    input_tokens: int
    output_tokens: int


def pick_device(pref: str = "auto") -> str:
    import torch
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(pref: str, device: str):
    import torch
    if pref != "auto":
        return getattr(torch, pref)
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    # CPU/MPS: bfloat16 でメモリを半減（Streamlit Community Cloud 等の小さい環境でも載るように）
    return torch.bfloat16 if device == "cpu" else torch.float32


# ---- アダプターの所在 -------------------------------------------------------
def resolve_adapter(adapter: str | None, adapters_dir: str) -> Path | None:
    """'latest' / 実験ID / パス のいずれかからアダプターディレクトリを返す。"""
    if not adapter:
        return None
    root = resolve(adapters_dir)
    if adapter == "latest":
        latest = root / "LATEST"
        if not latest.exists():
            return None
        adapter = latest.read_text(encoding="utf-8").strip()
    p = Path(adapter)
    if not p.is_absolute():
        p = root / adapter if (root / adapter).exists() else resolve(adapter)
    return p if (p / "adapter_config.json").exists() else None


def list_adapters(adapters_dir: str) -> list[dict[str, Any]]:
    """学習済みアダプター（実験）の一覧を新しい順で返す。"""
    root = resolve(adapters_dir)
    out = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir(), reverse=True):
        meta = d / "run_meta.json"
        if d.is_dir() and (d / "adapter_config.json").exists():
            info = read_json(meta) if meta.exists() else {}
            info.setdefault("run_id", d.name)
            info["path"] = str(d)
            out.append(info)
    return out


# ---- 生成モデル -------------------------------------------------------------
class SarashinaModel:
    """ベースモデル + 任意のLoRAアダプター。"""

    def __init__(self, base_model: str, adapter_path: str | Path | None = None,
                 device: str = "auto", dtype: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.base_model_name = base_model
        base_model = model_ref(base_model)
        self.device = pick_device(device)
        self.dtype = pick_dtype(dtype, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=self.dtype)
        self.base_revision = getattr(model.config, "_commit_hash", None) or "local"
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.adapter_meta: dict[str, Any] = {}
        if self.adapter_path:
            cfg = read_json(self.adapter_path / "adapter_config.json")
            trained_on = cfg.get("base_model_name_or_path")
            if trained_on and Path(trained_on).name != Path(self.base_model_name).name:
                raise ValueError(f"LoRAアダプターのベースモデル（{trained_on}）が"
                                 f"設定のベースモデル（{self.base_model_name}）と一致しません。")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(self.adapter_path))
            meta = self.adapter_path / "run_meta.json"
            self.adapter_meta = read_json(meta) if meta.exists() else {"run_id": self.adapter_path.name}
        self.model = model.to(self.device).eval()

    @property
    def has_adapter(self) -> bool:
        return self.adapter_path is not None

    @property
    def adapter_id(self) -> str:
        return self.adapter_meta.get("run_id", "") if self.has_adapter else ""

    def generate(self, messages: list[dict], gen_cfg: dict, max_new_tokens: int) -> GenerationOutput:
        import torch
        enc = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        n_in = int(enc["input_ids"].shape[-1])
        kwargs: dict[str, Any] = dict(
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(gen_cfg.get("do_sample", False)),
            repetition_penalty=float(gen_cfg.get("repetition_penalty", 1.0)),
            pad_token_id=self.tokenizer.pad_token_id,
        )
        torch.manual_seed(int(gen_cfg.get("seed", 42)))
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**enc, **kwargs)
        elapsed = time.perf_counter() - t0
        new_tokens = out[0, n_in:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return GenerationOutput(text=text, time_sec=round(elapsed, 3),
                                input_tokens=n_in, output_tokens=int(new_tokens.shape[-1]))

    def version_info(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model_name,
            "base_revision": self.base_revision,
            "adapter_id": self.adapter_id,
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
        }
