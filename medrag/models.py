"""学習前・学習後モデルの読み込みと生成。

要件「学習前後のLLMは同一のベースモデルとトークナイザーを使用し、LoRAアダプターの適用有無だけを変える」
を構造的に保証するため、ベースモデルを1回だけ読み込み、PEFT の disable_adapter() で
アダプターの ON/OFF を切り替える（2つ目のモデルをメモリに載せない）。
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import model_ref, resolve
from .versioning import read_json


@dataclass
class GenerationOutput:
    text: str
    time_sec: float
    input_tokens: int
    output_tokens: int


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(pref: str, device: str) -> torch.dtype:
    if pref != "auto":
        return getattr(torch, pref)
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
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


# ---- 比較用モデル -------------------------------------------------------------
class ComparisonModel:
    """1つのベースモデル + 任意のLoRAアダプター。use_lora で切り替えて生成する。"""

    def __init__(self, base_model: str, adapter_path: str | Path | None = None,
                 device: str = "auto", dtype: str = "auto"):
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
            self._check_adapter_base(self.adapter_path)
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(self.adapter_path))
            meta = self.adapter_path / "run_meta.json"
            self.adapter_meta = read_json(meta) if meta.exists() else {"run_id": self.adapter_path.name}
        self.model = model.to(self.device).eval()

    def _check_adapter_base(self, adapter_path: Path) -> None:
        cfg = read_json(adapter_path / "adapter_config.json")
        trained_on = cfg.get("base_model_name_or_path")
        if trained_on and Path(trained_on).name != Path(self.base_model_name).name:
            raise ValueError(
                f"LoRAアダプターのベースモデル（{trained_on}）が比較対象のベースモデル"
                f"（{self.base_model_name}）と一致しません。同一ベースでの比較が必須条件です。"
            )

    @property
    def has_adapter(self) -> bool:
        return self.adapter_path is not None

    @property
    def adapter_id(self) -> str:
        return self.adapter_meta.get("run_id", "") if self.has_adapter else ""

    def _ctx(self, use_lora: bool):
        if use_lora:
            if not self.has_adapter:
                raise RuntimeError("LoRAアダプターが読み込まれていません。先に学習を実行してください。")
            return contextlib.nullcontext()
        return self.model.disable_adapter() if self.has_adapter else contextlib.nullcontext()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    @torch.inference_mode()
    def generate(self, messages: list[dict], use_lora: bool, gen_cfg: dict) -> GenerationOutput:
        enc = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        n_in = int(enc["input_ids"].shape[-1])
        kwargs: dict[str, Any] = dict(
            max_new_tokens=int(gen_cfg.get("max_new_tokens", 512)),
            do_sample=bool(gen_cfg.get("do_sample", False)),
            repetition_penalty=float(gen_cfg.get("repetition_penalty", 1.0)),
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if kwargs["do_sample"]:
            kwargs.update(temperature=float(gen_cfg.get("temperature", 0.7)),
                          top_p=float(gen_cfg.get("top_p", 0.9)))
        torch.manual_seed(int(gen_cfg.get("seed", 42)))
        with self._ctx(use_lora):
            self._sync()
            t0 = time.perf_counter()
            out = self.model.generate(**enc, **kwargs)
            self._sync()
            elapsed = time.perf_counter() - t0
        new_tokens = out[0, n_in:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return GenerationOutput(text=text, time_sec=round(elapsed, 3),
                                input_tokens=n_in, output_tokens=int(new_tokens.shape[-1]))

    def _sync(self) -> None:
        if self.device == "cuda":
            torch.cuda.synchronize()

    def version_info(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model_name,
            "base_revision": self.base_revision,
            "adapter_id": self.adapter_id,
            "adapter_path": str(self.adapter_path) if self.adapter_path else "",
            "adapter_data_version": self.adapter_meta.get("data_version", ""),
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
        }
