"""食品ラベル画像 → 文字（Sarashina2.2-OCR / Tesseract）。

- GPU がある環境: Sarashina2.2-OCR（Sarashina2.2-3B-Instruct ベースの文書OCR）
- CPU 環境（Streamlit Community Cloud など）: Tesseract（日本語）。初回に高精度モデル
  tessdata_best をダウンロードして使い、取得できなければOS標準のモデルを使う。
どちらも結果は text.clean_ocr_text() でノイズを除去してから返す（文字化け対策）。
"""
from __future__ import annotations

import os
import urllib.request
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

from .config import resolve
from .text import clean_ocr_text

BEST_URL = "https://github.com/tesseract-ocr/tessdata_best/raw/main/{lang}.traineddata"


def pick_backend(pref: str) -> str:
    if pref != "auto":
        return pref
    try:
        import torch
        return "sarashina" if torch.cuda.is_available() else "tesseract"
    except Exception:  # noqa: BLE001
        return "tesseract"


# ---- Tesseract --------------------------------------------------------------
def _system_tessdata() -> Path | None:
    for d in ("/usr/share/tesseract-ocr/5/tessdata", "/usr/share/tesseract-ocr/4.00/tessdata",
              "/usr/share/tessdata", "/usr/local/share/tessdata"):
        if Path(d, "eng.traineddata").exists() or Path(d, "osd.traineddata").exists():
            return Path(d)
    return None


@lru_cache(maxsize=1)
def prepare_tessdata(tessdata_dir: str, use_best: bool) -> str | None:
    """tessdata_best の jpn / jpn_vert を用意し、そのディレクトリを返す（失敗時は None = 標準）。"""
    if not use_best:
        return None
    d = resolve(tessdata_dir)
    d.mkdir(parents=True, exist_ok=True)
    try:
        for lang in ("jpn", "jpn_vert", "eng"):
            p = d / f"{lang}.traineddata"
            if not p.exists() or p.stat().st_size < 1_000_000:
                urllib.request.urlretrieve(BEST_URL.format(lang=lang), p)
        sysd = _system_tessdata()  # osd（向き判定）は標準のものを共有
        if sysd and (sysd / "osd.traineddata").exists() and not (d / "osd.traineddata").exists():
            (d / "osd.traineddata").write_bytes((sysd / "osd.traineddata").read_bytes())
        return str(d)
    except Exception as e:  # noqa: BLE001
        print(f"[ocr] tessdata_best を取得できなかったため標準モデルを使います: {e}")
        return None


def _prep(image: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(image).convert("L")
    w, h = img.size
    if max(w, h) < 2000:  # 小さい文字は拡大した方が読める
        s = 2000 / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    img = ImageOps.autocontrast(img, cutoff=1).filter(ImageFilter.MedianFilter(3))
    return img


def read_tesseract(image: Image.Image, ocr_cfg: dict) -> str:
    import pytesseract
    tessdata = prepare_tessdata(ocr_cfg.get("tessdata_dir", "outputs/tessdata"),
                                bool(ocr_cfg.get("tesseract_best", True)))
    extra = f'--tessdata-dir "{tessdata}" ' if tessdata else ""
    img = _prep(image)
    best = ""
    for lang, psm in (("jpn+eng", 6), ("jpn+eng", 4), ("jpn_vert+jpn", 5)):
        try:
            txt = pytesseract.image_to_string(img, lang=lang, config=f"{extra}--oem 1 --psm {psm}")
        except pytesseract.TesseractError:
            continue
        txt = clean_ocr_text(txt)
        if "原材料" in txt:  # 欄名が読めたらそれを採用
            return txt
        if len(txt) > len(best):
            best = txt
    return best


# ---- Sarashina2.2-OCR ---------------------------------------------------------
@lru_cache(maxsize=1)
def _load_sarashina(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", dtype=torch.bfloat16,
                                                 trust_remote_code=True)
    return processor, model.eval()


def read_sarashina(image: Image.Image, ocr_cfg: dict) -> str:
    import torch
    processor, model = _load_sarashina(ocr_cfg["sarashina_model"])
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((1600, 1600))
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image", "image": image}]}],
        tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=2048, do_sample=False, repetition_penalty=1.2)
    text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return clean_ocr_text(text)


def read_label(image: Image.Image, ocr_cfg: dict) -> tuple[str, str]:
    """(文字, 使ったOCR名) を返す。"""
    if os.environ.get("FOODRAG_MOCK") == "1":
        return clean_ocr_text(resolve("data/eval/sample_label.txt").read_text(encoding="utf-8")), "mock"
    backend = pick_backend(ocr_cfg.get("backend", "auto"))
    if backend == "sarashina":
        return read_sarashina(image, ocr_cfg), "Sarashina2.2-OCR"
    return read_tesseract(image, ocr_cfg), "Tesseract"
