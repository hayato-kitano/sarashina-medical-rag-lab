"""食品ラベル画像 → テキスト（Sarashina2.2-OCR / Sarashina2.2-Vision-3B）"""
from __future__ import annotations

from functools import lru_cache

from PIL import Image, ImageOps

from . import config

VISION_PROMPT = (
    "この食品ラベルの画像に書かれている文字を、表示どおりに書き起こしてください。"
    "特に「原材料名」の欄は省略せず、記号（/、、、（））も含めて正確に書き起こしてください。"
)

SAMPLE_LABEL = (config.DATA_DIR / "sample_label.txt")


def preprocess(image: Image.Image, max_side: int = 1600) -> Image.Image:
    """スマホ写真の回転補正と縮小。小さすぎる文字を潰さない程度に抑える。"""
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    scale = max_side / max(w, h)
    if scale < 1:
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image


@lru_cache(maxsize=1)
def _load():
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(config.VISION_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.VISION_MODEL,
        device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    return processor, model


def read_label_tesseract(image: Image.Image) -> str:
    """CPU環境向け：Tesseract（jpn）で書き起こす。"""
    import pytesseract
    from PIL import ImageFilter

    img = ImageOps.exif_transpose(image).convert("L")
    w, h = img.size
    if max(w, h) < 1800:  # 小さい文字は拡大した方が読める
        s = 1800 / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    img = ImageOps.autocontrast(img).filter(ImageFilter.SHARPEN)
    text = pytesseract.image_to_string(img, lang="jpn+eng", config="--oem 1 --psm 6")
    # Tesseractは日本語の文字間に空白を入れるので取り除く
    import re
    text = re.sub(r"(?<=[^\x00-\x7F]) (?=[^\x00-\x7F])", "", text)
    return text.strip()


def read_label(image: Image.Image) -> str:
    """画像から文字を書き起こす。"""
    if config.MOCK:
        return SAMPLE_LABEL.read_text(encoding="utf-8")
    if config.OCR_BACKEND == "tesseract":
        return read_label_tesseract(image)

    import torch

    processor, model = _load()
    image = preprocess(image)

    content = [{"type": "image", "image": image}]
    # OCR特化モデルは画像のみを入力する仕様。汎用VLMには指示文を付ける。
    if "ocr" not in config.VISION_MODEL.lower():
        content.append({"type": "text", "text": VISION_PROMPT})

    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            repetition_penalty=1.2,
            use_cache=True,
        )
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
