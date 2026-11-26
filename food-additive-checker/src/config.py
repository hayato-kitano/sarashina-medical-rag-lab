"""アプリ全体の設定。環境変数で上書きできる。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
KB_PATH = Path(os.getenv("KB_PATH", DATA_DIR / "additives_kb.jsonl"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "index"))

# 1 にするとモデルを読み込まずに動作確認できる（GPU不要）
MOCK = os.getenv("SARASHINA_MOCK", "0") == "1"

def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


HAS_GPU = _has_gpu()

# --- 画像処理 ---
# "sarashina" : Sarashina2.2-OCR / Vision（GPU必須）
# "tesseract" : Tesseract日本語OCR（Streamlit Community Cloud などCPU環境向け）
OCR_BACKEND = os.getenv("OCR_BACKEND", "sarashina" if HAS_GPU else "tesseract")

# --- 画像処理（Sarashinaベースの視覚言語モデル） ---
# sarashina2.2-ocr   : 文書OCR特化（推奨。食品ラベルの書き起こし向き）
# sarashina2.2-vision-3b : 汎用VLM（指示文で「原材料名だけ抜き出して」等が可能）
VISION_MODEL = os.getenv("VISION_MODEL", "sbintuitions/sarashina2.2-ocr")

# --- 回答生成（Sarashina2.2-0.5B-Instruct + LoRA） ---
LLM_BASE = os.getenv("LLM_BASE", "sbintuitions/sarashina2.2-0.5b-instruct-v0.1")
LORA_ADAPTER = os.getenv("LORA_ADAPTER", str(ROOT / "outputs" / "sarashina-additive-lora"))

# --- RAG 埋め込み（Sarashina-Embedding）---
# "sarashina" : sbintuitions/sarashina-embedding-v1-1b を使う
# "bm25"      : 文字n-gram BM25（モデル不要の軽量版）
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "sarashina" if HAS_GPU else "bm25")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sbintuitions/sarashina-embedding-v1-1b")
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.80"))
BM25_THRESHOLD = float(os.getenv("BM25_THRESHOLD", "8.0"))

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "700"))

# 回答生成モード
# "full"    : 添加物ごとの詳しい説明をSarashinaが生成（GPU向け）
# "summary" : Sarashinaは短い要約だけを生成し、詳細は資料から表示（CPU向け・既定）
# "off"     : LLMを使わず資料から説明を組み立てる
LLM_MODE = os.getenv("LLM_MODE", "full" if HAS_GPU else "summary")
SUMMARY_MAX_NEW_TOKENS = int(os.getenv("SUMMARY_MAX_NEW_TOKENS", "220"))
