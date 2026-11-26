"""Sarashina Food Additive RAG — 食品ラベルの写真から、がんとの関連が指摘されている添加物を根拠付きで説明する。

構成は sarashina-medical-rag-lab（app.py / medrag / config.yaml）に準拠。
起動: streamlit run streamlit_app.py

文字化け対策：画面に出す文字列は、すべて Streamlit の標準部品（st.text / st.badge / st.caption 等）か、
Markdown エスケープ（foodrag.text.md_escape）を通してから表示する。HTML への埋め込みはしない。
"""
from __future__ import annotations

import hashlib
import io

import streamlit as st
from PIL import Image, UnidentifiedImageError

from foodrag import __version__
from foodrag.config import load_config, resolve
from foodrag.models import SarashinaModel, list_adapters, resolve_adapter
from foodrag.ocr import pick_backend, read_label
from foodrag.pipeline import RISK_LABEL, analyze_text
from foodrag.rag import Retriever, build_index, docs_changed_since_index, load_catalog
from foodrag.safety import DISCLAIMER
from foodrag.text import clean_ocr_text, md_escape
from foodrag.versioning import read_json

st.set_page_config(page_title="添加物チェック｜Sarashina Food Additive RAG", page_icon="🔍", layout="centered")
cfg = load_config()

BADGE = {"high": ("注意度 高", "red"), "medium": ("注意度 中", "orange"), "low": ("注意度 低", "green")}


# ---- リソース（キャッシュ）----------------------------------------------------
@st.cache_resource(show_spinner="Sarashina を読み込んでいます（初回はダウンロードに1〜2分かかります）…")
def get_model(base: str, adapter_path: str | None) -> SarashinaModel | None:
    try:
        return SarashinaModel(base, adapter_path, cfg["model"]["device"], cfg["model"]["dtype"])
    except Exception as e:  # noqa: BLE001  モデルが使えなくても資料ベースで動かす
        st.session_state["model_error"] = f"{type(e).__name__}: {e}"
        return None


@st.cache_resource(show_spinner="添加物の参考文書（RAGインデックス）を読み込んでいます…")
def get_retriever(version_key: str) -> Retriever:
    return Retriever(cfg)


@st.cache_resource
def get_catalog(version_key: str) -> dict:
    return load_catalog(cfg)


def ensure_index() -> Retriever | None:
    try:
        if docs_changed_since_index(cfg):
            with st.spinner("RAGインデックスを作成しています…"):
                build_index(cfg)
        man = read_json(resolve(cfg["rag"]["index_dir"]) / "manifest.json")
        return get_retriever(man["version"])
    except Exception as e:  # noqa: BLE001
        st.session_state["rag_error"] = f"{type(e).__name__}: {e}"
        return None


@st.cache_data(show_spinner=False, max_entries=30)
def ocr_cached(data: bytes, ocr_key: str) -> tuple[str, str]:
    return read_label(Image.open(io.BytesIO(data)), cfg["ocr"])


# ---- サイドバー -------------------------------------------------------------
st.sidebar.title("🔍 添加物チェック")
st.sidebar.caption("Sarashina Food Additive RAG　v" + __version__)
adapters = list_adapters(cfg["model"]["adapters_dir"])
default_adapter = resolve_adapter(cfg["model"]["adapter"], cfg["model"]["adapters_dir"])
if adapters:
    options = [a["path"] for a in adapters]
    idx = options.index(str(default_adapter)) if str(default_adapter) in options else 0
    adapter_path = st.sidebar.selectbox("LoRAアダプター", options, index=idx,
                                        format_func=lambda p: next(a["run_id"] for a in adapters if a["path"] == p))
else:
    adapter_path = None
    st.sidebar.info("LoRAアダプターは未学習です（ベースモデルで説明を生成）。README の手順で学習できます。")
st.sidebar.markdown(
    "- 文字読み取り：" + ("Sarashina2.2-OCR" if pick_backend(cfg["ocr"]["backend"]) == "sarashina" else "Tesseract（日本語）")
    + "\n- 説明生成：" + md_escape(cfg["model"]["base_model"].split("/")[-1])
    + "\n- RAG：" + md_escape(cfg["rag"]["embedding_model"].split("/")[-1]) + " ＋ FAISS"
)
st.sidebar.warning(DISCLAIMER)

model = get_model(cfg["model"]["base_model"], adapter_path)
retriever = ensure_index()
man_version = retriever.version if retriever else "no-index"
catalog = get_catalog(man_version)


# ---- 画面 ---------------------------------------------------------------------
st.title("🔍 添加物チェック")
st.caption("食品ラベル（原材料名）の写真から、がんとの関連が指摘されている添加物を、根拠（出典）つきで説明します。")

# 機能1：写真をアップ
tab_up, tab_cam = st.tabs(["📁 写真をアップ", "📷 カメラで撮る"])
with tab_up:
    up = st.file_uploader("原材料名が写った写真", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
with tab_cam:
    cam = st.camera_input("原材料名を撮影", label_visibility="collapsed")
photo = up or cam

if photo is None:
    st.info("原材料名の欄が大きく、明るく写るように撮影すると読み取り精度が上がります。")
    st.stop()

data = photo.getvalue()
try:
    Image.open(io.BytesIO(data)).verify()
except (UnidentifiedImageError, OSError):
    st.error("画像ファイルを読み込めませんでした。JPEG または PNG の写真を選んでください。")
    st.stop()
st.image(data, width="stretch")

photo_id = hashlib.sha1(data).hexdigest()[:16]
if st.session_state.get("photo_id") != photo_id:
    with st.spinner("ラベルの文字を読み取っています…"):
        text, engine = ocr_cached(data, cfg["ocr"]["backend"] + str(cfg["ocr"].get("tesseract_best")))
    st.session_state.update(photo_id=photo_id, ocr_text=text, ocr_engine=engine)

with st.expander("読み取った文字を確認・修正する"):
    st.caption("読み取りに誤りがあれば修正して「再判定」を押してください。")
    edited = st.text_area("読み取り結果", st.session_state["ocr_text"], height=180, label_visibility="collapsed")
    if st.button("再判定"):
        st.session_state["ocr_text"] = clean_ocr_text(edited)
        st.session_state["ocr_engine"] += "（手動修正）"

text = st.session_state["ocr_text"]
if not text.strip():
    st.warning("文字が読み取れませんでした。原材料名の部分を明るく・大きく撮り直してください。")
    st.stop()

key = hashlib.sha1((text + man_version + str(adapter_path) + __version__).encode()).hexdigest()
if st.session_state.get("result_key") != key:
    with st.spinner("添加物を調べ、Sarashina が根拠付きの説明を作成しています…"):
        st.session_state["result"] = analyze_text(text, st.session_state["ocr_engine"], model, retriever,
                                                  catalog, cfg)
    st.session_state["result_key"] = key
r = st.session_state["result"]

# 機能2：危険な添加物の説明
st.subheader("判定結果")
if r.findings:
    cols = st.columns(3)
    for col, lv in zip(cols, ("high", "medium", "low")):
        col.metric(BADGE[lv][0], f"{sum(f.risk_level == lv for f in r.findings)}件")
else:
    st.success("がんとの関連が指摘されている添加物は見つかりませんでした。")

st.markdown("**Sarashina による解説**" if r.answer_source.startswith("sarashina") else "**解説**（参考文書から作成）")
with st.container(border=True):
    for line in r.answer.splitlines():
        if line.strip():
            st.markdown(md_escape(line))
st.caption(f"根拠チェック：解説文の {round(r.coverage * 100)}% が参考文書の内容と一致"
           + ("　／　Sarashina の出力に根拠を確認できる文が無かったため、参考文書から作成しました"
              if r.generation_mode != "off" and r.answer_source == "資料" else ""))
if r.raw_output:
    with st.expander("Sarashina の出力（そのまま）"):
        st.caption("参考文書と内容が一致した文だけを上の解説に採用しています。")
        st.text(r.raw_output)
if r.error:
    st.caption("説明生成でエラーが発生したため、参考文書から作成しました：" + md_escape(r.error[:200]))

if r.findings:
    st.subheader("見つかった添加物")
    for f in r.findings:
        with st.container(border=True):
            label, color = BADGE.get(f.risk_level, ("-", "gray"))
            st.badge(label, color=color)
            st.markdown(f"**{md_escape(f.name)}**　\\[{f.ref}\\]")
            if f.label_text and f.label_text != f.name:
                st.caption("ラベル上の表記：" + md_escape(f.label_text))
            st.caption(f"IARC分類：{md_escape(f.iarc_group)}　／　用途：{md_escape(f.category)}")
            st.markdown(md_escape(f.summary))

st.subheader("根拠（参考文書）")
for ref in r.references:
    with st.expander(md_escape(f"[{ref['n']}] {ref['title']}")):
        st.text(ref["text"])
        for s in ref.get("sources") or []:
            st.markdown(f"- [{md_escape(s.get('title', ''))}]({s.get('url', '')})")
        st.caption(f"版：{ref.get('version') or '-'}　／　更新日：{ref.get('updated') or '-'}　／　"
                   f"{md_escape(ref.get('license') or '')}")

with st.expander("読み取り・処理の詳細"):
    st.caption(f"文字読み取り：{r.ocr_engine}　／　生成モード：{r.generation_mode}　／　生成時間：{r.time_sec:.1f}秒")
    st.text("原材料名欄：" + (r.ingredients_section or "（見つかりませんでした）"))
    st.text("添加物：" + ("、".join(r.additives) or "（なし）"))
    st.json(r.versions)
    for k in ("model_error", "rag_error"):
        if st.session_state.get(k):
            st.caption(md_escape(st.session_state[k]))

st.info(DISCLAIMER)
st.caption("Built with Sarashina（SB Intuitions）")
