"""添加物チェック（Streamlit版）

  streamlit run streamlit_app.py
"""
from __future__ import annotations

import html
import re

import streamlit as st
from PIL import Image

from src import config
from src.pipeline import analyze_text
from src.vision import read_label

st.set_page_config(page_title="添加物チェック", page_icon="🔍", layout="centered")

st.markdown(
    """
<style>
.block-container{max-width:760px;padding-top:2rem}
.badge{display:inline-block;font-size:12px;font-weight:700;padding:2px 10px;border-radius:999px;margin-right:6px}
.high{color:#c0392b;background:#fbeae8}.medium{color:#b9770e;background:#fdf3e1}.low{color:#4a7c59;background:#eaf3ec}
.cite{font-size:11px;font-weight:700;color:#1f4e5f;background:#e6f0f2;border-radius:4px;padding:0 5px;margin-left:2px}
.summary{background:#f7f9f9;border-left:4px solid #1f4e5f;padding:12px 16px;border-radius:6px;line-height:1.8}
.small{color:#6b6a66;font-size:13px}
</style>
""",
    unsafe_allow_html=True,
)

LEVEL = {"high": "注意度 高", "medium": "注意度 中", "low": "注意度 低"}


def cite_html(text: str) -> str:
    t = html.escape(text)
    return re.sub(r"\[(D\d+)\]", r'<span class="cite">\1</span>', t).replace("\n", "<br>")


@st.cache_data(show_spinner=False, max_entries=50)
def ocr_cached(data: bytes) -> str:
    import io
    return read_label(Image.open(io.BytesIO(data)))


@st.cache_data(show_spinner=False, max_entries=50)
def analyze_cached(text: str) -> dict:
    return analyze_text(text)


# ---------------------------------------------------------------- 画面
st.title("🔍 添加物チェック")
st.caption("食品ラベル（原材料名）の写真から、がんとの関連が指摘されている添加物を根拠つきで説明します。")

# 機能1：写真をアップ
tab_up, tab_cam = st.tabs(["📁 写真をアップ", "📷 カメラで撮る"])
with tab_up:
    up = st.file_uploader("原材料名が写った写真", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
with tab_cam:
    cam = st.camera_input("原材料名を撮影", label_visibility="collapsed")
photo = up or cam

if photo is not None:
    data = photo.getvalue()
    st.image(data, width="stretch")
    if st.session_state.get("photo_id") != hash(data):
        st.session_state.photo_id = hash(data)
        with st.spinner("ラベルの文字を読み取っています…"):
            try:
                st.session_state.ocr_text = ocr_cached(data)
            except Exception as e:
                st.session_state.ocr_text = ""
                st.error(f"文字を読み取れませんでした：{e}")

if st.session_state.get("ocr_text") is not None and photo is not None:
    with st.expander("読み取った文字を確認・修正する", expanded=False):
        st.caption("読み取りに誤りがあれば修正して「再判定」を押してください。")
        edited = st.text_area("読み取り結果", st.session_state.ocr_text, height=160, label_visibility="collapsed")
        if st.button("再判定"):
            st.session_state.ocr_text = edited

    text = st.session_state.ocr_text
    if not text.strip():
        st.warning("文字が読み取れませんでした。原材料名の部分を明るく・大きく撮り直してください。")
        st.stop()

    with st.spinner("添加物を調べ、Sarashinaが説明を作成しています…（初回はモデルの読み込みに1〜2分かかります）"):
        r = analyze_cached(text)

    # 機能2：危険な添加物の説明
    st.subheader("判定結果")
    if r["findings"]:
        n = {lv: sum(f["risk_level"] == lv for f in r["findings"]) for lv in LEVEL}
        c1, c2, c3 = st.columns(3)
        c1.metric("注意度 高", f"{n['high']}件")
        c2.metric("注意度 中", f"{n['medium']}件")
        c3.metric("注意度 低", f"{n['low']}件")
    else:
        st.success("がんとの関連が指摘されている添加物は見つかりませんでした。")

    st.markdown("**Sarashinaによる解説**" if not r["fallback_used"] else "**解説**（資料から作成）")
    st.markdown(f'<div class="summary">{cite_html(r["explanation"])}</div>', unsafe_allow_html=True)
    v = r["verification"]
    st.markdown(
        f'<p class="small">根拠チェック：解説文の{round(v["coverage"] * 100)}%が出典の内容と一致'
        + ("。モデル出力の裏付けが弱かったため、資料から作成した解説を表示しています。" if r["fallback_used"] and r["llm_mode"] != "off" else "")
        + "</p>",
        unsafe_allow_html=True,
    )

    if r.get("llm_raw"):
        with st.expander("Sarashinaの出力（そのまま）を見る"):
            st.caption("根拠（出典の内容）と一致した文だけを上の解説に採用しています。")
            st.text(r["llm_raw"])
    if r.get("llm_error"):
        st.caption(f"説明生成モデルでエラーが発生したため、資料から解説を作成しました：{r['llm_error'][:200]}")

    if r["findings"]:
        st.subheader("見つかった添加物")
        for f in r["findings"]:
            label = f["name"] if f["label_text"] == f["name"] else f"{f['label_text']} → {f['name']}"
            st.markdown(
                f'<span class="badge {f["risk_level"]}">{LEVEL[f["risk_level"]]}</span>'
                f'<b>{html.escape(label)}</b><span class="cite">{f["ref"]}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f'<p class="small">IARC分類：{html.escape(f["iarc_group"])}｜{html.escape(f["category"])}</p>', unsafe_allow_html=True)
            st.write(f["summary"])
            with st.expander("詳しく見る"):
                st.markdown(f"**根拠**　{f['evidence']}")
                st.markdown(f"**日本での扱い**　{f['japan_status']}")
                st.markdown(f"**アドバイス**　{f['advice']}")
            st.divider()

    st.subheader("根拠（出典）")
    for k, s in r["sources"].items():
        with st.expander(f"{k}　{s['name']}"):
            st.write(s["evidence"])
            for x in s["sources"]:
                st.markdown(f"- [{x['title']}]({x['url']})")

    st.info(r["disclaimer"])

st.markdown("---")
st.caption(
    f"Built with Sarashina（SB Intuitions）｜説明生成：{config.LLM_BASE.split('/')[-1]}"
    f"｜文字読み取り：{'Sarashina2.2-OCR' if config.OCR_BACKEND == 'sarashina' else 'Tesseract'}"
)
