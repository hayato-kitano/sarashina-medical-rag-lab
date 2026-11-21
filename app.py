"""Sarashina Medical RAG Lab — 学習前・学習後 Sarashina の回答を左右比較する Streamlit アプリ。

起動: streamlit run app.py
"""
from __future__ import annotations

import difflib
import html
import json

import pandas as pd
import streamlit as st

from medrag.config import load_config, resolve
from medrag.evaluation import (SCORE_COLUMNS, aggregate, compare_runs, list_eval_runs, load_questions,
                               load_run, run_batch, save_scores)
from medrag.models import ComparisonModel, list_adapters, resolve_adapter
from medrag.pipeline import MODE_LABELS, mode_of, result_rows, run_comparison
from medrag.rag import (Retriever, build_index, docs_changed_since_index, index_exists, iter_documents,
                        save_uploaded)
from medrag.safety import DISCLAIMER
from medrag.versioning import read_json

st.set_page_config(page_title="Sarashina Medical RAG Lab", page_icon="🩺", layout="wide")
cfg = load_config()


# ---- リソース（キャッシュ）----------------------------------------------------
@st.cache_resource(show_spinner="モデルを読み込んでいます（初回はダウンロードに数分かかります）…")
def get_model(base: str, adapter_path: str | None) -> ComparisonModel:
    return ComparisonModel(base, adapter_path, cfg["model"]["device"], cfg["model"]["dtype"])


@st.cache_resource(show_spinner="RAGインデックスを読み込んでいます…")
def get_retriever(version_key: str) -> Retriever:  # version_key が変わると再読込
    return Retriever(cfg)


def ensure_index() -> Retriever | None:
    if not index_exists(cfg):
        if not list(iter_documents(resolve(cfg["rag"]["docs_dir"]))):
            return None
        with st.spinner("RAGインデックスを作成しています…"):
            build_index(cfg)
    man = read_json(resolve(cfg["rag"]["index_dir"]) / "manifest.json")
    return get_retriever(man["version"])


# ---- サイドバー -------------------------------------------------------------
st.sidebar.title("🩺 Sarashina Medical RAG Lab")
adapters = list_adapters(cfg["model"]["adapters_dir"])
default_adapter = resolve_adapter(cfg["model"]["adapter"], cfg["model"]["adapters_dir"])
options = [a["path"] for a in adapters]
labels = {a["path"]: f"{a['run_id']}（data: {a.get('data_version', '?')}）" for a in adapters}
if options:
    idx = options.index(str(default_adapter)) if str(default_adapter) in options else 0
    adapter_path = st.sidebar.selectbox("学習後モデル（LoRAアダプター）", options, index=idx,
                                        format_func=lambda p: labels[p])
else:
    adapter_path = None
    st.sidebar.warning("LoRAアダプターがまだありません。学習後は学習前と同じ回答になります。"
                       "README の手順で学習してください。")
use_rag = st.sidebar.toggle("RAG（医療文書検索）", value=False,
                            help="ON にすると、検索は1回だけ行い、同じ検索結果を学習前・学習後の両方に渡します。")
st.sidebar.caption(f"比較モード：左 **{MODE_LABELS[mode_of(False, use_rag)]}** ／ 右 "
                   f"**{MODE_LABELS[mode_of(True, use_rag)]}**")
with st.sidebar.expander("共通の生成条件（学習前・学習後で同一）"):
    st.json(cfg["generation"])
    st.text_area("システムプロンプト", cfg["system_prompt"], height=200, disabled=True)
st.sidebar.info(DISCLAIMER)

model = get_model(cfg["model"]["base_model"], adapter_path)
retriever = ensure_index()
which = ("base", "lora") if model.has_adapter else ("base",)

tab_cmp, tab_batch, tab_score, tab_docs, tab_ver = st.tabs(
    ["💬 左右比較", "📋 一括評価", "✍️ 採点・集計", "📚 RAG文書", "🧪 学習実験・バージョン"])


# ---- 共通表示部品 -------------------------------------------------------------
def render_answer(a, title: str) -> None:
    st.subheader(title)
    st.caption(MODE_LABELS[a.mode])
    if a.status == "emergency_guide":
        st.error(a.text)
    elif a.status == "abstained":
        st.warning(a.text)
    else:
        st.markdown(a.text or "（空の回答）")
    m1, m2, m3 = st.columns(3)
    m1.metric("生成時間", f"{a.time_sec:.2f} 秒")
    m2.metric("入力トークン", a.input_tokens)
    m3.metric("出力トークン", a.output_tokens)
    if a.safety_flags:
        st.warning(f"⚠️ 自動安全性チェック：{a.safety_flags} の疑い（評価者が確認してください）")


def render_refs(chunks: list[dict]) -> None:
    for i, c in enumerate(chunks, 1):
        ok = c["score"] >= float(cfg["rag"]["min_score"])
        head = f"[{i}] {c['title']}　score={c['score']:.3f}" + ("" if ok else "（閾値未満）")
        with st.expander(head):
            st.markdown(f"- 参照元: `{c['doc_path']}`" + (f"（p.{c['page']}）" if c.get("page") else ""))
            st.markdown(f"- URL: {c.get('source_url') or '（出典情報なし）'}")
            st.markdown(f"- 版: {c.get('version') or '-'}　／　更新日: {c.get('updated') or '-'}")
            st.markdown(f"- ライセンス・利用条件: {c.get('license') or '-'}")
            st.text(c["text"])


def diff_html(a: str, b: str) -> str:
    """文字単位の差分（削除=赤, 追加=緑）。"""
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            out.append(html.escape(a[i1:i2]))
        if op in ("delete", "replace"):
            out.append(f"<del style='background:#fdd'>{html.escape(a[i1:i2])}</del>")
        if op in ("insert", "replace"):
            out.append(f"<ins style='background:#dfd;text-decoration:none'>{html.escape(b[j1:j2])}</ins>")
    sim = sm.ratio()
    body = "".join(out).replace("\n", "<br>")
    return f"<p>類似度: <b>{sim:.1%}</b></p><div style='line-height:1.8'>{body}</div>"


def append_history(rows: list[dict]) -> None:
    p = resolve(cfg["evaluation"]["history_file"])
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def load_history() -> pd.DataFrame:
    p = resolve(cfg["evaluation"]["history_file"])
    if not p.exists():
        return pd.DataFrame()
    with open(p, encoding="utf-8") as f:
        return pd.DataFrame([json.loads(x) for x in f if x.strip()])


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")  # Excel で文字化けしないよう BOM 付き


# ---- 1) 左右比較 -----------------------------------------------------------------
with tab_cmp:
    st.header("学習前 vs 学習後 Sarashina")
    samples = ["高血圧の診断基準を教えてください。", "HbA1cとは何ですか？",
               "熱中症が疑われるときの応急処置は？", "インフルエンザで学校を休む期間は？",
               "架空の薬「メディリンX」の効果を教えてください。"]
    pick = st.selectbox("質問例", ["（自由入力）"] + samples)
    question = st.text_area("質問", value="" if pick == "（自由入力）" else pick, height=90,
                            placeholder="医療に関する一般的な質問を入力してください（個人情報は入力しないでください）")
    run = st.button("両モデルで回答生成", type="primary", disabled=not question.strip())
    if use_rag and retriever is None:
        st.error("RAG文書が登録されていません。「RAG文書」タブで登録してください。")
        run = False

    if run:
        with st.spinner("同一条件で両モデルの回答を生成しています…"):
            res = run_comparison(question, use_rag, model, retriever, cfg, which=which)
        st.session_state["last"] = res
        append_history(result_rows(res))

    res = st.session_state.get("last")
    if res:
        if res.emergency:
            st.error("🚑 緊急性が疑われる内容が含まれています。\n\n" + res.emergency)
        if res.insufficient_evidence:
            st.warning("参考文書から十分な根拠が見つからなかったため、回答を保留しました（根拠不足時の回答保留）。")
        left, right = st.columns(2)
        with left:
            render_answer(res.answers["base"], "学習前 Sarashina")
        with right:
            if "lora" in res.answers:
                render_answer(res.answers["lora"], "学習後 Sarashina（LoRA）")
            else:
                st.subheader("学習後 Sarashina（LoRA）")
                st.info("LoRAアダプターが未作成です。")
        if res.use_rag:
            st.markdown("#### 参照した根拠（両モデル共通）")
            if res.retrieved:
                render_refs(res.retrieved)
            else:
                st.caption("検索結果なし")
        if "lora" in res.answers and res.answers["base"].generated:
            with st.expander("🔍 回答差分（学習前 → 学習後）", expanded=True):
                st.markdown(diff_html(res.answers["base"].text, res.answers["lora"].text),
                            unsafe_allow_html=True)
        with st.expander("この回答のバージョン情報"):
            st.json(res.versions)

    st.divider()
    st.markdown("#### 回答履歴")
    hist = load_history()
    if hist.empty:
        st.caption("まだ履歴はありません。")
    else:
        st.dataframe(hist[["timestamp", "question", "mode_label", "answer", "time_sec", "output_tokens"]]
                     .iloc[::-1], width="stretch", height=250)
        st.download_button("回答履歴をCSVでダウンロード", csv_bytes(hist), "chat_history.csv", "text/csv")


# ---- 2) 一括評価 -----------------------------------------------------------------
with tab_batch:
    st.header("事前登録した評価質問の一括実行")
    qs = load_questions(cfg["evaluation"]["questions_file"])
    c1, c2 = st.columns([2, 1])
    cats = c1.multiselect("カテゴリ", sorted(qs["category"].unique()), default=sorted(qs["category"].unique()))
    limit = c2.number_input("最大件数", 1, len(qs), min(20, len(qs)))
    avail = ["A", "C"] + (["B", "D"] if model.has_adapter else [])
    modes = st.multiselect("比較モード（4種類）", ["A", "B", "C", "D"], default=sorted(avail),
                           format_func=lambda m: MODE_LABELS[m])
    sel = qs[qs["category"].isin(cats)].head(int(limit))
    st.caption(f"対象 {len(sel)} 問 × {len(modes)} モード")
    with st.expander("対象の質問を表示"):
        st.dataframe(sel, width="stretch")
    need_rag = any(m in ("C", "D") for m in modes)
    need_lora = any(m in ("B", "D") for m in modes)
    blocked = (need_rag and retriever is None) or (need_lora and not model.has_adapter) or not modes
    if st.button("一括実行", type="primary", disabled=blocked):
        bar = st.progress(0.0)
        out = run_batch(sel, modes, model, retriever, cfg, progress=lambda p, m: bar.progress(p, text=m))
        st.success(f"完了：{out.name}（「採点・集計」タブで採点できます）")
        df = load_run(out)
        st.dataframe(aggregate(df), width="stretch")
        st.download_button("結果CSVをダウンロード", csv_bytes(df), f"{out.name}.csv", "text/csv")


# ---- 3) 採点・集計 -----------------------------------------------------------------
with tab_score:
    st.header("回答品質の手動採点と指標別集計")
    runs = list_eval_runs(cfg)
    if not runs:
        st.info("一括評価の結果がまだありません。")
    else:
        rd = st.selectbox("評価実行", runs, format_func=lambda p: p.name)
        df = load_run(rd)
        st.caption("医学的正確性・安全性は文字列一致で採点せず、評価基準に従って評価者が入力します。"
                   "根拠整合性はRAG時のみ、根拠不足時の応答は該当する質問のみ採点してください。")
        view = st.radio("表示", ["質問ごとに左右比較して採点", "表形式で一括編集"], horizontal=True)
        if view == "質問ごとに左右比較して採点":
            qids = list(dict.fromkeys(df["question_id"]))
            qid = st.selectbox("質問", qids, format_func=lambda q: f"{q}: {df[df.question_id == q].question.iloc[0]}")
            sub = df[df.question_id == qid]
            st.markdown(f"**カテゴリ**：{sub.category.iloc[0]}")
            cols = st.columns(len(sub))
            for col, (i, r) in zip(cols, sub.iterrows()):
                with col:
                    st.markdown(f"**{r.mode_label}**　（{r.time_sec}秒 / 出力{r.output_tokens}tok）")
                    st.markdown(r.answer)
                    if r.auto_safety_flags:
                        st.warning(f"自動フラグ：{r.auto_safety_flags}")
                    if r.references:
                        with st.popover("根拠"):
                            st.text(r.references.replace(" | ", "\n"))
                    k = f"{rd.name}-{i}"
                    sc = [""] + [str(x) for x in range(1, 6)]
                    df.at[i, "score_accuracy"] = st.selectbox("正確性", sc, sc.index(str(r.score_accuracy)) if str(r.score_accuracy) in sc else 0, key=k + "a")
                    df.at[i, "score_relevance"] = st.selectbox("関連性", sc, sc.index(str(r.score_relevance)) if str(r.score_relevance) in sc else 0, key=k + "r")
                    if r.rag:
                        df.at[i, "score_grounding"] = st.selectbox("根拠整合性", sc, sc.index(str(r.score_grounding)) if str(r.score_grounding) in sc else 0, key=k + "g")
                    ok = ["", "OK", "NG"]
                    df.at[i, "score_abstention"] = st.selectbox("根拠不足時の応答", ok, ok.index(r.score_abstention) if r.score_abstention in ok else 0, key=k + "b")
                    df.at[i, "score_safety"] = st.selectbox("安全性", ok, ok.index(r.score_safety) if r.score_safety in ok else 0, key=k + "s")
                    df.at[i, "comment"] = st.text_input("コメント", r.comment, key=k + "c")
            if st.button("この質問の採点を保存", type="primary"):
                save_scores(rd, df)
                st.success("保存しました")
        else:
            show = ["question_id", "category", "question", "mode_label", "answer", "auto_safety_flags",
                    *SCORE_COLUMNS]
            edited = st.data_editor(
                df[show], width="stretch", height=420, disabled=show[:6],
                column_config={k: st.column_config.TextColumn(v) for k, v in SCORE_COLUMNS.items()})
            if st.button("採点を保存", type="primary"):
                for c in SCORE_COLUMNS:
                    df[c] = edited[c]
                save_scores(rd, df)
                st.success("保存しました")
        st.markdown("#### 指標別の集計")
        st.dataframe(aggregate(df), width="stretch")
        st.download_button("採点済み結果CSVをダウンロード", csv_bytes(df), f"{rd.name}_scored.csv", "text/csv")

        st.markdown("#### 学習実験間の結果比較")
        pick = st.multiselect("比較する評価実行", runs, default=runs[:2], format_func=lambda p: p.name)
        if pick:
            comp = compare_runs(cfg, pick)
            st.dataframe(comp, width="stretch")
            st.download_button("比較表CSVをダウンロード", csv_bytes(comp), "experiment_comparison.csv", "text/csv")


# ---- 4) RAG文書 ------------------------------------------------------------------
with tab_docs:
    st.header("RAG文書の管理")
    st.caption("使用許諾・利用条件を確認した文書だけを登録してください。参照元・URL・版・更新日は回答画面に表示されます。")
    if index_exists(cfg):
        man = read_json(resolve(cfg["rag"]["index_dir"]) / "manifest.json")
        st.markdown(f"**インデックス版**：`{man['version']}`　文書 {man['n_documents']} 件 / "
                    f"チャンク {man['n_chunks']} 件　Embedding：`{man['embedding_model']}`　作成：{man['created_at']}")
        st.dataframe(pd.DataFrame(man["documents"]), width="stretch")
        if docs_changed_since_index(cfg):
            st.warning("文書が追加・更新されています。インデックスを再作成してください。")
    with st.form("upload"):
        f = st.file_uploader("文書を追加（PDF / TXT / Markdown）", type=["pdf", "txt", "md"])
        c1, c2 = st.columns(2)
        title = c1.text_input("文書名 *")
        url = c2.text_input("参照元URL")
        version = c1.text_input("版")
        updated = c2.text_input("更新日（YYYY-MM-DD）")
        lic = st.text_input("ライセンス・利用条件 *", placeholder="例：〇〇の利用規約に基づき転載可（確認日 2026-11-21）")
        if st.form_submit_button("登録") and f:
            if not title or not lic:
                st.error("文書名とライセンス・利用条件は必須です。")
            else:
                p = save_uploaded(resolve(cfg["rag"]["docs_dir"]), f.name, f.getvalue(),
                                  {"title": title, "source_url": url, "version": version,
                                   "updated": updated, "license": lic})
                st.success(f"保存しました：{p.name}。インデックスを再作成してください。")
    if st.button("インデックスを再作成"):
        with st.spinner("Embeddingを計算しています…"):
            man = build_index(cfg)
        st.cache_resource.clear()
        st.success(f"作成しました：{man['version']}")
        st.rerun()


# ---- 5) 学習実験・バージョン --------------------------------------------------------
with tab_ver:
    st.header("学習実験とバージョン追跡")
    st.markdown("#### 現在の比較条件")
    st.json({**model.version_info(), "rag_index_version": retriever.version if retriever else "(なし)"})
    exp = resolve(cfg["training"]["output_dir"]).parent / "experiments.csv"
    if exp.exists():
        st.markdown("#### 学習実験の一覧")
        st.dataframe(pd.read_csv(exp), width="stretch")
    for a in adapters:
        with st.expander(f"{a['run_id']}　data={a.get('data_version', '?')}　"
                         f"eval_loss={a.get('final_eval_loss', '?')}"):
            lh = resolve(a["path"]) / "loss_history.csv"
            if lh.exists():
                ldf = pd.read_csv(lh).set_index("step")
                st.line_chart(ldf[["train_loss", "eval_loss"]].apply(pd.to_numeric, errors="coerce"))
            st.json({k: v for k, v in a.items() if k not in ("log_history",)})
    data_root = resolve(cfg["data"]["processed_dir"])
    mans = sorted(data_root.glob("*/manifest.json")) if data_root.exists() else []
    if mans:
        st.markdown("#### 学習データ版")
        for m in mans:
            d = read_json(m)
            with st.expander(f"{d['data_version']}　選択 {d['n_selected']} 件 / 元 {d['n_source_rows']} 件"):
                st.json({k: v for k, v in d.items() if k != "template_preview"})
