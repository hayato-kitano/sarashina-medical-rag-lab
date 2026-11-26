# Sarashina Food Additive RAG（添加物チェック）

食品ラベル（原材料名）の写真をアップすると、がんとの関連が指摘されている添加物を見つけ、
参考文書の番号つき（[1] など）で説明する Streamlit アプリです。
構成は [sarashina-medical-rag-lab](https://github.com/hayato-kitano/sarashina-medical-rag-lab)
（`config.yaml` ＋ `medrag/` パッケージ ＋ `scripts/`）に合わせています。

⚠️ 一般的な情報提供が目的です。診断・治療の判断には使用できません。

公開版：https://sarashina-food-additive-checker.streamlit.app/

## 構成

| レイヤー | 技術 |
|---|---|
| 説明生成 | `sbintuitions/sarashina2.2-0.5b-instruct-v0.1`（＋ `models/adapters/LATEST` の LoRA があれば適用） |
| 文字読み取り | GPU：Sarashina2.2-OCR ／ CPU：Tesseract（日本語・tessdata_best を自動取得） |
| RAG | `intfloat/multilingual-e5-small` ＋ FAISS（文書は `data/rag_docs/*.md`、YAML front matter で出典・版・更新日を管理） |
| 添加物の照合 | 別名の最長一致 → OCR誤読の表記ゆれ一致 → e5 意味検索（文字の重なりも確認） |
| 根拠チェック | Sarashina の出力を文ごとに参考文書と照合し、一致した文だけ採用。途中で切れた文・根拠の無い文は除外 |
| 画面 | Streamlit（ローカル / Streamlit Community Cloud） |

```
streamlit_app.py          Streamlit アプリ（写真アップ → 判定結果・解説・根拠）
config.yaml               全設定（モデル・OCR・RAG・根拠チェック・学習）
foodrag/
  config.py  versioning.py   設定読み込み・版管理（medrag と同じ）
  models.py                  Sarashina＋LoRA の読み込みと生成（medrag/models.py ベース）
  rag.py                     文書読込・分割・e5・FAISS・版管理＋添加物照合（medrag/rag.py ベース）
  ocr.py                     Sarashina2.2-OCR / Tesseract
  label.py                   原材料名欄の抽出と添加物の分割
  pipeline.py                照合 → 参考文書 → 生成 → 根拠チェック
  text.py                    文字の正規化・OCRノイズ除去・Markdownエスケープ（文字化け対策）
  data_prep.py  safety.py    学習データ作成・注意書き
scripts/
  build_index.py             RAG インデックス作成
  prepare_data.py            LoRA 学習データ作成（data/processed/<data_version>/）
  train_lora.py              LoRA 学習（SFT / PEFT / TRL、medrag と同じ手順）
  analyze_image.py           画像1枚をコマンドラインで判定
data/rag_docs/             添加物 18 件＋解説 2 件（IARC分類・根拠・日本での扱い・出典URL）
models/adapters/           学習済み LoRA アダプター（LATEST が指すものを使用）
tests/                     モデル不要のテスト
```

## 文字化け対策（v2.0）

- 画面に出す文字列は Streamlit の標準部品（`st.badge` / `st.text` / `st.caption`）か、Markdown エスケープ（`foodrag/text.py: md_escape`）を通して表示し、HTML への埋め込みをやめました（OCRの誤読記号 `[` `*` `_` などで表示が崩れていた原因）。
- OCR 結果から制御文字・置換文字（U+FFFD）・ゼロ幅文字・記号だけの行を除去（`clean_ocr_text`）。
- 「ラベル上の表記」は読める文字列のときだけ表示し、添加物名は参考文書の正式名で表示します。

## 実行手順

```bash
pip install -r requirements.txt     # Tesseract 本体: apt install tesseract-ocr tesseract-ocr-jpn
streamlit run streamlit_app.py
python -m pytest tests              # モデル不要のテスト
```

### LoRA 学習（Google Colab T4 など）

```bash
pip install -r requirements-train.txt
python scripts/prepare_data.py --n 2000
python scripts/train_lora.py              # → models/adapters/<run_id>/ と LATEST を更新
```

学習したアダプター（`models/adapters/<run_id>/` と `LATEST`）をコミットすると、公開アプリが自動で読み込みます。

### Streamlit Community Cloud

- Branch：`food-additive-checker`　Main file path：`food-additive-checker/streamlit_app.py`　Python 3.12
- 依存関係：`food-additive-checker/requirements.txt`、APTパッケージ：ブランチのルートの `packages.txt`（Community Cloud はルートのものだけ読みます）

## ライセンス・注意

- Sarashina のモデルは各モデルカードのライセンスに従います。システムでは「Built with Sarashina」を表示しています。
- 参考文書は 2026年11月時点の公開情報を要約した検証用文書です。公開前に専門家の確認と出典の最新化を行ってください。
