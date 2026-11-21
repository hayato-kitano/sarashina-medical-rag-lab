"""学習データ変換：ApolloCorpus-ja → 品質フィルタ → 分割 → 会話形式（データ版付き）。

例:
  python scripts/prepare_data.py --n-samples 5000 --max-source-rows 60000
  python scripts/prepare_data.py --source-file data/raw/apollo_sample.jsonl   # ローカルファイルから
  # 任意：権利確認後に補助データを追加
  python scripts/prepare_data.py --include-alpacare 2000 --i-confirmed-license
"""
import argparse
import json

import _bootstrap  # noqa: F401
from medrag.config import load_config
from medrag.data_prep import load_alpacare, load_apollo, load_extra_jsonl, prepare


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--n-samples", type=int, help="フィルタ後に使う件数（既定: config の data.n_samples）")
    ap.add_argument("--max-source-rows", type=int, default=60000,
                    help="ApolloCorpus-ja から無作為に読み込む件数（全件=0）")
    ap.add_argument("--source-file", help="ApolloCorpus-ja 形式のローカル JSONL")
    ap.add_argument("--include-alpacare", type=int, default=0, help="JP-AlpaCare から追加する件数（任意）")
    ap.add_argument("--extra-jsonl", help="追加データ JSONL（例: AnswerCarefully を変換したもの）")
    ap.add_argument("--extra-prompt-field", default="prompt")
    ap.add_argument("--extra-response-field", default="response")
    ap.add_argument("--extra-source-name", default="extra")
    ap.add_argument("--i-confirmed-license", action="store_true",
                    help="補助データの利用条件を確認済みであることを明示する")
    ap.add_argument("--tokenizer", help="チャットテンプレートのプレビューに使うトークナイザー（既定: ベースモデル）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["data"]["seed"])
    records = load_apollo(cfg["data"]["source_dataset"], args.source_file, args.max_source_rows or None, seed)
    if args.include_alpacare or args.extra_jsonl:
        if not args.i_confirmed_license:
            raise SystemExit("補助データを使う場合は利用条件を確認のうえ --i-confirmed-license を付けてください。")
        if args.include_alpacare:
            records += load_alpacare(args.include_alpacare, seed)
        if args.extra_jsonl:
            records += load_extra_jsonl(args.extra_jsonl, args.extra_prompt_field,
                                        args.extra_response_field, args.extra_source_name)

    manifest = prepare(cfg, records, args.n_samples, args.tokenizer or cfg["model"]["base_model"])
    print(json.dumps({k: v for k, v in manifest.items() if k != "filters"}, ensure_ascii=False, indent=2))
    print(f"\n保存先: {cfg['data']['processed_dir']}/{manifest['data_version']}/")
    print("review_selected.csv / rejected.csv を確認し、医学的に不正確な行のIDを "
          f"{cfg['data']['manual_exclusion_file']} に追記して再実行してください。")


if __name__ == "__main__":
    main()
