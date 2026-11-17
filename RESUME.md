# 再開用メモ

新しい環境でこのプロジェクトを再開するときは、まずこのファイルを読む。

## 現在の状態（第 1 段階 完了）

- ブランチ: `claude/lucid-mayer-06omnh`（すべてコミット・push 済み）
- 成果物
  - シミュレーション Sim01〜Sim10（`sims/`）、モデル（`src/ambac/`）、出力（`data/`, `figures/`）
  - ノート（`docs/00`〜`10`、`docs/references.md`）
  - 技術報告書 [87 ページ版](report/AMBAC_technical_report.pdf) と [詳細版 159 ページ](report/AMBAC_technical_report_detailed.pdf)（LaTeX ソースは `report/`）
- ストーリーと目的は [docs/00_purpose.md](docs/00_purpose.md)、全体像は [README.md](README.md)

## 環境の準備

### Python（シミュレーション）

```bash
pip install -r requirements.txt
```

### LaTeX（報告書の組版、Ubuntu の場合）

```bash
apt-get update
```

```bash
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends texlive-luatex texlive-lang-japanese texlive-latex-recommended texlive-latex-extra texlive-pictures texlive-science latexmk fonts-noto-cjk
```

## 動作確認（数分）

```bash
cd sims && python sim01_single_arm.py && python sim02_cyclic_gait.py
```

```bash
cd report && latexmk -lualatex main.tex && latexmk -lualatex main_detailed.tex
```

Sim01 で「腕振り 180° での最大姿勢変化: -16.04°」、報告書が 87 ページ・159 ページになれば正常。

## 実行の順序と依存関係

- `sim03_rw_vs_ambac.py` は `data/sim03_optimal_loop.json` を保存し、Sim04・Sim05 がそれを読む。
- `sim05b_allocation.py` は学習済み方策 `data/sim05b_policy.npy` を再利用する（再学習は `RETRAIN=1`、約 12 分）。
- `sims/detail_figures.py` は詳細版の追加図（`figures/detail/`）と `data/detail_summary.txt` を作る。
- 参考文献を変えたら `python tools/references.py`、表を作り直すなら `python tools/report_tables.py` を実行してから報告書を組む。

## 次の作業の候補（第 2 段階）

詳細は `docs/00_purpose.md` の「第 2 段階」と、詳細版の「第 2 段階の詳細計画」の章を参照。

1. 部品の組み合わせ（アセンブル）を入力に、第 1 段階の評価をまとめて実行して性能表を出す仕組み
2. 校正値の定義と、AMBAC ループの周回中の姿勢の波形を使った慣性パラメータの自己同定（詳細版の D14 で実現性を確認済み）
3. アクチュエータ（油圧・電動）の簡易モデル
4. 人の動き（モーションキャプチャー）のデータとの接続（ひねりの扱いに注意）
5. 動的な動作（踏み出し、勢いを使った段差の乗り越え）

## 作業上の注意

- 報告書の正本は `report/`（`docs/` の Markdown を更新しても報告書には自動で反映されない）。
- 作品由来の情報（第 42 話の台詞、ガンダムセンチュリー、大西さんの記事）は、ユーザー提供の情報に基づく。記事名など未照合の点は参考文献に注記している。
- 修正の記録は報告書の付録「数値計算の方法と検証」にある（Sim06 の関節制約の不具合など）。
