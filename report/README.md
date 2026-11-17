# 技術報告書（LaTeX）

| 版 | PDF | ページ数 |
|---|---|---|
| 技術報告書 | [AMBAC_technical_report.pdf](AMBAC_technical_report.pdf) | 87 |
| 技術報告書（詳細版） | [AMBAC_technical_report_detailed.pdf](AMBAC_technical_report_detailed.pdf) | 159 |

詳細版には、技術報告書の内容に加えて次のものを収めた。

- 基礎編（剛体力学、幾何力学入門、最適化・制御・学習）、記号表、用語集
- 各章の詳細解説（詳しい導出、数値例題、アルゴリズム、全結果の表、追加の解析、考察課題）
- 関連する研究と事例、第 2 段階の詳細計画（アセンブル・校正値・AMBAC ループによる自己同定の可能性）
- 詳しい導出、考察課題の解答の手引き、数値出力の記録（付録）

## ソースの構成

- `main.tex`：本体（技術報告書）。`main_detailed.tex` は `\DETAILED` を定義して `main.tex` を読み込み、詳細版を組む
- `chapters/`：両方の版に共通の章
- `detail/`：詳細版だけに入る章・節（`\Detail{...}` で挿入）。`detail/tables/` は `tools/report_tables.py` で生成した表
- `tikz/`：模式図、`captions.tex`：図のキャプション、`references.tex`：参考文献（`tools/references.py` で生成）
- 詳細版の追加図は `../figures/detail/`（`sims/detail_figures.py` で生成）

## ビルド

TeX Live（LuaLaTeX、jlreq、tikz-3dplot、algorithmicx、tcolorbox）が必要。

```bash
cd report
```

```bash
latexmk -lualatex main.tex
```

```bash
latexmk -lualatex main_detailed.tex
```

参考文献や表を更新したときは、リポジトリのルートで次を実行してから組み直す。

```bash
python tools/references.py
```

```bash
python tools/report_tables.py
```
