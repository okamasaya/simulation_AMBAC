# 技術報告書（LaTeX）

- 完成版 PDF: [AMBAC_technical_report.pdf](AMBAC_technical_report.pdf)
- ソース: `main.tex`（本体）、`chapters/`（各章）、`tikz/`（模式図）、`captions.tex`（図のキャプション）、`references.tex`（参考文献、`tools/references.py` で生成）

## ビルド

TeX Live（LuaLaTeX、jlreq クラス、tikz-3dplot）が必要。

```bash
cd report
```

```bash
latexmk -lualatex main.tex
```

```bash
cp main.pdf AMBAC_technical_report.pdf
```

参考文献を更新したときは、リポジトリのルートで `python tools/references.py` を実行してから組み直す。
