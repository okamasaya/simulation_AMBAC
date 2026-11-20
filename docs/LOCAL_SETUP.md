# ローカル（デスクトップ）環境の準備

シミュレーション場（MuJoCo のビューア、Blender など）は画面が必要なので、デスクトップで開発する。
報告書の組版やまとまった計算はクラウド（Claude Code の環境「simulation_AMBAC」）に任せ、両者は GitHub の `main` でつなぐ。

## 役割分担

| 作業 | ローカル（デスクトップ） | クラウド |
|---|---|---|
| Sim01〜10 の実行・修正 | ○ | ○ |
| シミュレーション場（MuJoCo のビューア）、Blender | ◎（画面が必要） | × |
| DIY_AutoML との連携、Excel 出力、ダッシュボード | ◎（同じ PC でつなげる） | △ |
| 報告書（LaTeX）の組版 | △（TeX Live の導入が重い） | ◎（自動で準備される） |

## 必要なソフト

| ソフト | 必須かどうか | 用途 | 備考 |
|---|---|---|---|
| Git | 必須 | GitHub との同期 | |
| Python 3.11 以上 | 必須 | シミュレーション全般 | 3.13 で動作確認済み |
| エディタ（VS Code など） | 推奨 | 開発 | |
| PDF ビューア | 推奨 | 報告書を読む | |
| Blender（無料） | 任意 | 動きの可視化、人の動きのデータとの接続 | シミュレーション場の段階 6（docs/11） |
| TeX Live または MiKTeX | 任意 | 報告書をローカルで組む場合のみ | LuaLaTeX と日本語用の jlreq クラスが必要。数 GB あるので、通常はクラウドに任せる |
| Excel | 任意 | 性能表・サマリーを見る | DIY_AutoML の出力と共通 |

GPU は不要（すべて CPU で動く）。

## Python のライブラリ

| 区分 | ライブラリ | ファイル | 備考 |
|---|---|---|---|
| 本プロジェクトに必須 | numpy, scipy, matplotlib | `requirements.txt` | 確認済みの版は `requirements-lock.txt` |
| シミュレーション場 | mujoco | `requirements-field.txt` | 4.0.1 で動作確認済み。GPU 不要 |
| DIY_AutoML との連携（既に準備済みのもの） | pandas、scikit-learn などの機械学習系、Dash・Plotly、Graphic Walker（pygwalker）、openpyxl（Excel） | — | 本プロジェクトのコードはこれらを必要としない。連携は CSV・JSON で行う |
| 任意 | jupyter | — | `notebooks/` で対話的に試す場合 |

## 準備の手順（Windows の例）

DIY_AutoML とはライブラリの版が衝突しないよう、**このプロジェクト専用の仮想環境**を作る。

```bash
git clone https://github.com/okamasaya/simulation_AMBAC.git
```

```bash
cd simulation_AMBAC
```

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements-field.txt
```

PowerShell で仮想環境を有効化できない場合は、一度だけ次を実行する。

```bash
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 動作確認

```bash
cd sims
```

```bash
python sim01_single_arm.py
```

「腕振り 180° での最大姿勢変化: -16.04°」と表示されれば正常。

MuJoCo のビューア（画面で動きを見る・操作する）は次で起動できる。

```bash
python -m mujoco.viewer
```

## Windows での注意

- **文字コード**：シミュレーションは日本語や記号（²、≈ など）を出力する。出力をファイルに保存したときに文字化けやエラーが出る場合は、環境変数 `PYTHONUTF8=1` を設定する（PowerShell なら `$env:PYTHONUTF8=1`）。
- **実行時間**：多くは数秒〜数十秒。`sim05a_learn_ambac.py` は約 5 分、`sim05b_allocation.py` の再学習（`RETRAIN=1`）は約 12 分、`sim10_steps.py` は約 3 分。
- **実行の順序**：`sim03_rw_vs_ambac.py` が保存するファイルを Sim04・Sim05 が読むので、初回は Sim03 を先に実行する（`RESUME.md` 参照）。

## クラウドとの行き来

- ローカルで作業したら、`main` にコミット・push する。
- クラウドで再開するときは、Claude Code で環境「simulation_AMBAC」を選んで新しいセッションを始め、「RESUME.md を読んで」と伝える。
