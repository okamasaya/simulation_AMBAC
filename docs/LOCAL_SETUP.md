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

## 動きを目で確かめるツール（モーションチェック）

クラウドでは画面がないため、`sims/field_gifs.py` で GIF を作って確かめている（`figures/field/gifs/`）。
ローカルでは、次のツールで、より手軽に・自由な角度から動きを見られる。

| ツール | 向いていること | 準備 |
|---|---|---|
| **MuJoCo のビューア**（おすすめ、最初に使う） | 画面で回転・拡大しながら、その場で動きを見る。一時停止・コマ送り、関節を手で動かして試す、接触力の表示もできる。Geant の対話的な表示に近い | `pip install mujoco` だけ。`python -m mujoco.viewer --mjcf=モデル.xml`、または Python から `mujoco.viewer.launch_passive(m, d)` |
| Rerun（`pip install rerun-sdk`） | シミュレーションの記録を、時間軸を動かしながら見返す。3D とグラフ（傾き・トルクなど）を並べて見られる | pip で入る。記録を送る数行のコードが必要 |
| MeshCat（`pip install meshcat`） | ブラウザで 3D を表示する軽いビューア | pip で入る |
| Blender | 見栄えのよいアニメーション（影・質感・カメラワーク）。発表や説明の資料向け | MuJoCo の記録（各物体の位置と向きの時系列）を読み込む変換スクリプトが必要 |
| NVIDIA Isaac Sim | ロボットの本格的なシミュレーションと描画 | 高性能な NVIDIA の GPU が必要。重い |

組み立てた機体のモデルは、次のようにファイルに書き出してビューアで開ける。

```python
from ambac.field.assembly import build, load
xml, joints = build(load("robots", "linked_track_arm"), load("stages", "steps"))
open("linked_track_on_steps.xml", "w").write(xml)
```

```bash
python -m mujoco.viewer --mjcf=linked_track_on_steps.xml
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
