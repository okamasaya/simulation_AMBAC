# 番外編 01：アルミ棒 + iPhone 2 台のデータ

解析は `sims/bangai01_bar_imu.py`。記事は `articles/ambac-bangai-bar-iphone-imu.md`、経緯は `docs/13_error_evaluation_kinfit.md` の 13.7 節。

## フォルダの構成

```
data/bangai01/
  synthetic_pendulum/      合成データ（振り子）。phyphox の書き出しと同じ形。正解は truth.json
  synthetic_rotation/      合成データ（水平に回す）
  <実測の名前>/             実測は 1 回の記録ごとにフォルダを作る（例：2026-12-05_pendulum_A1）
    config.json            解析の設定（下を参照。synthetic_*/config.json をコピーして直す）
    phone1/                1 台目の phyphox の書き出し（zip を展開したもの。zip のままでもよい：phone1.zip）
    phone2/                2 台目
  result_<名前>.json       解析の結果（自動で書かれる）
  toys.json, mujoco_check.json
```

## 記録の手順（phyphox）

1. 2 台とも phyphox の「リモートアクセス」をオンにし、PC のブラウザから開始・停止する（同じ Wi-Fi）。
2. 実験 A（振り子）は「ジャイロスコープ」、実験 B（回転）はジャイロと加速度の両方を記録する（「加速度（g を含む）」を使う。「g を除いた加速度」は iPhone の内部の推定を通るので、生の値ではない）。
3. **はじめの 3 秒は、棒を完全に静止させておく。** この区間から、ジャイロのバイアス・雑音の大きさ・雑音の時間相関を推定する（`static` の設定）。
4. 振り子は 10° くらい引き上げて、1 秒ほど止めてから静かに放す。20 周期以上記録する。
5. 書き出しは「CSV (Comma, decimal point)」を推奨（セミコロン・タブ区切り、小数点のカンマにも対応している）。`meta/time.csv`（記録の開始時刻）も残しておく。2 台の時刻の大まかな合わせに使う。
6. **最初の実測では、読み込んだ列が正しいかを必ず確かめる**（列名はゆるく照合している。`python -c "import sys; sys.path.insert(0,'sims'); import bangai01_bar_imu as B; p=B.load_phone('data/bangai01/<名前>/phone1'); print(p['gyro'][:3], p['t_gyro'][:3])"`）。

## config.json

| キー | 意味 |
|---|---|
| `experiment` | `"pendulum"`（実験 A）または `"rotation"`（実験 B） |
| `phones` | 2 台のフォルダ（または zip）の名前。1 台目の時計を基準にする |
| `gyro_axis` | 回転の軸に当たるジャイロの軸（画面を振り子の面に平行に貼るなら `z`） |
| `static` | 静止区間 [始め, 終わり]（各 iPhone の時刻、秒） |
| `skip_cycles` | 放してから当てはめを始めるまでの周期の数（放す瞬間の手の影響を避ける） |
| `seq_cycles` | 逐次更新の 1 区間の周期の数 |
| `phone2_sign` | 2 台目の向き（+1 / −1）。分かっていれば書く。`null` なら相関から決める |
| `rotation.accel_axis` / `accel_sign` | 棒に沿った加速度の軸と、回転軸から外向きを正にする符号 |
| `rotation.process_q` | ω の滑らかさの事前情報の強さ（既定 1.5 rad/s^2.5。系統誤差で 3 倍・1/3 倍に振る） |
| `bom` | 部品表。すべて [値, 標準偏差]、SI 単位。`r` は支点（回転軸）から iPhone の中心まで |

部品表で一番効くのは **iPhone の重心の位置**（`com_offset`、既定 0 ± 3 mm で T0 に 0.12%）。iPhone を定規の縁に載せて釣り合う位置を測れば、予測の誤差を数分の 1 にできる。

## 実行

```
python sims/bangai01_bar_imu.py analyze data/bangai01/<名前>/config.json
```

図は `figures/bangai/bangai01_<名前>.png`、数値は `data/bangai01/result_<名前>.json`。
