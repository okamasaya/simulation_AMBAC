"""Field07: アクチュエータの簡易モデル — 荷物運びに必要なモーターの定格（シミュレーション場の段階 7）.

機体: tank_arm（戦車型台車＋2 関節の腕＋荷物 10 kg）。部品表の actuator に、
  ・肩のトルクの上限（N·m）  ・全関節の速度の上限（目標角の変化の速さ、°/s）
を与え、荷物運びの安全な最短時間（傾き 1° 未満・ずれ 5 mm 未満・腕が目標に 1° 以内で届く）を求める。
時間は、腕が目標の 1° 以内に入って以後とどまる実際の時刻（完了時間）で比べる（速度の上限があると指令どおりには動けないため）。
地球（練習場）と月で、必要な定格がどう変わるかを見る（肘のトルクの上限は肩の 0.6 倍）。
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import copy
import json

import matplotlib.pyplot as plt
import numpy as np

from common import DATA, FIG
from ambac.field.assembly import load
from ambac.field.evaluate import safe_time

FIGF = FIG / "field"
BASE = load("robots", "tank_arm")
TORQUES = [None, 400, 250, 180, 140, 110, 80, 50, 30, 20]
SPEEDS = [None, 180, 120, 90, 60]

res = {}
print("=== 荷物運びの完了時間の最短値 [s]（tank_arm、行: 肩のトルクの上限、列: 速度の上限）===")
for st in ["practice", "moon"]:
    stage = load("stages", st)
    res[st] = {}
    print(f"\n[{st}]  g = {stage['gravity']}" + "".join(f"{('∞' if v is None else v):>9}" for v in SPEEDS) + "  （°/s）")
    for tq in TORQUES:
        row = []
        for sp in SPEEDS:
            r = copy.deepcopy(BASE)
            r["actuator"]["torque_limit"] = None if tq is None else {"pitch": tq, "elbow1": 0.6 * tq}
            r["actuator"]["speed_limit_deg_s"] = sp
            Tc, s = safe_time(r, stage, t_hi=15.0)
            T = None if np.isinf(Tc) else round(float(s["done_time_s"]), 3)   # 実際の完了時間
            res[st][f"{tq}_{sp}"] = dict(torque=tq, speed=sp, T=T, peak_torque=None if T is None else round(s["peak_torque_Nm"], 1))
            row.append(T)
        print(f"  肩 {('∞' if tq is None else tq):>4} N·m      " + "".join(f"{('不可' if v is None else f'{v:.2f}'):>9}" for v in row), flush=True)
json.dump(res, open(DATA / "field07_actuator.json", "w"), indent=2, ensure_ascii=False)

# 静的に荷物を水平に支えるのに必要な肩のトルク（腕の質量 4+3 kg、荷物 10 kg、長さ 0.5+0.5 m）
for st, g in [("practice", 9.80665), ("moon", 1.625)]:
    print(f"参考: {st} で腕を水平に伸ばして荷物を支える肩のトルク = {(4 * 0.25 + 3 * 0.75 + 10 * 1.0) * g:.0f} N·m")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
for k, st in enumerate(["practice", "moon"]):
    for sp, col in zip(SPEEDS, ["k", "tab:blue", "tab:green", "tab:orange", "tab:red"]):
        xs, ys = [], []
        for tq in TORQUES:
            v = res[st][f"{tq}_{sp}"]["T"]
            if v is not None:
                xs.append(600 if tq is None else tq); ys.append(v)
        ax[k].plot(xs, ys, "o-", color=col, label=f"speed limit {'none' if sp is None else f'{sp} deg/s'}")
    ax[k].set_xscale("log"); ax[k].set_xlabel("shoulder torque limit [N m]  (600 = no limit)")
    ax[k].set_title({"practice": "Earth (practice stage)", "moon": "Moon"}[st]); ax[k].grid(alpha=0.3)
ax[0].set_ylabel("shortest safe completion time [s]"); ax[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIGF / "field07_actuator.png", dpi=130)
print("図 → figures/field/field07_actuator.png")
