"""Field02: 地上の試験場（MuJoCo）で Sim06 と照合する（シミュレーション場の段階 2）.

Test A: 荷物運び（前方の地面 → 頭上 → 後部デッキ）。動作時間 T を変え、台車が傾き始める（転倒の始まり）・滑り始める速さを、
        Python の ZMP・摩擦モデル（MuJoCo が実際にたどった関節の動きを入力）と比べる。
Test B: 月面で、腕（旋回関節）を速く振ってゆっくり戻したときの台車のヨーを、Sim06 Part C のスティックスリップのモデルと比べる。
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import json

import numpy as np
import matplotlib.pyplot as plt

from common import DATA, FIG
from ambac.ground import TankArm, min_jerk
from ambac.field.ground import G_EARTH, G_MOON, tank_arm_mjcf, run

D = np.degrees
RAD = np.radians
OUT = FIG / "field"; OUT.mkdir(exist_ok=True)
res = {}

# =============================================================================
# Test A: 荷物運び
# =============================================================================
Q0, Q1 = RAD([-20.0, -50.0]), RAD([130.0, 60.0])
xmlA = tank_arm_mjcf()
ref = lambda tau, t: Q0 + (Q1 - Q0) * min_jerk(tau)
py = TankArm()
rowsA = []
print("=== Test A: 荷物運び（地球, μ=0.6）===")
print("  T[s]  MuJoCo: 最大の傾き / 水平のずれ   |  Python（同じ関節の動き）: ZMP の最大 / 必要な摩擦係数")
for T in (2.0, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9):
    o = run(xmlA, Q0, ref, T)
    mv = (o["t"] >= 0) & (o["t"] <= T)
    tilt = D(np.abs(o["pitch"][o["t"] >= 0]).max())
    slide = np.linalg.norm(o["xy"][-1] - o["xy"][np.argmin(abs(o["t"]))]) * 1000
    qp = o["q"][mv]; tt = o["t"][mv]
    tu = np.linspace(0, T, 401); qu = np.c_[np.interp(tu, tt, qp[:, 0]), np.interp(tu, tt, qp[:, 1])]
    p, N, Fx = py.reactions(qu, T)
    zmax = np.abs(p).max(); mu_need = np.abs(Fx / N).max()
    rowsA.append((T, tilt, slide, zmax, mu_need))
    print(f"  {T:4.2f}  {tilt:6.2f}° / {slide:6.1f} mm          |  {zmax:.3f} m（接地範囲 0.4 m） / {mu_need:.2f}")
rowsA = np.array(rowsA)


def threshold(T, val, lim):
    """val が lim を初めて超える T（線形補間）。T は降順"""
    for i in range(1, len(T)):
        if val[i] >= lim > val[i - 1]:
            return T[i - 1] + (lim - val[i - 1]) * (T[i] - T[i - 1]) / (val[i] - val[i - 1])
    return np.nan


T_tip_py = threshold(rowsA[:, 0], rowsA[:, 3], 0.4)
T_tip_mj = threshold(rowsA[:, 0], rowsA[:, 1], 1.0)
T_slip_py = threshold(rowsA[:, 0], rowsA[:, 4], 0.6)
T_slip_mj = threshold(rowsA[:, 0], rowsA[:, 2], 5.0)
res["testA"] = dict(T_tip_python=T_tip_py, T_tip_mujoco=T_tip_mj, T_slip_python=T_slip_py, T_slip_mujoco=T_slip_mj,
                    rows=rowsA.tolist())
print(f"  転倒の始まり: Python（ZMP が 0.4 m に達する）{T_tip_py:.2f} s / MuJoCo（傾きが 1° を超える）{T_tip_mj:.2f} s")
print(f"  滑りの始まり: Python（必要な摩擦係数が 0.6 に達する）{T_slip_py:.2f} s / MuJoCo（台車が 5 mm 以上ずれる）{T_slip_mj:.2f} s")

# =============================================================================
# Test B: 月面での旋回のスティックスリップ
# =============================================================================
import importlib.util
src = open("sim06_ground_tank.py", encoding="utf-8").read()
ns = {"__name__": "field"}
exec(src[:src.index("# =============================================================================\n# Part A")], ns)
exec(src[src.index("L_T, W_T = 0.8, 0.6"):src.index('print("\\n=== Part C')], ns)
yaw_py = ns["yaw_cycle"]
SWING, T_SLOW = RAD(90.0), 3.0
rowsB = []
print("\n=== Test B: 腕を水平に 90° 速く振り 3 s で戻す（1 往復の台車のヨー）===")
print("  重力   T_fast[s]  MuJoCo: ヨー / 浮き上がり（接触点 0 の時間）/ 最大の傾き  |  Python: 一様な接地圧 / 四隅の接触")
Rbar_saved = ns["R_BAR"]
for gname, g in (("地球", G_EARTH), ("月", G_MOON)):
    xmlB = tank_arm_mjcf(g=g, turret=True, kp=(20000,), kv=(1500,))
    for Tf in (0.4, 0.6, 0.8, 1.0, 1.4, 2.0, 2.6):
        Tc = Tf + T_SLOW

        def refB(tau, t, Tf=Tf):
            if t < 0: return [0.0]
            if t < Tf: return [SWING * min_jerk(t / Tf)]
            return [SWING * (1 - min_jerk(min((t - Tf) / T_SLOW, 1.0)))]
        o = run(xmlB, [0.0], refB, Tc, hold=0.5)
        mv = o["t"] >= 0
        i0 = np.argmin(abs(o["t"])); yaw_mj = D(o["yaw"][-1] - o["yaw"][i0])
        air = float(np.sum(o["ncon"][mv] == 0) * (o["t"][1] - o["t"][0]))
        tilt = D(max(np.abs(o["pitch"][mv]).max(), np.abs(o["roll"][mv]).max()))
        ns["R_BAR"] = Rbar_saved; a_ = D(yaw_py(Tf, g, dt=5e-4))
        ns["R_BAR"] = 0.5; b_ = D(yaw_py(Tf, g, dt=5e-4)); ns["R_BAR"] = Rbar_saved
        rowsB.append((g, Tf, yaw_mj, air, tilt, a_, b_))
        flag = "（浮き上がり）" if air > 0.01 else ""
        print(f"  {gname}   {Tf:4.1f}      {yaw_mj:+7.2f}° / {air:4.2f} s / {tilt:5.1f}°{flag}  |  {a_:+7.2f}° / {b_:+7.2f}°")
rowsB = np.array(rowsB)
res["testB"] = dict(rows=rowsB.tolist())
json.dump(res, open(DATA / "field02_ground.json", "w"), indent=2, ensure_ascii=False)

# ---- 図 -------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(rowsA[:, 0], rowsA[:, 1], "o-", label="MuJoCo: max tilt [deg]")
ax0b = ax[0].twinx(); ax0b.plot(rowsA[:, 0], rowsA[:, 3], "s--C1", label="Python: max |ZMP| [m]"); ax0b.axhline(0.4, color="C1", lw=.8)
ax0b.set_ylim(0, 1.0); ax[0].axvspan(rowsA[:, 0].min() - 0.05, T_tip_mj, color="gray", alpha=.12)
ax[0].text(0.95, 11, "base tilting:\nPython (fixed base)\nnot valid here", fontsize=7)
ax[0].axvline(T_tip_py, color="C1", ls=":"); ax[0].axvline(T_tip_mj, color="C0", ls=":")
ax[0].set_xlabel("motion time T [s]"); ax[0].set_ylabel("MuJoCo max tilt [deg]"); ax0b.set_ylabel("Python max |ZMP| [m]")
ax[0].set_title(f"A: tip-over onset  Python {T_tip_py:.2f} s / MuJoCo {T_tip_mj:.2f} s"); ax[0].grid(alpha=.3)
ax[1].plot(rowsA[:, 0], rowsA[:, 2], "o-", label="MuJoCo: base slide [mm]")
ax1b = ax[1].twinx(); ax1b.plot(rowsA[:, 0], rowsA[:, 4], "s--C1"); ax1b.axhline(0.6, color="C1", lw=.8)
ax1b.set_ylim(0, 1.5); ax[1].axvspan(rowsA[:, 0].min() - 0.05, T_tip_mj, color="gray", alpha=.12)
ax[1].set_xlabel("motion time T [s]"); ax[1].set_ylabel("MuJoCo base slide [mm]"); ax1b.set_ylabel("Python required μ")
ax[1].set_title("A: slip (gray: base already tilting — tipping comes first)"); ax[1].grid(alpha=.3)
for gv, c, gname in ((G_EARTH, "C0", "Earth"), (G_MOON, "C3", "Moon")):
    r = rowsB[np.isclose(rowsB[:, 0], gv)]
    ok = r[:, 3] < 0.01
    ax[2].plot(r[:, 1], r[:, 2], "o-", color=c, label=f"MuJoCo {gname}")
    ax[2].plot(r[~ok, 1], r[~ok, 2], "x", color="k", ms=10, mew=2)
    ax[2].plot(r[:, 1], r[:, 5], "s--", color=c, alpha=.6, label=f"Python {gname} (r̄=0.271)")
    ax[2].plot(r[:, 1], r[:, 6], "^:", color=c, alpha=.6, label=f"Python {gname} (r=0.5)")
ax[2].set_xlabel("fast-stroke time [s]"); ax[2].set_ylabel("net yaw per cycle [deg]"); ax[2].legend(fontsize=6, ncol=2)
ax[2].set_title("B: stick–slip yaw (×: base lifted off the ground)"); ax[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "field02_ground.png", dpi=150)
