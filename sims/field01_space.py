"""Field01: 宇宙の試験場（MuJoCo）で Sim01・Sim02 と照合する（シミュレーション場の段階 1）.

照合の方法: MuJoCo が実際にたどった関節角の時系列を、第 1 段階の Python モデル（ambac.planar）に入れて
胴体の回転を計算し直し、MuJoCo の胴体の回転と比べる（モーターの追従誤差に左右されない比較）。
あわせて、Sim01・Sim02 の基準値（−16.04°、+2.9778°/周）とも比べる。
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import json

import numpy as np
import matplotlib.pyplot as plt

from common import DATA, FIG, human_like, smooth_step
from ambac.field.space import planar_robot_mjcf, run

D = np.degrees
RAD = np.radians
OUT = FIG / "field"; OUT.mkdir(exist_ok=True)
res = {}


def python_theta(robot, q_traj):
    """関節角の時系列から、平面モデルで胴体の回転を計算（区間ごとの中点で接続を評価）"""
    th = [0.0]
    for a, b in zip(q_traj[:-1], q_traj[1:]):
        th.append(th[-1] - robot.connection((a + b) / 2) @ (b - a))
    return np.array(th)


# ---- Test 1: 腕 1 本の往復（Sim01） -----------------------------------------------
r1 = human_like()                                     # 胴体 60 kg, 腕 4 kg・0.7 m（肩 (0, 0.35)）
xml1 = planar_robot_mjcf(60.0, (0.35, 1.0), [("arm", (0.0, 0.35), 4.0, 0.7, 0)])
q0, q1 = -np.pi / 2, np.pi / 2
summary = []
for T in (4.0, 1.0):
    def ref(t, T=T):
        s, _ = smooth_step(t / (T / 2)) if t < T / 2 else smooth_step(2 - t / (T / 2))
        return [q0 + (q1 - q0) * s]
    o = run(xml1, [q0], ref, T + 0.5)
    th_py = python_theta(r1, o["q"])
    i_mid = np.argmin(abs(o["t"] - T / 2))
    res[f"sim01_T{T}"] = dict(theta_mid_mj=D(o["theta"][i_mid]), theta_mid_py=D(th_py[i_mid]),
                              theta_end_mj=D(o["theta"][-1]), max_diff=D(np.abs(o["theta"] - th_py).max()),
                              L_max=float(np.abs(o["L"]).max()), q_mid=D(o["q"][i_mid, 0]))
    summary.append((T, o))
    print(f"[Sim01, T={T} s] 腕 {res[f'sim01_T{T}']['q_mid']:.2f}° のとき 胴体: MuJoCo {res[f'sim01_T{T}']['theta_mid_mj']:+.4f}° / "
          f"Python {res[f'sim01_T{T}']['theta_mid_py']:+.4f}°（基準 Sim01: −16.04°）, 往復後 {res[f'sim01_T{T}']['theta_end_mj']:+.2e}°, "
          f"最大差 {res[f'sim01_T{T}']['max_diff']:.2e}°, |L|max {res[f'sim01_T{T}']['L_max']:.1e}")

# ---- Test 2: 腕＋脚の長方形ループ（Sim02） ----------------------------------------------
r2 = human_like(arm_mass=4.0, leg_mass=12.0)
xml2 = planar_robot_mjcf(60.0, (0.35, 1.0), [("arm", (0.0, 0.35), 4.0, 0.7, 0), ("leg", (0.0, -0.15), 12.0, 0.9, 0)])
corners = np.array([[-90, -90], [90, -90], [90, -30], [-90, -30], [-90, -90]], float)
corners = RAD(corners); Tseg = 1.0; T1 = 4 * Tseg; ncyc = 3


def ref2(t):
    t = t % T1; k = min(int(t // Tseg), 3)
    s, _ = smooth_step((t - k * Tseg) / Tseg)
    return corners[k] + (corners[k + 1] - corners[k]) * s


o2 = run(xml2, corners[0], ref2, ncyc * T1 + 0.5)
th_py2 = python_theta(r2, o2["q"])
per = []
for c in range(1, ncyc + 1):
    i = np.argmin(abs(o2["t"] - c * T1))
    per.append(D(o2["theta"][i]) / c)
res["sim02"] = dict(per_cycle_mj=float(np.mean(per)), per_cycle_py=float(D(th_py2[np.argmin(abs(o2["t"] - ncyc * T1))]) / ncyc),
                    max_diff=D(np.abs(o2["theta"] - th_py2).max()), L_max=float(np.abs(o2["L"]).max()),
                    track_err=D(np.abs(o2["q"] - np.array([ref2(t) for t in o2["t"]])).max()))
print(f"[Sim02] 1 周あたり: MuJoCo {res['sim02']['per_cycle_mj']:+.4f}° / Python {res['sim02']['per_cycle_py']:+.4f}°"
      f"（基準 Sim02: +2.9778°）, 最大差 {res['sim02']['max_diff']:.2e}°, |L|max {res['sim02']['L_max']:.1e}, 関節の追従誤差 max {res['sim02']['track_err']:.2f}°")

json.dump(res, open(DATA / "field01_space.json", "w"), indent=2, ensure_ascii=False)

# ---- 図 ----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for T, o in summary:
    ax[0].plot(o["t"] / T, D(o["theta"]), label=f"MuJoCo, T={T} s")
    ax[0].plot(o["t"] / T, D(python_theta(r1, o["q"])), "k:", lw=1)
ax[0].axhline(-16.04, color="gray", ls="--", lw=.8)
ax[0].set_xlabel("normalized time t/T"); ax[0].set_ylabel("base attitude [deg]")
ax[0].set_title("Field01 vs Sim01: single-arm swing (dotted: Python model)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(o2["t"], D(o2["theta"]), label="MuJoCo")
ax[1].plot(o2["t"], D(th_py2), "k:", lw=1, label="Python model (same joint path)")
for c in range(1, ncyc + 1):
    ax[1].plot(c * T1, 2.9778 * c, "r*", ms=10)
ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("base attitude [deg]")
ax[1].set_title("Field01 vs Sim02: arm–leg loop (★: Sim02 reference per cycle)"); ax[1].legend(); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "field01_space.png", dpi=150)

# ---- 試験場の様子（画像）--------------------------------------------------------------
try:
    import mujoco
    m = mujoco.MjModel.from_xml_string(xml2); d = mujoco.MjData(m)
    frames = []
    for t in (0.0, 1.0, 2.0, 3.0):
        d.qpos[:] = 0; d.qpos[3] = 1.0; d.qpos[7:9] = ref2(t)
        k = np.argmin(abs(o2["t"] - t))
        d.qpos[3:7] = [np.cos(o2["theta"][k] / 2), 0, -np.sin(o2["theta"][k] / 2), 0]   # −y 軸まわりに θ
        mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, 300, 300)
        cam = mujoco.MjvCamera(); cam.lookat[:] = [0, 0, 0]; cam.distance = 3.2; cam.azimuth = 90; cam.elevation = 0
        r.update_scene(d, camera=cam); frames.append(r.render()); r.close()
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
    for a, f, t in zip(ax, frames, (0, 1, 2, 3)):
        a.imshow(f); a.set_title(f"t = {t} s"); a.axis("off")
    fig.suptitle("Field01: space test field (MuJoCo, no gravity) — arm & leg loop")
    fig.tight_layout(); fig.savefig(OUT / "field01_snapshots.png", dpi=120)
    print("画像を保存: figures/field/field01_snapshots.png")
except Exception as e:  # 描画できない環境では省略
    print("描画を省略:", type(e).__name__, e)
