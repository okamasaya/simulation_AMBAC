"""Sim02: 2肢の周期運動による正味の姿勢変化（幾何学的位相 / 非ホロノミックAMBAC）.

腕(q1)と脚(q2)を関節空間で長方形ループ状に動かすと、
1周ごとに形状は元に戻るのに胴体姿勢だけが Δθ ずつ進む（落下猫と同じ原理）。
数値積分の結果を、曲率 F(q) の面積分（ストークスの定理）と照合する。
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad

from common import DATA, FIG, human_like, integrate_attitude, smooth_step

robot = human_like(arm_mass=4.0, leg_mass=12.0)

# 関節空間の長方形ループ（反時計回り）: 角を順に結ぶ
q1a, q1b = -np.pi / 2, np.pi / 2      # 腕: 下 ⇄ 上
q2a, q2b = -np.pi / 2, -np.pi / 6     # 脚: 真下 ⇄ 前方へ60°
corners = np.array([[q1a, q2a], [q1b, q2a], [q1b, q2b], [q1a, q2b], [q1a, q2a]])
Tseg = 1.0; T1 = 4 * Tseg; ncyc = 5


def q_fun(t):
    t = t % T1; k = min(int(t // Tseg), 3)
    s, _ = smooth_step((t - k * Tseg) / Tseg)
    return corners[k] + (corners[k + 1] - corners[k]) * s


def qd_fun(t):
    t = t % T1; k = min(int(t // Tseg), 3)
    _, ds = smooth_step((t - k * Tseg) / Tseg)
    return (corners[k + 1] - corners[k]) * ds / Tseg


t, th = integrate_attitude(robot, q_fun, qd_fun, ncyc * T1, n=4001)
Q = np.array([q_fun(ti) for ti in t])
np.savetxt(DATA / "sim02_timeseries.csv", np.c_[t, Q, th], delimiter=",",
           header="t[s],q_arm[rad],q_leg[rad],theta_base[rad]", comments="")
dth_sim = th[np.searchsorted(t, T1)]

# ストークスによる予測: Δθ = -∬ F dq1 dq2（反時計回りループ）
dth_stokes, _ = dblquad(lambda q2, q1: -robot.curvature([q1, q2]),
                        q1a, q1b, q2a, q2b, epsabs=1e-9)
print(f"1周あたりの正味回転  数値積分: {np.degrees(dth_sim):+.4f}°  "
      f"ストークス予測: {np.degrees(dth_stokes):+.4f}°")
print(f"{ncyc}周後: {np.degrees(th[-1]):+.3f}°  → 360°回転に必要な周回数 ≈ "
      f"{360 / abs(np.degrees(dth_sim)):.1f}")

# 曲率マップ
g1 = np.linspace(-np.pi, np.pi, 121); g2 = np.linspace(-np.pi, np.pi, 121)
F = np.array([[robot.curvature([a, b]) for a in g1] for b in g2])
np.savez(DATA / "sim02_curvature.npz", q1=g1, q2=g2, F=F)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
vmax = np.abs(F).max()
im = ax[0].pcolormesh(np.degrees(g1), np.degrees(g2), F, cmap="RdBu_r",
                      vmin=-vmax, vmax=vmax, shading="auto")
ax[0].plot(np.degrees(corners[:, 0]), np.degrees(corners[:, 1]), "k-", lw=2)
ax[0].annotate("", xy=np.degrees(corners[1]) * [0.5, 1], xytext=np.degrees(corners[0]),
               arrowprops=dict(arrowstyle="->", lw=2))
ax[0].set_xlabel("arm q1 [deg]"); ax[0].set_ylabel("leg q2 [deg]")
ax[0].set_title("curvature F(q)  (net rotation = -∬F)")
fig.colorbar(im, ax=ax[0])
ax[1].plot(t, np.degrees(th), color="C3")
for k in range(1, ncyc + 1): ax[1].axvline(k * T1, color="gray", lw=.5)
ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("base attitude θ [deg]")
ax[1].set_title(f"Sim02: {ncyc} cycles of arm-leg loop"); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "sim02_cyclic.png", dpi=150)
