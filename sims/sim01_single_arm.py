"""Sim01: 腕1本の振りによる姿勢変化（可逆的AMBAC）.

1) 腕を下(-90°)→前→上(+90°)→戻す、という往復で胴体姿勢がどう変わるか
2) 質量比・長さ比を振ったときの最大姿勢変化量（スケール則の確認）
"""
import numpy as np
import matplotlib.pyplot as plt

from common import DATA, FIG, human_like, integrate_attitude, smooth_step
from ambac.planar import Limb, PlanarRobot

# ---- 1) 往復運動の時系列 ------------------------------------------------
robot = human_like()
T, q0, q1 = 4.0, -np.pi / 2, np.pi / 2


def q_fun(t):
    s, _ = smooth_step(t / (T / 2)) if t < T / 2 else smooth_step(2 - t / (T / 2))
    return np.array([q0 + (q1 - q0) * s])


def qd_fun(t):
    if t < T / 2:
        _, ds = smooth_step(t / (T / 2)); sign = 1
    else:
        _, ds = smooth_step(2 - t / (T / 2)); sign = -1
    return np.array([(q1 - q0) * ds * sign / (T / 2)])


t, th = integrate_attitude(robot, q_fun, qd_fun, T)
qs = np.array([q_fun(ti)[0] for ti in t])
# 検証: 復元した theta_dot で全角運動量が 0 に保たれているか
thd = np.gradient(th, t)
Lres = np.array([robot.angular_momentum(th[i], [qs[i]], thd[i], qd_fun(t[i]))
                 for i in range(len(t))])
np.savetxt(DATA / "sim01_timeseries.csv", np.c_[t, qs, th, Lres], delimiter=",",
           header="t[s],q_arm[rad],theta_base[rad],L_residual[kg m^2/s]", comments="")

fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
ax[0].plot(t, np.degrees(qs)); ax[0].set_ylabel("arm angle q [deg]")
ax[1].plot(t, np.degrees(th), color="C3"); ax[1].set_ylabel("base attitude θ [deg]")
ax[1].set_xlabel("time [s]")
ax[0].set_title("Sim01: single-arm swing (L = 0)")
for a in ax: a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "sim01_timeseries.png", dpi=150)
print(f"[1] 腕振り 180° での最大姿勢変化: {np.degrees(th.min()):.2f}° / "
      f"往復後の残留: {np.degrees(th[-1]):.2e}° / max|L|: {np.abs(Lres).max():.2e}")

# ---- 2) 質量比・長さ比スイープ ------------------------------------------
# 腕の振り q0→q1 による Δθ = -∫Γ(q)dq は経路のみに依存（速度・時間に依存しない）
mu_list = np.linspace(0.01, 0.4, 40)          # 腕質量 / 全質量
lam_list = [0.35, 0.5, 0.7, 1.0]              # 腕長 / 胴体高さ(1.0m)
qq = np.linspace(q0, q1, 801)
rows = []
fig, ax = plt.subplots(figsize=(7, 4.5))
for lam in lam_list:
    dth = []
    for mu in mu_list:
        M = 64.0; ma = mu * M; m0 = M - ma
        r = PlanarRobot(m0, m0 * (0.35**2 + 1.0**2) / 12.0,
                        [Limb((0.0, 0.35), ma, lam * 1.0)])
        G = np.array([r.connection([qi])[0] for qi in qq])
        d = -np.trapezoid(G, qq)
        dth.append(d); rows.append((mu, lam, d))
    ax.plot(mu_list, np.degrees(np.abs(dth)), label=f"arm length ratio {lam}")
ax.set_xlabel("arm mass / total mass  μ"); ax.set_ylabel("|Δθ| for 180° swing [deg]")
ax.set_title("Sim01: attitude change vs. mass/length ratio (scale-free)")
ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "sim01_sweep.png", dpi=150)
np.savetxt(DATA / "sim01_sweep.csv", np.array(rows), delimiter=",",
           header="mu,length_ratio,delta_theta[rad]", comments="")
print("[2] スイープ結果を data/sim01_sweep.csv に保存")

# スケール不変性の確認: 全長さ x10, 全質量 x1000 でも Δθ は同じ
def dtheta(scale_L, scale_M):
    r = PlanarRobot(60 * scale_M, 60 * scale_M * ((0.35 * scale_L)**2 + scale_L**2) / 12,
                    [Limb((0.0, 0.35 * scale_L), 4 * scale_M, 0.7 * scale_L)])
    G = np.array([r.connection([qi])[0] for qi in qq])
    return -np.trapezoid(G, qq)
print(f"[3] スケール則: 人間大 {np.degrees(dtheta(1,1)):.4f}° / "
      f"18m級(長さx10,質量x1000) {np.degrees(dtheta(10,1000)):.4f}°")
