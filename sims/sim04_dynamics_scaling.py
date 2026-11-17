"""Sim04: 順動力学による AMBAC のトルク・パワー・エネルギーとスケーリング則.

Part A: 動力学モデルの検証（運動学モデルとの一致、エネルギー収支、角運動量保存）
Part B: 人間大モデルで最適ループを回すための関節トルク・パワー → トルク上限から f_max を決定
Part C: スケーリング（18 m 級）: 必要トルク・達成可能な周回周波数・90°旋回時間・角運動量容量
Part D: 同じ旋回をスラスタで行う場合の推進剤との比較

時間スケーリングの厳密な性質:
    同じ経路を f 倍速で回すと、τ = M q'' + (速度の2次項) なので τ ∝ f²、パワー ∝ f³。
長さスケーリング（幾何学的相似・同密度, 長さ s 倍）:
    質量 ∝ s³, 慣性 ∝ s⁵  → 同じ f での τ ∝ s⁵ f²
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from common import DATA, FIG, human_like
from ambac.planar import Limb, PlanarRobot

D = np.degrees
loopj = json.load(open(DATA / "sim03_optimal_loop.json"))
C1, C2, R = loopj["center_arm_rad"], loopj["center_leg_rad"], loopj["radius_rad"]
DTH_C = loopj["dtheta_per_cycle_rad"]


def scaled_human(s: float) -> PlanarRobot:
    """人間大モデル（身長約 1.8 m, 76 kg）を長さ s 倍・同密度で相似拡大."""
    base = human_like(arm_mass=4.0, leg_mass=12.0)
    m = s**3
    return PlanarRobot(base.base_mass * m, base.base_inertia * s**5,
                       [Limb((lb.attach[0] * s, lb.attach[1] * s), lb.mass * m, lb.length * s)
                        for lb in base.limbs])


def loop_traj(phi, phid, phidd=0.0):
    c, sn = np.cos(phi), np.sin(phi)
    q = np.array([C1 + R * c, C2 + R * sn])
    qd = np.array([-R * sn, R * c]) * phid
    qdd = np.array([-R * c, -R * sn]) * phid**2 + np.array([-R * sn, R * c]) * phidd
    return q, qd, qdd


def cycle_profile(robot, f, npts=400):
    """定常周回（周波数 f）1 周分の θ', 関節トルク, パワー."""
    ph = np.linspace(0, 2 * np.pi, npts, endpoint=False)
    w = 2 * np.pi * f
    out = []
    for p in ph:
        q, qd, qdd = loop_traj(p, w)
        thd, thdd, tau = robot.inverse_dynamics(q, qd, qdd)
        out.append([*q, thd, *tau, *(tau * qd)])
    return ph / w, np.array(out)   # cols: q1,q2,thd,tau1,tau2,P1,P2


# =============================================================================
# Part A: 検証
# =============================================================================
hum = scaled_human(1.0)
F_REF = 0.5
w_ref = 2 * np.pi * F_REF
T_A = 2 / F_REF                       # 2 周


def tau_ff(t):                        # 定常周回のフィードフォワード・トルク
    return hum.inverse_dynamics(*loop_traj(w_ref * t, w_ref))[2]


q0, qd0, _ = loop_traj(0.0, w_ref)
thd0 = hum.inverse_dynamics(q0, qd0, np.zeros(2))[0]
y0 = np.r_[0.0, q0, thd0, qd0]       # (θ, q1, q2, θ', q1', q2')


def rhs(t, y):
    acc = hum.forward_dynamics(y[1:3], y[3:], tau_ff(t))
    return np.r_[y[3:], acc]


sol = solve_ivp(rhs, (0, T_A), y0, rtol=1e-10, atol=1e-12, dense_output=True)
tt = np.linspace(0, T_A, 801)
Y = sol.sol(tt)
q_ref = np.array([loop_traj(w_ref * t, w_ref)[0] for t in tt]).T
L_hist = np.array([hum.mass_matrix(Y[1:3, i])[0] @ Y[3:, i] for i in range(len(tt))])
E_hist = np.array([hum.kinetic_energy(Y[1:3, i], Y[3:, i]) for i in range(len(tt))])
P_hist = np.array([tau_ff(t) @ Y[4:, i] for i, t in enumerate(tt)])
W_hist = np.concatenate([[0], np.cumsum((P_hist[1:] + P_hist[:-1]) / 2 * np.diff(tt))])
print("=== Part A: 動力学モデルの検証（人間大, 0.5 Hz, 2 周, トルクのみ入力・開ループ）===")
print(f"関節軌道の追従誤差 max      : {D(np.abs(Y[1:3] - q_ref).max()):.2e}°")
print(f"2 周後の正味回転 (動力学)   : {D(Y[0, -1]):+.4f}°  (運動学 Sim03: {D(2 * DTH_C):+.4f}°)")
print(f"角運動量 |L| max            : {np.abs(L_hist).max():.2e} kg m²/s")
print(f"エネルギー収支 |ΔT - ∫P dt| : {np.abs(E_hist - E_hist[0] - W_hist).max():.2e} J")

# =============================================================================
# Part B: 人間大のトルク・パワー
# =============================================================================
TAU_LIM_H = np.array([40.0, 120.0])   # 肩・股関節の持続可能トルクの目安 [N m]
t1, prof = cycle_profile(hum, F_REF)
tau_pk = np.abs(prof[:, 3:5]).max(0)
P_pos = np.clip(prof[:, 5:7], 0, None).mean(0)      # 回生なしの平均正パワー
P_pk = np.abs(prof[:, 5:7]).max(0)
f_max_h = F_REF * np.sqrt((TAU_LIM_H / tau_pk).min())   # τ ∝ f²
E90 = P_pos.sum() / F_REF * (np.pi / 2 / abs(DTH_C))      # 0.5 Hz で 90° 回る場合のエネルギー
print("\n=== Part B: 人間大（76 kg）, 最適ループ ===")
print(f"0.5 Hz での必要トルク peak: 肩 {tau_pk[0]:.1f} N m, 股 {tau_pk[1]:.1f} N m")
print(f"0.5 Hz でのパワー peak: 肩 {P_pk[0]:.1f} W, 股 {P_pk[1]:.1f} W / 平均正仕事率 計 {P_pos.sum():.1f} W")
print(f"トルク上限 (肩 {TAU_LIM_H[0]:.0f}, 股 {TAU_LIM_H[1]:.0f} N m) での f_max = {f_max_h:.2f} Hz")
print(f"90° 旋回: f_max で {np.pi / 2 / abs(DTH_C) / f_max_h:.1f} s, 0.5 Hz で消費 {E90:.0f} J（回生なし）")
np.savetxt(DATA / "sim04_cycle_human_0p5Hz.csv", np.c_[t1, prof], delimiter=",", comments="",
           header="t[s],q_arm[rad],q_leg[rad],theta_dot[rad/s],tau_arm[Nm],tau_leg[Nm],P_arm[W],P_leg[W]")

fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
ax[0].plot(tt, D(Y[0]), label="θ (forward dynamics, torque input)")
ax[0].plot(tt, D(DTH_C) * tt * F_REF, "k--", lw=.8, label="net rotation trend (kinematic)")
ax[0].set_ylabel("θ [deg]"); ax[0].legend(fontsize=8)
for k, nm in enumerate(["shoulder", "hip"]):
    ax[1].plot(tt, [tau_ff(t)[k] for t in tt], label=nm)
    ax[1].axhline(TAU_LIM_H[k], color=f"C{k}", ls=":", lw=.8); ax[1].axhline(-TAU_LIM_H[k], color=f"C{k}", ls=":", lw=.8)
ax[1].set_ylabel("joint torque [N m]"); ax[1].legend(fontsize=8)
ax[2].plot(tt, P_hist, "C2", label="total power"); ax[2].plot(tt, E_hist, "C4", label="kinetic energy [J]")
ax[2].set_ylabel("[W] / [J]"); ax[2].set_xlabel("time [s]"); ax[2].legend(fontsize=8)
for a in ax: a.grid(alpha=.3)
fig.suptitle("Sim04-A/B: human scale, optimal loop at 0.5 Hz (dotted: torque limits)")
fig.tight_layout(); fig.savefig(FIG / "sim04_human_cycle.png", dpi=150)

# =============================================================================
# Part C: スケーリング
# =============================================================================
# アクチュエータ仮定:
#   (i)  相似則: 同じ技術で相似拡大 → 最大トルク ∝ s³（断面積 s² × レバー s）
#   (ii) 人間の機敏さを保つのに必要なトルク（f を固定）→ ∝ s⁵
S_LIST = np.array([0.5, 1, 2, 5, 10, 20])
rows = []
for s in S_LIST:
    rob = scaled_human(s)
    _, pr = cycle_profile(rob, F_REF, npts=200)
    tpk = np.abs(pr[:, 3:5]).max(0)
    fmax = F_REF * np.sqrt((TAU_LIM_H * s**3 / tpk).min())
    A_mean = 1 / np.mean([1 / rob.coupling(pr[i, :2])[0] for i in range(len(pr))])
    h = fmax * abs(DTH_C) * A_mean
    T90 = np.pi / 2 / abs(DTH_C) / fmax
    Ppos = np.clip(pr[:, 5:7], 0, None).mean(0).sum() * (fmax / F_REF)**3
    rows.append([s, rob.total_mass, *tpk, fmax, T90, h, Ppos])
rows = np.array(rows)
np.savetxt(DATA / "sim04_scaling.csv", rows, delimiter=",", comments="",
           header="scale,total_mass[kg],tau_arm_at_0.5Hz[Nm],tau_leg_at_0.5Hz[Nm],f_max[Hz],T90[s],h_ambac[kg m^2/s],P_pos_at_fmax[W]")
print("\n=== Part C: スケーリング（トルク上限 ∝ s³ と仮定）===")
print(" s   質量[t]  股τ@0.5Hz[N m]  f_max[Hz]  90°旋回[s]  h[kg m²/s]  平均正パワー@f_max[kW]")
for r in rows:
    print(f"{r[0]:4.1f} {r[1]/1e3:8.3f} {r[3]:14.3g} {r[4]:10.3f} {r[5]:10.1f} {r[6]:11.3g} {r[7]/1e3:12.3g}")
k5 = np.polyfit(np.log(S_LIST), np.log(rows[:, 3]), 1)[0]
kf = np.polyfit(np.log(S_LIST), np.log(rows[:, 4]), 1)[0]
kh = np.polyfit(np.log(S_LIST), np.log(rows[:, 6]), 1)[0]
print(f"数値で得た指数: τ@固定f ∝ s^{k5:.2f}, f_max ∝ s^{kf:.2f}, h ∝ s^{kh:.2f}  (理論: 5, -1, 4)")

ms = rows[S_LIST == 10][0]
print(f"\n18 m 級 (s=10, {ms[1]/1e3:.0f} t):")
print(f"  人間と同じ 0.5 Hz で回すのに必要な股関節トルク {ms[3]/1e6:.2f} MN m（人間の {ms[3]/rows[1,3]:.0f} 倍）")
print(f"  相似則トルク（股 {TAU_LIM_H[1]:.0f} N m × s³ = {TAU_LIM_H[1]:.0f} kN m）では f_max {ms[4]:.3f} Hz, 90° 旋回 {ms[5]:.0f} s")

f10 = np.pi / 2 / abs(DTH_C) / 10.0
print(f"  90° を 10 s で旋回するには f = {f10:.2f} Hz → 股関節トルク {ms[3] * (f10 / F_REF)**2 / 1e6:.1f} MN m"
      f"（相似則の {ms[3] * (f10 / F_REF)**2 / (TAU_LIM_H[1] * 1e3):.0f} 倍）")

fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
ax[0].loglog(S_LIST, rows[:, 3], "o-", label="hip torque needed @0.5 Hz (∝s⁵)")
ax[0].loglog(S_LIST, TAU_LIM_H[1] * S_LIST**3, "s--", label="hip torque available (∝s³)")
ax[0].set_xlabel("length scale s"); ax[0].set_ylabel("[N m]"); ax[0].legend(fontsize=7)
ax[1].loglog(S_LIST, rows[:, 5], "o-C3"); ax[1].set_xlabel("length scale s"); ax[1].set_ylabel("90° turn time [s]")
ax[2].loglog(S_LIST, rows[:, 6], "o-C2"); ax[2].set_xlabel("length scale s"); ax[2].set_ylabel("AMBAC momentum capacity [kg m²/s]")
for a in ax:
    a.grid(alpha=.3, which="both"); a.axvline(1, color="gray", lw=.6); a.axvline(10, color="gray", lw=.6)
ax[1].set_title("s=1: human (1.8 m),  s=10: 18 m class")
fig.tight_layout(); fig.savefig(FIG / "sim04_scaling.png", dpi=150)

# =============================================================================
# Part D: スラスタとの比較（18 m 級）
# =============================================================================
#   静止→静止の 90° 旋回を時間 T でバンバン（加速→減速）で行う。
#   必要な角力積 = 2 A ω_peak = 4 A θ / T、推進剤 = 角力積 / (r Isp g0)
G0 = 9.80665
ISP = 300.0                    # 化学推進相当 [s]
s = 10.0
rob = scaled_human(s)
A_rob = rob.coupling([C1, C2])[0]
r_arm = 0.5 * s                # スラスタのモーメントアーム（胴体半高） [m]
print("\n=== Part D: 18 m 級の 90° 旋回 スラスタ vs AMBAC ===")
print(f"慣性モーメント A ≈ {A_rob:.3g} kg m², スラスタ腕 {r_arm:.1f} m, Isp {ISP:.0f} s")
for T in (5.0, 10.0, 30.0, ms[5]):
    mp = 4 * A_rob * (np.pi / 2) / T / (r_arm * ISP * G0)
    print(f"  T = {T:6.1f} s: 推進剤 {mp:7.2f} kg / 回")
E_amb = ms[7] * ms[5]
print(f"  AMBAC (T = {ms[5]:.0f} s): 推進剤 0 kg、関節の正仕事 約 {E_amb/1e6:.2f} MJ（回生なし）")
