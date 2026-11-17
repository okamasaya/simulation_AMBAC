"""Sim03: リアクションホイール(RW) と AMBAC の比較.

統一的な見方: どちらも L = A(q)θ' + B(q)·q' の「運動量交換」であり、
    θ' = (L - B(q)·q') / A(q)
で記述される。違いは関節の性質だけ:
    RW    : 連続回転できる関節。回しても質量配置が変わらない → Γ = 一定
    AMBAC : 可動範囲のある関節。質量配置が変わる → Γ(q) が形状に依存、曲率で正味回転

Part A: Γ の比較と、関節可動域内での手足ループ最適化（1周あたり回転の最大化）
Part B: 90° の rest-to-rest 姿勢変更
Part C: 一定外乱トルク下の姿勢保持と「飽和」— RW のホイール速度飽和 vs AMBAC の周回周波数飽和

※ 本 Sim は速度指令レベル（運動学）での比較。関節/モータトルクと消費エネルギーは Sim04（順動力学）で扱う。
"""
import numpy as np
import matplotlib.pyplot as plt

from common import DATA, FIG, human_like, smooth_step
from ambac.planar import reaction_wheel

D = np.degrees

# ---- 機体モデル --------------------------------------------------------------
amb = human_like(arm_mass=4.0, leg_mass=12.0)          # 腕+脚 (AMBAC 用)
rw = human_like(arm_mass=4.0, leg_mass=12.0)
rw.limbs.append(reaction_wheel(mass=2.0, radius=0.12))  # 胴体に小型 RW を追加
W = 2                                                   # RW の関節番号
OMEGA_MAX = 600.0                                       # RW 最大回転数 [rad/s] (~5700 rpm)
F_MAX = 1.0                                             # 手足ループ最大周回周波数 [Hz]

# 関節可動域（胴体座標: -90°=真下, 0°=前方, +90°=真上）
Q1_LIM = np.radians([-150, 150])   # 腕（肩）
Q2_LIM = np.radians([-120, 30])    # 脚（股関節）

# =============================================================================
# Part A: Γ と曲率、ループ最適化
# =============================================================================
n = 121
g1 = np.linspace(*Q1_LIM, n); g2 = np.linspace(*Q2_LIM, n)
G = np.array([[amb.connection([a, b]) for a in g1] for b in g2])   # [i2, i1, k]
Ainv = np.array([[1.0 / amb.coupling([a, b])[0] for a in g1] for b in g2])
h1, h2 = g1[1] - g1[0], g2[1] - g2[0]
F = np.gradient(G[..., 1], h1, axis=1) - np.gradient(G[..., 0], h2, axis=0)
Q1, Q2 = np.meshgrid(g1, g2)

# 上界: 可動域内で F が同符号の領域全体を囲った場合
bound_neg = -np.sum(F[F < 0]) * h1 * h2
bound_pos = -np.sum(F[F > 0]) * h1 * h2

# 円ループの全探索（中心・半径、可動域内に収まるもの）
best = (0, None)
for r in np.radians(np.arange(10, 91, 5)):
    for c1 in g1[::4]:
        for c2 in g2[::4]:
            if c1 - r < Q1_LIM[0] or c1 + r > Q1_LIM[1] or c2 - r < Q2_LIM[0] or c2 + r > Q2_LIM[1]:
                continue
            mask = (Q1 - c1)**2 + (Q2 - c2)**2 <= r**2
            dth = -np.sum(F[mask]) * h1 * h2          # 反時計回りループの正味回転
            if abs(dth) > abs(best[0]):
                best = (dth, (c1, c2, r))
dth_c, (c1, c2, R) = best


def loop(phi):
    return np.array([c1 + R * np.cos(phi), c2 + R * np.sin(phi)])


def loop_d(phi):
    return np.array([-R * np.sin(phi), R * np.cos(phi)])


# ループ上で厳密に（線積分で）検証
ph = np.linspace(0, 2 * np.pi, 2001)
Gl = np.array([amb.connection(loop(p)) for p in ph])
dth_c = -np.trapezoid(np.einsum("ik,ki->i", Gl, np.array([loop_d(p) for p in ph]).T), ph)
inv_A_mean = np.mean([1.0 / amb.coupling(loop(p))[0] for p in ph[:-1]])
print("=== Part A ===")
print(f"Sim02 の長方形ループ: +2.98°/周")
print(f"最適円ループ: 中心(腕 {D(c1):.0f}°, 脚 {D(c2):.0f}°), 半径 {D(R):.0f}° → {D(dth_c):+.2f}°/周")
print(f"可動域内の理論上界（同符号曲率の全領域）: {D(bound_neg):+.2f}° / {D(bound_pos):+.2f}°")
A_rw, B_rw = rw.coupling([np.pi / 2 * -1, -np.pi / 2, 0.0])
print(f"RW の Γ_w = I_w/A = {B_rw[W] / A_rw:.5f} (一定) → ホイール1回転あたり {D(2*np.pi*B_rw[W]/A_rw):.2f}°")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
vmax = np.abs(F).max()
im = ax[0].pcolormesh(D(g1), D(g2), F, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
ax[0].contour(D(g1), D(g2), F, levels=[0], colors="k", linewidths=.8, linestyles="--")
lp = np.array([loop(p) for p in ph])
ax[0].plot(D(lp[:, 0]), D(lp[:, 1]), "k-", lw=2, label=f"optimal loop {D(dth_c):+.1f}°/cycle")
ax[0].set_xlabel("arm q1 [deg]"); ax[0].set_ylabel("leg q2 [deg]")
ax[0].set_title("curvature within joint limits"); ax[0].legend(loc="lower right", fontsize=8)
fig.colorbar(im, ax=ax[0])
qq = np.linspace(*Q1_LIM, 200)
ax[1].plot(D(qq), [amb.connection([a, -np.pi / 2])[0] for a in qq], label="AMBAC arm Γ1(q1) (leg down)")
ax[1].plot(D(qq), [amb.connection([-np.pi / 2, b])[1] if Q2_LIM[0] <= b <= Q2_LIM[1] else np.nan
                   for b in qq], label="AMBAC leg Γ2(q2) (arm down)")
ax[1].axhline(B_rw[W] / A_rw, color="C3", label="reaction wheel Γw (constant)")
ax[1].set_xlabel("joint angle [deg]"); ax[1].set_ylabel("Γ = B/A  [-]")
ax[1].set_title("mechanical connection: RW vs limbs"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "sim03_connection.png", dpi=150)

# =============================================================================
# Part B: 90° rest-to-rest
# =============================================================================
TARGET = np.radians(90.0)
# RW: Δθ = -(I_w/A) Δφ_w
q_frozen = np.array([-np.pi / 2, -np.pi / 2])
A0, B0 = rw.coupling([*q_frozen, 0.0])
dphi_w = -TARGET * A0 / B0[W]
T_rw = 10.0
t = np.linspace(0, T_rw, 1001)
s, ds = smooth_step(t / T_rw)
th_rw = -(B0[W] / A0) * dphi_w * s
w_rw = dphi_w * ds / T_rw

# AMBAC: 最適ループを N 周（周波数 0.5 Hz, 位相は滑らかに加減速）
N = int(round(TARGET / abs(dth_c)))
sgn = np.sign(TARGET / dth_c)
T_amb = N / 0.5
dt = 0.002
ta = np.arange(0, T_amb + dt, dt)
th_a = np.zeros_like(ta); qa = np.zeros((len(ta), 2))


def amb_rate(phi, phid, L=0.0):
    A, B = amb.coupling(loop(phi))
    return (L - B @ (loop_d(phi) * phid)) / A


def phase(tt):
    s, ds = smooth_step(tt / T_amb)
    return sgn * 2 * np.pi * N * s, sgn * 2 * np.pi * N * ds / T_amb


for i in range(len(ta) - 1):  # RK4
    t0, y = ta[i], th_a[i]
    k = []
    for tt in (t0, t0 + dt / 2, t0 + dt / 2, t0 + dt):
        k.append(amb_rate(*phase(tt)))
    th_a[i + 1] = y + dt / 6 * (k[0] + 2 * k[1] + 2 * k[2] + k[3])
qa = np.array([loop(phase(tt)[0]) for tt in ta])
print("\n=== Part B: 90° rest-to-rest ===")
print(f"RW   : ホイール {abs(dphi_w)/(2*np.pi):.1f} 回転, 最大 {np.abs(w_rw).max()*60/(2*np.pi):.0f} rpm, 所要 {T_rw:.0f} s → {D(th_rw[-1]):.2f}°")
print(f"AMBAC: 最適ループ {N} 周, 所要 {T_amb:.0f} s → {D(th_a[-1]):.2f}°（整数周回による端数あり）, 途中の最大姿勢 {D(th_a.max()):.1f}°")
np.savetxt(DATA / "sim03_rest2rest_rw.csv", np.c_[t, th_rw, w_rw], delimiter=",",
           header="t[s],theta[rad],wheel_rate[rad/s]", comments="")
np.savetxt(DATA / "sim03_rest2rest_ambac.csv", np.c_[ta, th_a, qa][::5], delimiter=",",
           header="t[s],theta[rad],q_arm[rad],q_leg[rad]", comments="")

fig, ax = plt.subplots(2, 2, figsize=(11, 5.5), sharex="col")
ax[0, 0].plot(t, D(th_rw), "C3"); ax[0, 0].set_ylabel("θ [deg]"); ax[0, 0].set_title("Reaction wheel")
ax[1, 0].plot(t, w_rw * 60 / (2 * np.pi), "C3"); ax[1, 0].set_ylabel("wheel speed [rpm]")
ax[0, 1].plot(ta, D(th_a), "C0"); ax[0, 1].set_title(f"AMBAC ({N} limb cycles)")
ax[1, 1].plot(ta, D(qa[:, 0]), label="arm"); ax[1, 1].plot(ta, D(qa[:, 1]), label="leg")
ax[1, 1].set_ylabel("joint angle [deg]"); ax[1, 1].legend(fontsize=8)
for a in ax.flat: a.grid(alpha=.3)
ax[0, 0].axhline(90, color="gray", ls=":"); ax[0, 1].axhline(90, color="gray", ls=":")
ax[1, 0].set_xlabel("time [s]"); ax[1, 1].set_xlabel("time [s]")
fig.suptitle("Sim03-B: 90° rest-to-rest reorientation"); fig.tight_layout()
fig.savefig(FIG / "sim03_rest2rest.png", dpi=150)

# =============================================================================
# Part C: 一定外乱トルク下の姿勢保持
#   L(t) = τ_d t が増え続ける。内部運動では L を消せないので、
#   RW はホイールに、AMBAC は手足の周期運動に角運動量を「溜める」しかない。
# =============================================================================
TAU_D = 0.01          # 外乱トルク [N m]（例: 物体の接触、排気、重力傾斜など）
T_END = 1200.0
KP = 0.5              # 姿勢フィードバックゲイン [1/s]
dt = 0.005
tc = np.arange(0, T_END + dt, dt)

# --- RW: ホイール角速度を直接指令（飽和あり）
th = 0.0; th_r = np.zeros_like(tc); wr = np.zeros_like(tc)
for i, ti in enumerate(tc):
    L = TAU_D * ti
    w_cmd = (L - A0 * (-KP * th)) / B0[W]
    w = np.clip(w_cmd, -OMEGA_MAX, OMEGA_MAX)
    th_r[i], wr[i] = th, w
    th += dt * (L - B0[W] * w) / A0

# --- AMBAC: ループ周回周波数 f を指令（平均化モデルで設計、飽和あり）
#   1周平均の姿勢変化率 ≈ f·Δθc + L·<1/A>  → f = (-KP θ̄ - L<1/A>) / Δθc
#   θ は周回内で大きく揺れる（形状変化による可逆成分）。この周期成分 θosc(φ) は
#   ループ形状から事前に計算できるので、差し引いた「周回平均姿勢」θ̄ でフィードバックする。
inc = -np.einsum("ik,ki->i", Gl, np.array([loop_d(p) for p in ph]).T)   # dθ/dφ (L=0)
geo = np.concatenate([[0], np.cumsum((inc[1:] + inc[:-1]) / 2 * np.diff(ph))])
osc = geo - ph / (2 * np.pi) * dth_c        # 周期成分
osc -= osc.mean()


def theta_osc(phi):
    return np.interp(phi % (2 * np.pi), ph, osc)


th, phi = theta_osc(0.0), 0.0                # 周回平均姿勢 0 から開始
th_a2 = np.zeros_like(tc); th_bar = np.zeros_like(tc); fa = np.zeros_like(tc)
for i, ti in enumerate(tc):
    L = TAU_D * ti
    tb = th - theta_osc(phi)
    f = np.clip((-KP * tb - L * inv_A_mean) / dth_c, -F_MAX, F_MAX)
    th_a2[i], th_bar[i], fa[i] = th, tb, f
    phid = 2 * np.pi * f
    th += dt * amb_rate(phi, phid, L)
    phi += dt * phid

h_rw = B0[W] * OMEGA_MAX
h_amb = F_MAX * abs(dth_c) / inv_A_mean
print("\n=== Part C: 外乱トルク {:.3f} N m 下の姿勢保持 ===".format(TAU_D))
print(f"RW    の角運動量容量 h = I_w ω_max          = {h_rw:.3f} kg m²/s → 保持限界 {h_rw / TAU_D:.0f} s")
sat = tc < 0.9 * h_amb / TAU_D
print(f"AMBAC 保持中: 周回平均姿勢 |θ̄| ≤ {D(np.abs(th_bar[sat]).max()):.2f}°, 瞬時姿勢の揺れ ±{D(np.abs(th_a2[sat]).max()):.1f}°")
print(f"AMBAC の角運動量容量 h = f_max |Δθc| / <1/A> = {h_amb:.3f} kg m²/s → 保持限界 {h_amb / TAU_D:.0f} s")
np.savetxt(DATA / "sim03_disturbance.csv", np.c_[tc, TAU_D * tc, th_r, wr, th_a2, th_bar, fa][::20],
           delimiter=",", comments="",
           header="t[s],L[kg m^2/s],theta_rw[rad],wheel_rate[rad/s],theta_ambac[rad],theta_ambac_cycle_avg[rad],loop_freq[Hz]")

fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
ax[0].plot(tc, D(th_r), "C3", label="reaction wheel")
ax[0].plot(tc, D(th_a2), "C0", lw=.3, alpha=.5, label="AMBAC θ (instantaneous)")
ax[0].plot(tc, D(th_bar), "C0", lw=1.5, label="AMBAC θ (cycle-averaged)")
ax[0].set_ylabel("θ [deg]"); ax[0].set_ylim(-60, 60); ax[0].legend(fontsize=8, loc="lower right")
ax[1].plot(tc, wr * 60 / (2 * np.pi), "C3"); ax[1].axhline(OMEGA_MAX * 60 / (2 * np.pi), color="C3", ls=":")
ax[1].set_ylabel("wheel [rpm]")
ax[2].plot(tc, fa, "C0"); ax[2].axhline(-F_MAX, color="C0", ls=":"); ax[2].axhline(F_MAX, color="C0", ls=":")
ax[2].set_ylabel("loop freq [Hz]"); ax[2].set_xlabel("time [s]")
for a in ax: a.grid(alpha=.3)
for a, h in ((ax[0], h_amb), (ax[0], h_rw)):
    a.axvline(h / TAU_D, color="gray", ls="--", lw=.8)
fig.suptitle(f"Sim03-C: attitude hold under constant disturbance {TAU_D} N·m")
fig.tight_layout(); fig.savefig(FIG / "sim03_disturbance.png", dpi=150)
