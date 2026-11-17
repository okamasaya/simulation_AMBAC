"""Sim06: 地上を動く機体（戦車型台車 + 腕）— 地上版 AMBAC.

宇宙では角運動量が保存されたが、地上では接触反力（垂直力・摩擦）が働く。
ただし接触反力には限界がある:
  ・ZMP（反力の作用点）は履帯の接地範囲 [−a, a] の中にしか来られない → 外れると転倒
  ・水平反力は摩擦 μN までしか出せない → 超えると滑り
腕の動きの反作用がこれらの限界を超えるかどうかが、地上版 AMBAC の問題になる。

Part A: 荷物運び動作（前方の地面から拾い、頭上を通して後部デッキへ置く）の速さと転倒・滑り
        ＋ スケーリング（相似拡大・重力）: 安全な最短時間 ∝ √(s/g)（フルード則）
Part B: 地上版の能動的質量バランス — 腕の軌道を最適化して、転ばずに最短時間で運ぶ
Part C: 上から見た旋回（ヨー）: 腕の振りで台車が回る／滑るか。地球・火星・月の重力で比較
        摩擦があると「速く振って、ゆっくり戻す」非対称な往復で、関節 1 つでも台車が回る（スティックスリップ）
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize

from common import DATA, FIG
from ambac.ground import TankArm, min_jerk

D = np.degrees
G = {"Earth": 9.80665, "Mars": 3.721, "Moon": 1.625}
MU = 0.6                                   # 履帯（ゴム）と路面の摩擦係数の目安
MARGIN = 0.05                              # ZMP の安全余裕 [m]（a に対する比で相似拡大）
robot = TankArm()
Q0 = np.radians([-20.0, -50.0])            # 前方の地面の荷物をつかむ姿勢
Q1 = np.radians([130.0, 60.0])             # 後部デッキに置く姿勢
K = 401
TAU = np.linspace(0, 1, K)
Q_NAIVE = Q0 + (Q1 - Q0) * min_jerk(TAU)[:, None]


def margins(r, q_path, T, margin):
    """転倒余裕（>0 で安全）, 滑り余裕（>0 で安全）"""
    p, N, Fx = r.reactions(q_path, T)
    tip = (r.half_track - margin) - np.abs(p)
    slip = MU * N - np.abs(Fx)
    return tip.min(), (slip / (MU * N.max())).min()


def t_min(r, q_path, margin, kind="both"):
    """安全に実行できる最短時間（時間スケーリングで ZMP は単調に悪化するので二分法）."""
    def f(T):
        a, b = margins(r, q_path, T, margin)
        return {"tip": a, "slip": b, "both": min(a, b)}[kind]
    if f(0.2 * np.sqrt(r.l1 / 0.5)) > 0:
        return 0.2
    return brentq(f, 0.2 * np.sqrt(r.l1 / 0.5) * 0.5, 100.0, xtol=1e-4)


# =============================================================================
# Part A
# =============================================================================
print("=== Part A: 荷物運び（前方地面 → 頭上 → 後部デッキ）, 総重量 "
      f"{robot.masses.sum():.0f} kg, 荷物 {robot.payload:.0f} kg ===")
T_tip = t_min(robot, Q_NAIVE, MARGIN, "tip")
T_slip = t_min(robot, Q_NAIVE, MARGIN, "slip")
print(f"最小躍度の単純軌道: 転倒しない最短 {T_tip:.2f} s, 滑らない最短 {T_slip:.2f} s（μ = {MU}）")
T_SHOW = [4.0, 2.0, T_tip, 1.2]
rowsA = []
for T in T_SHOW:
    p, N, Fx = robot.reactions(Q_NAIVE, T)
    rowsA.append(np.c_[TAU * T, p, N, Fx])
    print(f"  T = {T:4.2f} s: ZMP [{p.min():+.3f}, {p.max():+.3f}] m（接地範囲 ±{robot.half_track}）,"
          f" 必要摩擦係数 max {np.abs(Fx / N).max():.2f}")
np.savez(DATA / "sim06_partA_naive.npz", T=np.array(T_SHOW), data=np.array(rowsA))  # data: (T, τ, [t, p, N, Fx])

# スケーリング（相似拡大・重力）
print("\n--- スケーリング: 安全な最短時間（転倒・滑りの両方）---")
S_LIST = [1, 2, 5, 10]
scal = []
for gname, g in G.items():
    for s in S_LIST:
        r = robot.scaled(s, g)
        T = t_min(r, Q_NAIVE, MARGIN * s)
        scal.append((s, g, T, T / np.sqrt(s / (g / G["Earth"]))))
    print(f"{gname:5s}: " + ", ".join(f"s={s:>2d}: {x[2]:5.2f} s" for s, x in zip(S_LIST, scal[-len(S_LIST):])))
scal = np.array(scal)
print(f"T_min / √(s · g_E/g) = {scal[:, 3].min():.3f} 〜 {scal[:, 3].max():.3f}（一定ならフルード則が厳密に成立）")
np.savetxt(DATA / "sim06_scaling.csv", scal, delimiter=",", comments="",
           header="scale,g[m/s2],T_min[s],T_min/sqrt(s*gE/g)")

# =============================================================================
# Part B: 軌道最適化（地上版の能動的質量バランス）
# =============================================================================
NB = 4                                       # 関節ごとの基底関数の数
BASIS = np.array([(TAU * (1 - TAU))**3 * np.sin((k + 1) * np.pi * TAU) for k in range(NB)]).T * 64
QD_MAX, QDD_MAX = 6.0, 60.0                  # 関節速度・加速度の上限（アクチュエータ制限の目安）
CK = np.arange(0, K, 2)                      # 制約を課すサンプル


def path(c):
    return Q_NAIVE + BASIS @ c.reshape(NB, 2)


def cons(x, r=robot, margin=MARGIN):
    T, c = x[0], x[1:]
    qp = path(c)
    p, N, Fx = r.reactions(qp, T)
    h = 1 / (K - 1)
    qd = np.gradient(qp, h, axis=0) / T; qdd = np.gradient(np.gradient(qp, h, axis=0), h, axis=0) / T**2
    a = r.half_track - margin
    return np.concatenate([(a - p)[CK], (a + p)[CK], (MU * N - Fx)[CK] / 100, (MU * N + Fx)[CK] / 100,
                           (QD_MAX - np.abs(qd))[::2].ravel(), (QDD_MAX - np.abs(qdd))[::2].ravel() / 10])


T_naive = t_min(robot, Q_NAIVE, MARGIN)
_, cn = 0, cons(np.r_[T_naive, np.zeros(2 * NB)])
best = None
for seed in range(6):
    rng = np.random.default_rng(seed)
    x0 = np.r_[T_naive * 1.05, rng.normal(0, 0.3 if seed else 0.0, 2 * NB)]
    res = minimize(lambda x: x[0], x0, method="SLSQP", constraints=[{"type": "ineq", "fun": cons}],
                   bounds=[(0.3, 5.0)] + [(-3, 3)] * (2 * NB), options=dict(maxiter=300, ftol=1e-6))
    if res.success and cons(res.x).min() > -1e-4 and (best is None or res.x[0] < best.x[0]):
        best = res
T_opt = best.x[0]; Q_OPT = path(best.x[1:])
print("\n=== Part B: 腕の軌道最適化（転倒・滑り・関節速度/加速度の制約下で最短時間）===")
print(f"単純軌道（最小躍度）: {T_naive:.2f} s → 最適化軌道: {T_opt:.2f} s（{(1 - T_opt / T_naive) * 100:.0f}% 短縮）")
p_n, N_n, F_n = robot.reactions(Q_NAIVE, T_opt)
p_o, N_o, F_o = robot.reactions(Q_OPT, T_opt)
print(f"同じ {T_opt:.2f} s で実行した場合の ZMP: 単純 [{p_n.min():+.2f}, {p_n.max():+.2f}] m → "
      f"最適化 [{p_o.min():+.2f}, {p_o.max():+.2f}] m（接地範囲 ±{robot.half_track}）")
np.savez(DATA / "sim06_partB_opt.npz", tau=TAU, q_naive=Q_NAIVE, q_opt=Q_OPT, T_naive=T_naive, T_opt=T_opt)

# 月面での同じ比較
moon = robot.scaled(1.0, G["Moon"])
T_naive_m = t_min(moon, Q_NAIVE, MARGIN)
best_m = None
for seed in range(6):
    rng = np.random.default_rng(seed)
    x0 = np.r_[T_naive_m * 1.05, rng.normal(0, 0.3 if seed else 0.0, 2 * NB)]
    res = minimize(lambda x: x[0], x0, method="SLSQP",
                   constraints=[{"type": "ineq", "fun": lambda x: cons(x, moon)}],
                   bounds=[(0.3, 10.0)] + [(-3, 3)] * (2 * NB), options=dict(maxiter=300, ftol=1e-6))
    if res.success and cons(res.x, moon).min() > -1e-4 and (best_m is None or res.x[0] < best_m.x[0]):
        best_m = res
print(f"月面: 単純軌道 {T_naive_m:.2f} s → 最適化軌道 {best_m.x[0]:.2f} s")

# ---- 図 A/B ---------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
for k, (T, d) in enumerate(zip(T_SHOW, rowsA)):
    ax[0, 0].plot(d[:, 0] / T, d[:, 1], label=f"T = {T:.2f} s")
ax[0, 0].axhspan(-robot.half_track, robot.half_track, color="C2", alpha=.08)
for sgn in (-1, 1):
    ax[0, 0].axhline(sgn * robot.half_track, color="k", lw=.8)
ax[0, 0].set_ylim(-1.0, 1.0); ax[0, 0].set_xlabel("normalized time τ"); ax[0, 0].set_ylabel("ZMP p [m]")
ax[0, 0].set_title("Part A: ZMP during naive (min-jerk) motion\n(shaded: track contact = no tip-over)")
ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=.3)
for gname, mk in zip(G, "osd"):
    sel = np.isclose(scal[:, 1], G[gname])
    ax[0, 1].loglog(scal[sel, 0], scal[sel, 2], mk + "-", label=gname)
ss = np.array([1, 10])
ax[0, 1].loglog(ss, scal[0, 2] * np.sqrt(ss), "k:", label="∝ √s (Froude)")
ax[0, 1].set_xlabel("length scale s"); ax[0, 1].set_ylabel("min safe time T [s]")
ax[0, 1].set_title("Part A: scaling of safe motion time"); ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=.3, which="both")


def draw(a, q_path, color, label):
    pos, _ = robot.bodies(q_path)
    S = np.array(robot.shoulder)
    for i in range(0, K, 40):
        e = pos[i, 1] * 2 - S
        a.plot([S[0], e[0], pos[i, 3, 0]], [S[1], e[1], pos[i, 3, 1]], "-", color=color, alpha=.35, lw=1)
    a.plot(pos[:, 3, 0], pos[:, 3, 1], "-", color=color, lw=2, label=label)


draw(ax[1, 0], Q_NAIVE, "C0", "naive (min-jerk)")
draw(ax[1, 0], Q_OPT, "C3", "optimized")
a_ = robot.half_track
ax[1, 0].add_patch(plt.Rectangle((-a_, 0), 2 * a_, 0.5, fc="gray", alpha=.3))
ax[1, 0].plot([-1.2, 1.2], [0, 0], "k-", lw=1)
ax[1, 0].set_aspect("equal"); ax[1, 0].set_xlim(-1.3, 1.1); ax[1, 0].set_ylim(-0.1, 1.9)
ax[1, 0].set_title("Part B: arm paths (payload trajectory)"); ax[1, 0].legend(fontsize=8)
ax[1, 1].plot(TAU, p_n, "C0", label=f"naive, T = {T_opt:.2f} s (tips over)")
ax[1, 1].plot(TAU, p_o, "C3", label=f"optimized, T = {T_opt:.2f} s")
for sgn in (-1, 1):
    ax[1, 1].axhline(sgn * robot.half_track, color="k", lw=.8)
    ax[1, 1].axhline(sgn * (robot.half_track - MARGIN), color="k", lw=.5, ls=":")
ax[1, 1].set_ylim(-1.0, 1.0); ax[1, 1].set_xlabel("normalized time τ"); ax[1, 1].set_ylabel("ZMP p [m]")
ax[1, 1].set_title(f"Part B: ZMP, naive {T_naive:.2f} s → optimized {T_opt:.2f} s")
ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "sim06_tipover.png", dpi=150)

# =============================================================================
# Part C: 上から見た旋回（ヨー）とスティックスリップ
# =============================================================================
#   台車の中心に旋回関節（ターレット）があり、腕（棒 7 kg, 1 m）＋荷物 10 kg を水平に振る。
#   台車の並進は止まっているとし、ヨー回転だけを考える（台車中心まわり）。
#   L = A θ' + B q',  A = I_b + I_a,  B = I_a（一定）,  dL/dt = τ_fric
#   静止摩擦: 台車が止まっているとき、必要な摩擦トルク |B q''| ≤ τ_f なら止まったまま
#   動摩擦  : 滑っているときは τ_fric = −τ_f sign(θ')
#   τ_f = μ N r̄（r̄: 接地面上の点の台車中心からの平均距離）
L_T, W_T = 0.8, 0.6                                    # 接地面（2 本の履帯をまとめた長方形）
M_TOT = robot.masses.sum()
I_B = robot.base_mass * (L_T**2 + W_T**2) / 12
L_ARM, M_ARM, M_PAY = 1.0, robot.m1 + robot.m2, robot.payload
I_A = M_ARM * L_ARM**2 / 3 + M_PAY * L_ARM**2
A_Y, B_Y = I_B + I_A, I_A
xx, yy = np.meshgrid(np.linspace(-L_T / 2, L_T / 2, 201), np.linspace(-W_T / 2, W_T / 2, 151))
R_BAR = np.hypot(xx, yy).mean()
SWING = np.radians(90.0)
T_SLOW = 3.0


def yaw_cycle(T_fast, g, mu=MU, n_cycles=1, dt=2e-4, record=False):
    """速く振る（T_fast）→ ゆっくり戻す（T_SLOW）を n 回。戻り値: 正味のヨー角 [rad]"""
    tau_f = mu * M_TOT * g * R_BAR
    Tc = T_fast + T_SLOW
    th = thd = 0.0; hist = []
    t = 0.0
    while t < n_cycles * Tc:
        tc = t % Tc
        if tc < T_fast:
            s = tc / T_fast; qdd = SWING * (60 * s - 180 * s**2 + 120 * s**3) / T_fast**2
        else:
            s = (tc - T_fast) / T_SLOW; qdd = -SWING * (60 * s - 180 * s**2 + 120 * s**3) / T_SLOW**2
        drive = -B_Y * qdd                     # 摩擦がなければ胴体に加わる反作用トルク
        if abs(thd) < 1e-9 and abs(drive) <= tau_f:
            thdd = 0.0; thd = 0.0              # 静止摩擦で止まっている
        else:
            dirn = np.sign(thd) if abs(thd) >= 1e-9 else np.sign(drive)
            thdd = (drive - tau_f * dirn) / A_Y
            if abs(thd) >= 1e-9 and np.sign(thd + dt * thdd) != np.sign(thd):
                thd = 0.0; thdd = 0.0          # 滑りが止まる
        thd += dt * thdd; th += dt * thd
        if record:
            hist.append((t, th))
        t += dt
    return (th, np.array(hist)) if record else th


print("\n=== Part C: 腕の水平振り（90°、速く振って 3 s で戻す）による台車のヨー ===")
print(f"ヨー慣性: 台車 {I_B:.1f}, 腕+荷物 {I_A:.1f} kg m², 平均半径 r̄ = {R_BAR:.3f} m")
for gname, g in G.items():
    tau_f = MU * M_TOT * g * R_BAR
    T_th = np.sqrt(B_Y * SWING * 5.7735 / tau_f)       # min-jerk の最大加速度 5.77 Δq/T²
    print(f"  {gname:5s}: 摩擦トルク上限 {tau_f:6.1f} N m → 振りが {T_th:.2f} s より速いと台車が滑る")
TF = np.geomspace(0.25, 3.0, 25)
net = {gname: np.array([yaw_cycle(T, g) for T in TF]) for gname, g in G.items()}
net0 = np.array([yaw_cycle(T, 1.0, mu=0.0) for T in TF[::6]])
print(f"  摩擦ゼロ（宇宙と同じ）: 1 往復の正味ヨー max |{D(np.abs(net0).max()):.1e}|°（関節 1 つでは回らない）")
for gname in G:
    print(f"  {gname:5s}: 1 往復の正味ヨー  " + ", ".join(f"T_fast={T:.2f}s: {D(v):+6.2f}°"
                                          for T, v in zip(TF[::6], net[gname][::6])))
np.savetxt(DATA / "sim06_yaw_ratchet.csv", np.c_[TF, *[net[k] for k in G]], delimiter=",", comments="",
           header="T_fast[s]," + ",".join(f"net_yaw_{k}[rad]" for k in G))
_, hm = yaw_cycle(0.6, G["Moon"], n_cycles=5, record=True)
_, he = yaw_cycle(0.6, G["Earth"], n_cycles=5, record=True)
print(f"  例: 0.6 s で振り 3 s で戻す往復を 5 回 → 地球 {D(he[-1, 1]):+.1f}°, 月 {D(hm[-1, 1]):+.1f}°")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
for gname in G:
    ax[0].plot(TF, D(net[gname]), "o-", ms=3, label=gname)
ax[0].axhline(0, color="k", lw=.6)
ax[0].set_xscale("log"); ax[0].set_xlabel("fast-stroke time T_fast [s]  (return: 3 s)")
ax[0].set_ylabel("net base yaw per cycle [deg]")
ax[0].set_title("Part C: stick–slip ratchet (single turret joint)"); ax[0].legend(); ax[0].grid(alpha=.3, which="both")
ax[1].plot(he[::50, 0], D(he[::50, 1]), label="Earth"); ax[1].plot(hm[::50, 0], D(hm[::50, 1]), label="Moon")
ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("base yaw [deg]")
ax[1].set_title("5 cycles, T_fast = 0.6 s"); ax[1].legend(); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "sim06_yaw_ratchet.png", dpi=150)
