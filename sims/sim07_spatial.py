"""Sim07: 3 次元化 — 球面座標の肩を持つ腕による姿勢変化（宇宙）と転倒（地上）.

腕の向きは肩を原点とする球面座標 (ψ: 方位角, α: 仰角) で指令する。
胴体の姿勢は回転行列（SO(3)）で扱う（オイラー角の特異点を避けるため）。

Part A（宇宙・自由浮遊）
  A1: 腕で円錐を描く（宇宙飛行士の「腕回し」）→ 1 周あたりの胴体回転（大きさ・回転軸）
      小さな円錐では回転角 ∝ 半頂角² （2 次効果, Sim05 の議論の 3 次元版）
  A2: 円錐の軸の向きによる違い（前方・横・上方）
  A3: 球面座標の特異点（極 = 真上）付近を通るループでの関節速度の発散（ジンバルロック）
  A4: 回転の非可換性 — 「ヨー → ピッチ」と「ピッチ → ヨー」で最終姿勢が違う
Part B（地上・戦車型台車）
  B1: 静的な定格荷重マップ（腕の向きごとの最大荷重）— クレーンの定格荷重表に相当
  B2: 腕を水平に 180° 旋回（前 → 左 → 後）させるときの転倒限界と、腕の上げ下げを最適化した旋回
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize
from scipy.spatial.transform import Rotation

from common import DATA, FIG
from ambac.spatial import Arm3D, FloatingBody3D, TankArm3D, sph, sph_jac, to_sph

D = np.degrees
RAD = np.radians

# =============================================================================
# Part A: 宇宙（人間大）
# =============================================================================
# 胴体 60 kg（奥行 0.25 × 幅 0.4 × 高さ 1.0 m）, 右肩 (0, −0.2, 0.35), 腕 4 kg・0.7 m
body = FloatingBody3D(60.0, (0.25, 0.4, 1.0), Arm3D(shoulder=(0.0, -0.2, 0.35), mass=4.0, length=0.7))


def frame(axis):
    a = axis / np.linalg.norm(axis)
    t = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(t, a); e1 /= np.linalg.norm(e1); e2 = np.cross(a, e1)
    return a, e1, e2


def cone(axis, beta, T, direction=+1):
    """腕の先端方向 u が軸 axis のまわりに半頂角 beta の円錐を描く。関節 (ψ, α) の軌道を返す."""
    a, e1, e2 = frame(np.asarray(axis, float))
    w = direction * 2 * np.pi / T

    def u(t):
        ph = w * t
        return np.cos(beta) * a + np.sin(beta) * (np.cos(ph) * e1 + np.sin(ph) * e2)

    def ud(t):
        ph = w * t
        return np.sin(beta) * w * (-np.sin(ph) * e1 + np.cos(ph) * e2)

    def q(t):
        return np.array(to_sph(u(t)))

    def qd(t):
        psi, al = q(t)
        J = sph_jac(psi, al)                     # (3,2)
        return np.linalg.lstsq(J, ud(t), rcond=None)[0]

    return q, qd


def holonomy(axis, beta, direction=+1, T=2.0, n=3000, n_cycles=1):
    q, qd = cone(axis, beta, T, direction)
    Rs = body.integrate(q, qd, n_cycles * T, n=n * n_cycles)
    return Rs[-1].as_rotvec()


def ax_dir(psi_deg, alpha_deg):
    return sph(RAD(psi_deg), RAD(alpha_deg))


print("=== Part A1: 前方を軸に腕で円錐を描く（右腕、反時計回り = 軸方向から見て）===")
betas = np.array([5, 10, 20, 30, 45, 60, 70])
rows = []
for b in betas:
    rv = holonomy(ax_dir(0, 0), RAD(b))
    rows.append([b, *rv])
    print(f"  半頂角 {b:3d}°: 1 周あたり胴体回転 {D(np.linalg.norm(rv)):7.3f}°, 回転軸 {np.round(rv / np.linalg.norm(rv), 3)}")
rows = np.array(rows)
k = np.polyfit(np.log(RAD(betas[:3])), np.log(np.linalg.norm(rows[:3, 1:], axis=1)), 1)[0]
print(f"  小さい円錐での指数: 回転角 ∝ β^{k:.2f}（理論: 2 = 幾何学的位相の 2 次効果）")
rv_rev = holonomy(ax_dir(0, 0), RAD(45), direction=-1)
print(f"  逆回り（半頂角 45°）: {np.round(D(rv_rev), 3)}°（順回り {np.round(D(rows[4, 1:]), 3)}° と符号反転）")
np.savetxt(DATA / "sim07_cone_forward.csv", rows, delimiter=",", comments="",
           header="beta[deg],rotvec_x[rad],rotvec_y[rad],rotvec_z[rad]")

print("\n=== Part A2: 円錐の軸の向き（半頂角 45°）===")
AXES = {"forward (ψ=0, α=0)": ax_dir(0, 0), "right side (ψ=-90, α=0)": ax_dir(-90, 0),
        "up (α=90): azimuth sweep": np.array([0, 0, 1.0]), "front-up (ψ=0, α=45)": ax_dir(0, 45)}
a2 = {}
for name, ax_ in AXES.items():
    rv = holonomy(ax_, RAD(45))
    a2[name] = rv
    print(f"  {name:26s}: {D(np.linalg.norm(rv)):6.2f}°/周, 回転軸 {np.round(rv / np.linalg.norm(rv), 2)}"
          f"（円錐の軸との角度 {D(np.arccos(abs(rv @ ax_) / np.linalg.norm(rv))):.0f}°）")

print("\n=== Part A3: 球面座標の特異点（極 = 真上）の近くを通る円錐（T = 2 s）===")
a3 = []
for dpole in [30, 10, 3, 1, 0.3]:
    beta = RAD(30); axis = ax_dir(0, 90 - 30 - dpole)    # 円錐の最上点と真上との角度 = dpole
    q, qd = cone(axis, beta, 2.0)
    ts = np.linspace(0, 2.0, 4001)
    vmax = np.abs(np.array([qd(t) for t in ts])).max(0)
    tipspeed = np.sin(beta) * 2 * np.pi / 2.0
    a3.append([dpole, *vmax])
    print(f"  極との最小角距離 {dpole:5.1f}°: 最大関節速度 ψ' {vmax[0]:8.2f} rad/s, α' {vmax[1]:.2f} rad/s"
          f"（腕先端の角速度は一定 {tipspeed:.2f} rad/s）")
np.savetxt(DATA / "sim07_singularity.csv", np.array(a3), delimiter=",", comments="",
           header="min_angle_to_pole[deg],max_psi_dot[rad/s],max_alpha_dot[rad/s]")

print("\n=== Part A4: 回転の非可換性 ===")
yaw_axis, pitch_axis = np.array([0, 0, 1.0]), ax_dir(-90, 0)
rv_y1 = holonomy(yaw_axis, RAD(45)); rv_p1 = holonomy(pitch_axis, RAD(45))
n_y = int(round(RAD(30) / np.linalg.norm(rv_y1))); n_p = int(round(RAD(30) / np.linalg.norm(rv_p1)))
qy, qyd = cone(yaw_axis, RAD(45), 2.0); qp, qpd = cone(pitch_axis, RAD(45), 2.0)


def seq(first, second):
    R = Rotation.identity()
    for (q_, qd_, n) in (first, second):
        R = body.integrate(q_, qd_, 2.0 * n, n=1500 * n, R0=R)[-1]
    return R


Y = (qy, qyd, n_y); P = (qp, qpd, n_p)
R_yp = seq(Y, P); R_py = seq(P, Y)
diff = (R_yp.inv() * R_py).magnitude()
print(f"  ヨー用円錐 {n_y} 周（約 {D(np.linalg.norm(rv_y1)) * n_y:.0f}°）とピッチ用円錐 {n_p} 周（約 {D(np.linalg.norm(rv_p1)) * n_p:.0f}°）")
print(f"  ヨー→ピッチ: 回転ベクトル {np.round(D(R_yp.as_rotvec()), 2)}°")
print(f"  ピッチ→ヨー: 回転ベクトル {np.round(D(R_py.as_rotvec()), 2)}°")
print(f"  2 つの最終姿勢の差: {D(diff):.2f}°（平面ではループの順序を入れ替えても結果は同じ）")

# ---- 図 A ---------------------------------------------------------------------
fig = plt.figure(figsize=(15, 4.6))
ax0 = fig.add_subplot(1, 3, 1)
ax0.loglog(betas, D(np.linalg.norm(rows[:, 1:], axis=1)), "o-", label="simulation")
ax0.loglog(betas, D(np.linalg.norm(rows[0, 1:])) * (betas / betas[0])**2, "k:", label="∝ β²")
ax0.set_xlabel("cone half-angle β [deg]"); ax0.set_ylabel("body rotation per cycle [deg]")
ax0.set_title("A1: arm cone about forward axis"); ax0.legend(); ax0.grid(alpha=.3, which="both")
ax1 = fig.add_subplot(1, 3, 2, projection="3d")
q, _ = cone(ax_dir(0, 0), RAD(45), 2.0)
tip = np.array([np.array(body.arm.shoulder) + body.arm.length * sph(*q(t)) for t in np.linspace(0, 2, 100)])
ax1.plot(*tip.T, "C3", lw=2, label="hand path (cone)")
S = np.array(body.arm.shoulder)
for t in np.linspace(0, 2, 9)[:-1]:
    e = S + body.arm.length * sph(*q(t)); ax1.plot(*np.c_[S, e], "C3", alpha=.3)
bx, by, bz = np.array(body.base_dims) / 2
for xs, ys in (([-bx, bx], [-by, -by]), ([-bx, bx], [by, by]), ([-bx, -bx], [-by, by]), ([bx, bx], [-by, by])):
    for zz in (-bz, bz):
        ax1.plot(xs, ys, [zz, zz], "k-", lw=.6)
for xx in (-bx, bx):
    for yy in (-by, by):
        ax1.plot([xx, xx], [yy, yy], [-bz, bz], "k-", lw=.6)
rv = rows[4, 1:]; ax1.quiver(0, 0, 0, *(rv / np.linalg.norm(rv) * 0.6), color="C0", lw=2, label="net body rotation axis")
ax1.set_box_aspect((1, 1, 1.4)); ax1.set_xlim(-0.6, 0.9); ax1.set_ylim(-0.9, 0.6); ax1.set_zlim(-0.6, 0.9)
ax1.set_title(f"A1: β=45°, {D(np.linalg.norm(rv)):.1f}°/cycle"); ax1.legend(fontsize=7, loc="upper left")
ax2 = fig.add_subplot(1, 3, 3)
a3 = np.array(a3)
ax2.loglog(a3[:, 0], a3[:, 1], "o-", label="azimuth rate ψ'")
ax2.loglog(a3[:, 0], a3[:, 2], "s-", label="elevation rate α'")
ax2.axhline(np.sin(RAD(30)) * np.pi, color="k", ls=":", label="hand angular speed (const)")
ax2.set_xlabel("closest angle of the loop to the pole [deg]"); ax2.set_ylabel("max joint speed [rad/s]")
ax2.set_title("A3: spherical-coordinate singularity (gimbal lock)"); ax2.legend(fontsize=8); ax2.grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig(FIG / "sim07_space.png", dpi=150)

# =============================================================================
# Part B: 地上（戦車型台車 + 球面座標の腕）
# =============================================================================
G = {"Earth": 9.80665, "Moon": 1.625}
MU, MARGIN = 0.6, 0.05
tank = TankArm3D()
print(f"\n=== Part B1: 静的な定格荷重マップ（台車 {tank.base_mass:.0f} kg, 接地面 {2 * tank.track_half[0]:.1f}×{2 * tank.track_half[1]:.1f} m,"
      f" 腕 {tank.arm.mass:.0f} kg・{tank.arm.length:.1f} m）===")
PSI = np.linspace(-180, 180, 145); ALP = np.linspace(-40, 85, 51)
chart = np.zeros((len(ALP), len(PSI)))
a_, b_ = tank.track_half[0] - MARGIN, tank.track_half[1] - MARGIN
arm0 = Arm3D(tank.arm.shoulder, tank.arm.mass, tank.arm.length, payload=0.0)
for i, al in enumerate(ALP):
    for j, ps in enumerate(PSI):
        p, m, _ = arm0.points(RAD([ps, al]))
        M0 = tank.base_mass + m.sum()
        c = (tank.base_mass * np.array(tank.base_com[:2]) + (m[:, None] * p[:, :2]).sum(0)) / M0
        t = np.array(tank.arm.shoulder[:2]) + tank.arm.length * sph(RAD(ps), RAD(al))[:2]
        lam = 1.0
        for k_, lim in ((0, a_), (1, b_)):
            dv = t[k_] - c[k_]
            if abs(dv) > 1e-12:
                l = (np.sign(dv) * lim - c[k_]) / dv
                if l > 0:
                    lam = min(lam, l)
        chart[i, j] = np.inf if lam >= 1 else M0 * lam / (1 - lam)
for ps in (0, 45, 90, 180):
    j = np.argmin(abs(PSI - ps)); i = np.argmin(abs(ALP - 0))
    print(f"  腕を水平に方位 {ps:4d}°（{'前' if ps == 0 else '斜め前' if ps == 45 else '横' if ps == 90 else '後'}）: 最大荷重 {chart[i, j]:.1f} kg")
np.savez(DATA / "sim07_load_chart.npz", psi_deg=PSI, alpha_deg=ALP, max_payload_kg=chart)

# ---- B2: 水平旋回 ----------------------------------------------------------------
K = 301
TAU = np.linspace(0, 1, K)
mj = 10 * TAU**3 - 15 * TAU**4 + 6 * TAU**5
PSI_PATH = RAD(180) * mj                              # 前 → 左 → 後


def make_path(alpha_tau, psi_tau=PSI_PATH):
    return np.stack([psi_tau, alpha_tau], -1)


QD_MAX, QDD_MAX = 6.0, 40.0                # 関節速度・加速度の上限（すべての方策に共通）
ALPHA_MIN = RAD(-35)                       # 荷物が地面・台車に当たらない仰角の下限


def safety(r, qp, T):
    """転倒・滑り・関節速度/加速度の余裕（すべて > 0 で実行可能）"""
    zmp, N, F = r.zmp_path(qp, T)
    tip = r.margin(zmp) - MARGIN
    slip = (MU * N - np.linalg.norm(F, axis=1)) / (MU * N.max())
    h = 1 / (len(qp) - 1)
    qd = np.gradient(qp, h, axis=0) / T; qdd = np.gradient(np.gradient(qp, h, axis=0), h, axis=0) / T**2
    joint = np.minimum((QD_MAX - np.abs(qd)).min(1) / QD_MAX, (QDD_MAX - np.abs(qdd)).min(1) / QDD_MAX)
    return np.minimum(np.minimum(tip, slip), joint), zmp


def t_min(r, qp):
    f = lambda T: safety(r, qp, T)[0].min()
    return brentq(f, 0.3, 60.0, xtol=1e-3) if f(0.3) < 0 else 0.3


NB = 4
BAS = np.array([np.sin(np.pi * TAU) ** 2 * np.sin((k + 1) * np.pi * TAU / 1) if k else np.sin(np.pi * TAU) ** 2
                for k in range(NB)]).T
BAS_PSI = np.array([(TAU * (1 - TAU))**3 * np.sin((k + 1) * np.pi * TAU) for k in range(3)]).T * 64
def opt_path(x):
    return make_path(BAS @ x[:NB], PSI_PATH + BAS_PSI @ x[NB:])


def cons(x, r):
    T = x[0]; qp = opt_path(x[1:])
    s, _ = safety(r, qp, T)
    return np.concatenate([s[::3], RAD(85) - qp[::3, 1], qp[::3, 1] - ALPHA_MIN])


print("\n=== Part B2: 腕（荷物 10 kg）を水平に 180° 旋回（前 → 左 → 後）===")
print(f"  制約: 転倒（余裕 {MARGIN} m）, 滑り（μ={MU}）, 関節速度 {QD_MAX} rad/s, 加速度 {QDD_MAX} rad/s², 仰角 ≥ {D(ALPHA_MIN):.0f}°")
res_B = {}
for gname, g in G.items():
    r = TankArm3D(g=g)
    flat = make_path(np.zeros(K))
    raised = make_path(RAD(60) * np.sin(np.pi * TAU) ** 2)
    lowered = make_path(ALPHA_MIN * np.sin(np.pi * TAU) ** 2)
    T_flat, T_raised, T_low = t_min(r, flat), t_min(r, raised), t_min(r, lowered)
    best = None
    inits = [np.zeros(NB + 3), np.r_[RAD(60), np.zeros(NB + 2)], np.r_[ALPHA_MIN, np.zeros(NB + 2)]]
    for seed in range(9):
        rng = np.random.default_rng(seed)
        x0 = np.r_[max(T_flat, T_raised, T_low), inits[seed % 3] + rng.normal(0, 0.2 if seed >= 3 else 0.0, NB + 3)]
        res = minimize(lambda x: x[0], x0, method="SLSQP", constraints=[{"type": "ineq", "fun": lambda x: cons(x, r)}],
                       bounds=[(0.3, 60)] + [(-1.5, 1.5)] * NB + [(-1, 1)] * 3, options=dict(maxiter=300))
        if res.success and cons(res.x, r).min() > -1e-4 and (best is None or res.x[0] < best.x[0]):
            best = res
    qo = opt_path(best.x[1:])
    res_B[gname] = dict(T_flat=T_flat, T_raised=T_raised, T_low=T_low, T_opt=best.x[0], q_opt=qo,
                        zmp_flat=safety(r, flat, T_flat)[1], zmp_opt=safety(r, qo, best.x[0])[1],
                        zmp_flat_at_opt=safety(r, flat, best.x[0])[1])
    print(f"  {gname:5s}: 水平のまま {T_flat:.2f} s / 途中で 60° 持ち上げ {T_raised:.2f} s / 途中で 35° 下げる {T_low:.2f} s"
          f" / 最適化 {best.x[0]:.2f} s"
          f"（仰角の最大 {D(qo[:, 1].max()):.0f}°, 最小 {D(qo[:, 1].min()):.0f}°）")
np.savez(DATA / "sim07_slew.npz", tau=TAU, **{f"{g}_{k}": v for g, d in res_B.items() for k, v in d.items()})

# ---- 図 B -------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
cm = np.where(np.isinf(chart), np.nan, chart)
im = ax[0].pcolormesh(PSI, ALP, np.minimum(cm, 60), cmap="viridis", shading="auto")
ax[0].contour(PSI, ALP, np.minimum(np.nan_to_num(cm, nan=999), 60), levels=[30], colors="w", linewidths=1.5)
ax[0].set_xlabel("azimuth ψ [deg]  (0: front, ±90: sides, ±180: rear)"); ax[0].set_ylabel("elevation α [deg]")
ax[0].set_title("B1: static max payload [kg]\n(white line: 30 kg, blank: unlimited = payload above footprint)"); fig.colorbar(im, ax=ax[0], label="kg (clipped at 60)")
d = res_B["Earth"]
a0, b0 = tank.track_half
ax[1].add_patch(plt.Rectangle((-a0, -b0), 2 * a0, 2 * b0, fc="C2", alpha=.1, ec="k"))
ax[1].add_patch(plt.Rectangle((-a0 + MARGIN, -b0 + MARGIN), 2 * (a0 - MARGIN), 2 * (b0 - MARGIN), fc="none", ec="k", ls=":"))
ax[1].plot(*d["zmp_flat"].T, "C0", label=f"horizontal arm, T = {d['T_flat']:.2f} s")
ax[1].plot(*d["zmp_flat_at_opt"].T, "C0--", lw=.8, label=f"horizontal arm, T = {d['T_opt']:.2f} s (tips)")
ax[1].plot(*d["zmp_opt"].T, "C3", label=f"optimized, T = {d['T_opt']:.2f} s")
ax[1].set_aspect("equal"); ax[1].set_xlim(-0.7, 0.7); ax[1].set_ylim(-0.5, 0.6)
ax[1].set_xlabel("x (front) [m]"); ax[1].set_ylabel("y (left) [m]")
ax[1].set_title("B2: ZMP path during 180° slew (Earth)"); ax[1].legend(fontsize=7, loc="lower left")
for gname, c in (("Earth", "C3"), ("Moon", "C1")):
    q = res_B[gname]["q_opt"]
    ax[2].plot(D(q[:, 0]), D(q[:, 1]), c, label=f"{gname}: optimized ({res_B[gname]['T_opt']:.2f} s)")
ax[2].plot(D(PSI_PATH), np.zeros(K), "C0", label="horizontal")
ax[2].axhline(D(ALPHA_MIN), color="k", ls=":", lw=.8)
ax[2].set_xlabel("azimuth ψ [deg]"); ax[2].set_ylabel("elevation α [deg]")
ax[2].set_title("B2: arm direction during slew"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "sim07_ground.png", dpi=150)
