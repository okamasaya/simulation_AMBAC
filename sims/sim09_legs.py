"""Sim09: 「足なんて飾り」は本当か — 宇宙（AMBAC）と地上（安定性）での脚の価値.

Part A: 宇宙 — 脚は AMBAC の道具として役に立つか（総質量は同じ 76 kg）
  A1（平面）: 1 周あたりに稼げる回転の上限（可動域内の同符号曲率の面積分）を構成ごとに比較
     ・腕＋脚（人間型, Sim03 と同じ）
     ・腕＋腕（脚を外し、脚の質量は胴体へ）
     ・腕だけ（脚を外し、脚の質量は胴体へ）→ 平面では関節 1 つなので正味の回転は不可能
     ・腕＋脚（脚を 2 倍に重く：推進剤タンク・装甲を脚に置いた場合）
  A2（3 次元）: 股関節（球関節）で脚に円錐を描かせる vs 肩で腕に円錐を描かせる
Part B: 地上 — 二本脚で立つ機体 vs 戦車型台車（Sim06 の荷物運び動作）
  ・戦車型: 接地長 0.8 m、重心が低い
  ・二本脚: 足裏の長さ 0.25 m（支持範囲 ±0.125 m）、重心が高い（股関節 0.95 m、肩 1.45 m）
  ・二本脚＋腰の戦略: 股関節で上体を前後に傾け、腕の反作用を打ち消す（地上版 AMBAC）
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize

from common import DATA, FIG
from ambac.planar import Limb, PlanarRobot
from ambac.spatial import Arm3D, FloatingBody3D, sph
from ambac.ground import TankArm, min_jerk

D = np.degrees
RAD = np.radians

# =============================================================================
# Part A1: 平面 — 構成ごとの 1 周あたり回転の上限
# =============================================================================
M0, I0_PER_M = 60.0, (0.35**2 + 1.0**2) / 12            # Sim01 の胴体（慣性は質量に比例させる）
ARM = dict(attach=(0.0, 0.35), mass=4.0, length=0.7)
LEG = dict(attach=(0.0, -0.15), mass=12.0, length=0.9)
LIM_ARM = RAD([-150, 150]); LIM_LEG = RAD([-120, 30])
# 腕を 2 本目として使う場合、同じ肩に付けると 2 本が同じ動きになり冗長なので、
# 実際の人と同じく左右の肩に付けたものを矢状面に投影する（取付位置は同じ, 独立に動かせる）


def robot_of(limbs, extra_base_mass):
    mb = M0 + extra_base_mass
    return PlanarRobot(mb, mb * I0_PER_M, [Limb(**l) for l in limbs])


def bound_per_cycle(robot, lim1, lim2, n=121):
    g1 = np.linspace(*lim1, n); g2 = np.linspace(*lim2, n)
    G = np.array([[robot.connection([a, b]) for a in g1] for b in g2])
    F = np.gradient(G[..., 1], g1, axis=1) - np.gradient(G[..., 0], g2, axis=0)
    h = (g1[1] - g1[0]) * (g2[1] - g2[0])
    return max(-np.sum(F[F < 0]) * h, np.sum(F[F > 0]) * h), F


CONFIGS = {
    "arm + leg (human)": ([ARM, LEG], 0.0, LIM_ARM, LIM_LEG),
    "arm + arm (no legs)": ([ARM, dict(ARM)], 12.0 - 4.0, LIM_ARM, LIM_ARM),
    "arm only (no legs)": ([ARM], 12.0, None, None),
    "arm + heavy leg (x2)": ([ARM, dict(LEG, mass=24.0)], -12.0, LIM_ARM, LIM_LEG),
}
print("=== Part A1: 宇宙（平面）— 1 周あたりに稼げる回転の上限（総質量 76 kg で比較）===")
resA = {}
for name, (limbs, dm, l1, l2) in CONFIGS.items():
    r = robot_of(limbs, dm)
    tot = r.total_mass
    if len(limbs) < 2:
        resA[name] = (0.0, None)
        print(f"  {name:22s}: 総質量 {tot:.0f} kg → 関節 1 つなので正味の回転は 0°（往復で元に戻る）")
        continue
    b, F = bound_per_cycle(r, l1, l2)
    resA[name] = (b, F)
    print(f"  {name:22s}: 総質量 {tot:.0f} kg → 上限 {D(b):5.2f}°/周 → 90° に {np.ceil(90 / D(b)):.0f} 周")

# =============================================================================
# Part A2: 3 次元 — 股関節で脚の円錐 vs 肩で腕の円錐
# =============================================================================
from sim07_helpers import cone_holonomy  # noqa: E402

torso = dict(base_mass=46.0, base_dims=(0.25, 0.4, 0.8))          # 胴体＋頭（脚・腕を除く）
arm3 = Arm3D(shoulder=(0.0, -0.2, 0.30), mass=4.0, length=0.7)
leg3 = Arm3D(shoulder=(0.0, -0.1, -0.40), mass=12.0, length=0.9)
# 残りの手足（反対側の腕・脚）は胴体に固定された質量として加える
fixed_extra = 4.0 + 12.0
b_arm = FloatingBody3D(torso["base_mass"] + fixed_extra + 12.0, torso["base_dims"], arm3)   # 脚は固定
b_leg = FloatingBody3D(torso["base_mass"] + fixed_extra + 4.0, torso["base_dims"], leg3)    # 腕は固定
print("\n=== Part A2: 宇宙（3 次元）— 円錐運動 1 周あたりの胴体の回転 ===")
rowsA2 = []
for name, body, axis, beta in (
        ("腕（肩）: 前方軸, 半頂角 45°", b_arm, sph(0, 0), RAD(45)),
        ("腕（肩）: 前下方 30° 軸, 半頂角 30°", b_arm, sph(0, RAD(-60)), RAD(30)),
        ("脚（股関節）: 前下方 30° 軸, 半頂角 30°", b_leg, sph(0, RAD(-60)), RAD(30)),
        ("脚（股関節）: 前下方 30° 軸, 半頂角 45°", b_leg, sph(0, RAD(-60)), RAD(45))):
    rv = cone_holonomy(body, axis, beta)
    rowsA2.append((name, D(np.linalg.norm(rv)), rv / np.linalg.norm(rv)))
    print(f"  {name:32s}: {D(np.linalg.norm(rv)):6.2f}°/周, 回転軸 {np.round(rv / np.linalg.norm(rv), 2)}")
print("  ※ 股関節の可動域は肩より狭い（屈曲 ≈120°, 伸展 ≈20°, 外転 ≈45°）。同じ軸・半頂角（前下方 30°, 30°）で比べると脚は腕の"
      f" {rowsA2[2][1] / rowsA2[1][1]:.1f} 倍回る")

# =============================================================================
# Part B: 地上 — 二本脚 vs 戦車型
# =============================================================================
MU, MARGIN = 0.6, 0.02
Q0 = RAD([-20.0, -50.0]); Q1 = RAD([130.0, 60.0])
K = 401
TAU = np.linspace(0, 1, K)
Q_NAIVE = Q0 + (Q1 - Q0) * min_jerk(TAU)[:, None]


class Standing(TankArm):
    """TankArm に「股関節で上体を傾ける」自由度を加えたもの.
    base を 脚（固定）と 上体（股関節まわりに回転）に分け、肩は上体に付く。"""

    def __init__(self, legs_mass, legs_com, torso_mass, hip, torso_com_r, shoulder_r, half_support, **kw):
        super().__init__(base_mass=legs_mass + torso_mass, half_track=half_support, **kw)
        self.legs_mass, self.legs_com = legs_mass, np.array(legs_com)
        self.torso_mass, self.hip = torso_mass, np.array(hip)
        self.torso_com_r, self.shoulder_r = torso_com_r, shoulder_r
        self.I_torso = torso_mass * (0.25**2 + 0.6**2) / 12

    def reactions_lean(self, q_path, lean, T):
        """lean: 上体の前傾角（鉛直から前向きが正）"""
        K = len(q_path)
        e = np.stack([np.sin(lean), np.cos(lean)], -1)          # 上体の向き（x, z）
        tc = self.hip + self.torso_com_r * e
        S = self.hip + self.shoulder_r * e
        a1 = q_path[:, 0] - lean; a2 = a1 + q_path[:, 1]          # 腕の角度は上体基準
        e1 = np.stack([np.cos(a1), np.sin(a1)], -1); e2 = np.stack([np.cos(a2), np.sin(a2)], -1)
        el = S + self.l1 * e1; tip = el + self.l2 * e2
        pos = np.stack([np.broadcast_to(self.legs_com, tc.shape), tc, S + 0.5 * self.l1 * e1,
                        el + 0.5 * self.l2 * e2, tip], 1)
        ang = np.stack([np.zeros(K), -lean, a1, a2, a2], 1)
        m = np.array([self.legs_mass, self.torso_mass, self.m1, self.m2, self.payload])
        I = np.array([0.0, self.I_torso, self.m1 * self.l1**2 / 12, self.m2 * self.l2**2 / 12, 0.0])
        h = 1 / (K - 1)
        acc = np.gradient(np.gradient(pos, h, axis=0), h, axis=0) / T**2
        alp = np.gradient(np.gradient(ang, h, axis=0), h, axis=0) / T**2
        x, z = pos[..., 0], pos[..., 1]; ax, az = acc[..., 0], acc[..., 1]
        N = (m * (az + self.g)).sum(-1); Fx = (m * ax).sum(-1)
        p = ((m * (x * (az + self.g) - z * ax)).sum(-1) + (I * alp).sum(-1)) / N
        return p, N, Fx


tank = Standing(legs_mass=50.0, legs_com=(0.0, 0.20), torso_mass=30.0, hip=(0.0, 0.40),
                torso_com_r=0.15, shoulder_r=0.35, half_support=0.40)          # 肩 0.75 m（Sim06 と同じ）
biped = Standing(legs_mass=24.0, legs_com=(0.0, 0.50), torso_mass=56.0, hip=(0.0, 0.95),
                 torso_com_r=0.25, shoulder_r=0.50, half_support=0.125)        # 肩 1.45 m


def safe(r, qp, lean, T):
    p, N, Fx = r.reactions_lean(qp, lean, T)
    return min(((r.half_track - MARGIN) - np.abs(p)).min(), ((MU * N - np.abs(Fx)) / (MU * N.max())).min())


def tmin(r, qp, lean):
    f = lambda T: safe(r, qp, lean, T)
    if f(60.0) < 0:
        return np.inf
    return brentq(f, 0.2, 60.0, xtol=1e-3) if f(0.2) < 0 else 0.2


NB = 4
BAS = np.array([(TAU * (1 - TAU))**3 * np.sin((k + 1) * np.pi * TAU) for k in range(NB)]).T * 64
BAS_L = np.array([np.sin(np.pi * TAU) ** 2 * np.cos(k * np.pi * TAU) for k in range(NB)]).T
MJ = min_jerk(TAU)
QD_MAX, QDD_MAX, LEAN_MAX = 6.0, 60.0, RAD(40)
LEAN_QD, LEAN_QDD = 3.0, 15.0                  # 上体は重いので腕より遅い（股関節トルクの制約の代わり）


def optimize(r, use_lean, T0):
    # 上体の傾き: 開始・終了時の静的な傾き (L0, L1) の間を最小躍度でつなぎ、途中を補正する
    nv = 2 * NB + (NB + 2 if use_lean else 0)

    def unpack(x):
        qp = Q_NAIVE + BAS @ x[1:1 + 2 * NB].reshape(NB, 2)
        if use_lean:
            L0, L1 = x[1 + 2 * NB], x[2 + 2 * NB]
            lean = L0 + (L1 - L0) * MJ + BAS_L @ x[3 + 2 * NB:]
        else:
            lean = np.zeros(K)
        return x[0], qp, lean

    def cons(x):
        T, qp, lean = unpack(x)
        p, N, Fx = r.reactions_lean(qp, lean, T)
        h = 1 / (K - 1)
        qd = np.gradient(qp, h, axis=0) / T; qdd = np.gradient(np.gradient(qp, h, axis=0), h, axis=0) / T**2
        a = r.half_track - MARGIN
        c = [(a - p)[::2], (a + p)[::2], ((MU * N - np.abs(Fx)) / 100)[::2],
             (QD_MAX - np.abs(qd))[::2].ravel(), (QDD_MAX - np.abs(qdd))[::2].ravel() / 10]
        if use_lean:
            ld = np.gradient(lean, h) / T; ldd = np.gradient(ld, h) / T
            c += [(LEAN_MAX - np.abs(lean))[::2], (LEAN_QD - np.abs(ld))[::2], (LEAN_QDD - np.abs(ldd))[::2] / 10]
        c.append(((N - 0.3 * r.masses.sum() * r.g) / 100)[::2])     # 床から浮かない
        return np.concatenate(c)

    best = None
    for seed in range(6):
        rng = np.random.default_rng(seed)
        x0 = np.r_[T0, rng.normal(0, 0.3 if seed else 0.0, nv)]
        res = minimize(lambda x: x[0], x0, method="SLSQP", constraints=[{"type": "ineq", "fun": cons}],
                       bounds=[(0.2, 60)] + [(-3, 3)] * nv, options=dict(maxiter=500))
        if res.success and cons(res.x).min() > -1e-4 and (best is None or res.x[0] < best.x[0]):
            best = res
    return (best.x[0], *unpack(best.x)[1:]) if best else (np.inf, None, None)


print("\n=== Part B: 地上 — 荷物運び（Sim06 と同じ腕の動作, 荷物 10 kg）の転ばない最短時間 ===")
print(f"  支持範囲: 戦車 ±{tank.half_track} m, 二本脚（足裏） ±{biped.half_track} m / 安全余裕 {MARGIN} m")
zero = np.zeros(K)
resB = {}
for name, r in (("tank", tank), ("biped", biped)):
    t_naive = tmin(r, Q_NAIVE, zero)
    p_static, _, _ = r.reactions_lean(Q_NAIVE, zero, 1e3)
    resB[name] = dict(naive=t_naive, static=(p_static.min(), p_static.max()))
    print(f"  {name:6s}: 静的な ZMP の範囲 [{p_static.min():+.3f}, {p_static.max():+.3f}] m, 単純軌道の最短 {t_naive:.2f} s")
for name, r, lean in (("tank", tank, False), ("tank", tank, True), ("biped", biped, False), ("biped", biped, True)):
    T0 = resB[name]["naive"] if np.isfinite(resB[name]["naive"]) else 5.0
    T, qp, ln = optimize(r, lean, T0 * 1.02)
    key = name + ("+hip" if lean else "")
    resB[key + "_opt"] = (T, qp, ln)
    msg = f"最短 {T:.2f} s" if np.isfinite(T) else "解なし（静止状態でも転倒）"
    print(f"  {key:10s}: 腕の軌道を最適化{'＋上体の前後傾' if lean else ''} → {msg}"
          + (f"（上体の傾き {D(ln.min()):+.0f}°〜{D(ln.max()):+.0f}°, 負=後ろに反る）" if lean and ln is not None else ""))

np.savez(DATA / "sim09_ground.npz", tau=TAU,
         **{f"{k}_T": v[0] for k, v in resB.items() if k.endswith("_opt")},
         **{f"{k}_q": v[1] for k, v in resB.items() if k.endswith("_opt") and v[1] is not None},
         **{f"{k}_lean": v[2] for k, v in resB.items() if k.endswith("_opt") and v[2] is not None})
with open(DATA / "sim09_space.csv", "w") as fh:
    fh.write("config,bound_deg_per_cycle\n")
    for k, (b, _) in resA.items():
        fh.write(f"{k},{D(b):.3f}\n")
    for n, v, ax_ in rowsA2:
        fh.write(f"3D {n},{v:.3f}\n")

# ---- 図 ------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
names = list(resA); vals = [D(resA[k][0]) for k in names]
ax[0].barh(names, vals, color=["C0", "C1", "C7", "C2"])
for i, v in enumerate(vals):
    ax[0].text(v + 0.2, i, f"{v:.1f}°", va="center")
ax[0].set_xlabel("max body rotation per limb cycle [deg] (planar, 76 kg)")
ax[0].set_title("A1: space — are legs useful for AMBAC?"); ax[0].invert_yaxis(); ax[0].grid(alpha=.3, axis="x")
labels = ["tank\nnaive", "tank\nopt. arm", "tank\n+waist lean", "biped\nnaive", "biped\nopt. arm", "biped\n+hip lean"]
Ts = [resB["tank"]["naive"], resB["tank_opt"][0], resB["tank+hip_opt"][0], resB["biped"]["naive"],
      resB["biped_opt"][0], resB["biped+hip_opt"][0]]
ax[1].bar(labels, [t if np.isfinite(t) else 0 for t in Ts], color=["C0", "C0", "C9", "C3", "C3", "C2"])
for i, t in enumerate(Ts):
    if np.isfinite(t):
        ax[1].text(i, t + 0.04, f"{t:.2f} s", ha="center", fontsize=9)
    else:
        ax[1].text(i, 0.05, "tips over\neven at rest", ha="center", fontsize=7, rotation=90, va="bottom")
ax[1].set_ylim(0, max(t for t in Ts if np.isfinite(t)) * 1.15)
ax[1].tick_params(axis="x", labelsize=8)
ax[1].set_ylabel("min safe time for pick & place [s]"); ax[1].set_title("B: ground — tank vs standing biped")
ax[1].grid(alpha=.3, axis="y")
T, qp, ln = resB["biped+hip_opt"]
p, _, _ = biped.reactions_lean(qp, ln, T)
p0, _, _ = biped.reactions_lean(Q_NAIVE, zero, T)          # 同じ時間・上体を傾けない場合
ax[2].plot(TAU, p0, "C3", label=f"biped, naive arm, no lean (T = {T:.2f} s)")
ax[2].plot(TAU, p, "C2", label=f"biped, optimized arm + hip lean (T = {T:.2f} s)")
ax2b = ax[2].twinx(); ax2b.plot(TAU, D(ln), "C2:", lw=1); ax2b.set_ylabel("torso lean [deg] (dotted)")
for sgn in (-1, 1):
    ax[2].axhline(sgn * biped.half_track, color="k", lw=.8)
ax[2].set_ylim(-0.6, 0.6); ax[2].set_xlabel("normalized time τ"); ax[2].set_ylabel("ZMP [m] (foot: ±0.125 m)")
ax[2].set_title("B: hip strategy (ground AMBAC with the torso)"); ax[2].legend(fontsize=7, loc="lower left")
ax[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "sim09_legs.png", dpi=150)
