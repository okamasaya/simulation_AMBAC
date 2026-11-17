"""Sim10: 段差の乗り越え — 戦車型（履帯）vs 二本脚.

準静的（ゆっくり動く）モデルで、越えられる段差の高さの上限を求める。

Part A: 戦車型（Sim06 の機体: 台車 80 kg ＋ 腕 7 kg, 履帯の接地長 0.8 m）
  ・形の限界: 履帯の前端（持ち上がった部分）の高さ h_nose より高い段差には、前端が引っかかる
  ・力の限界: 前端を段の角に乗せて車体が傾いた状態で、後ろの接地点と段の角の 2 点で支えながら
    履帯の駆動力（摩擦 μ 以内）で這い上がれるか（各接点の摩擦の範囲内で力のつり合いが成り立つか）。
    重心が後ろの接地点より後ろに来ると、後ろに転倒する。
  ・腕の姿勢で重心を前に移すと有利になるか（地上版の能動的質量バランス）
Part B: 二本脚（Sim09 の機体: 股関節 0.95 m, 大腿・下腿 各 0.45 m, 総質量 97 kg）
  ・可動域の限界: 振り出す脚の足を段の上（高さ h ＋ すき間 5 cm, 前方 0.30 m）に置けるか
    （股関節の屈曲 ≤ 120°, 膝の屈曲 ≤ 140°）
  ・トルクの限界: 段の上に置いた脚で体を持ち上げ始める瞬間（股関節が足首の真上）に、
    膝トルク τ_knee = M g (d − x_c) と 股関節トルク τ_hip がともに上限以下になる重心位置 x_c があるか
    （d: 膝の前方へのはみ出し, x_c: 足首から見た重心の前方位置。上体を前に傾けて x_c を足裏の範囲内で前へ移せる）
    → 「上体を立てたまま（x_c = 0）」と「上体の前傾を使う」の 2 通りを比べる
Part C: 相似拡大（長さ s 倍, 質量 s³ 倍, アクチュエータのトルク上限 ∝ s³）での比較
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, linprog

from common import DATA, FIG
from ambac.ground import TankArm

D = np.degrees
RAD = np.radians
G = 9.80665
MU = 0.6

# =============================================================================
# Part A: 戦車型
# =============================================================================
L_TRACK = 0.8              # 履帯の接地長（底面の直線部分）[m]
H_NOSE = 0.25              # 履帯の前端の持ち上がりの高さ（形で決まる限界）[m]
tank = TankArm()           # 台車 80 kg, 腕 4+3 kg, 肩 (0, 0.75)（台車座標, 原点は接地面中央）


def tank_com(q, s=1.0):
    """腕の姿勢 q（荷物なし）での機体全体の重心（車体座標: x 前方, z 上, 原点は履帯底面の中央）"""
    r = tank.scaled(s); r.payload = 0.0
    pos, _ = r.bodies(np.asarray(q))
    m = r.masses[:3]
    return (m[:, None] * pos[:3]).sum(0) / m.sum(), m.sum()


def climb_feasible(h, com, mass, mu=MU, s=1.0, n=60):
    """段差 h を這い上がれるか。車体が後ろの接地点 R と段の角 E の 2 点で支えられる区間を調べる。
    戻り値: (可否, 最も厳しい位置での必要摩擦係数)"""
    L = L_TRACK * s
    if h >= L:
        return False, np.inf
    worst = 0.0
    # 段の角が車体底面上の位置 xe（後端からの距離 se）にある。se は L（前端）から重心の真下まで減少する
    x_rear = -L / 2
    for se in np.linspace(L, max(h + 1e-3, com[0] - x_rear + 1e-3), n):
        th = np.arcsin(h / se)                          # 車体の傾き（前上がり）
        c, sn = np.cos(th), np.sin(th)
        Rm = np.array([[c, -sn], [sn, c]])
        R = np.array([0.0, 0.0])                        # 後ろの接地点を原点に
        E = Rm @ np.array([se, 0.0])                    # 段の角（= (se cosθ, h)）
        C = Rm @ (np.array(com) - np.array([x_rear, 0.0]))
        if C[0] < 0:                                    # 重心が後ろの接地点より後ろ → 後ろに転倒
            return False, np.inf
        # 未知数: N1, T1（R: 鉛直・水平前向き）, N2, T2（E: 車体底面に垂直・沿って前向き）
        n2 = np.array([-sn, c]); t2 = np.array([c, sn])
        A_eq = np.array([[0.0, 1.0, n2[0], t2[0]],      # x 方向
                         [1.0, 0.0, n2[1], t2[1]],      # z 方向
                         [0.0, 0.0, E[0] * n2[1] - E[1] * n2[0], E[0] * t2[1] - E[1] * t2[0]]])  # R まわりのモーメント
        b_eq = np.array([0.0, mass * G, C[0] * mass * G])
        # 必要な摩擦係数の最小値 m を求める: |T1| ≤ m N1, |T2| ≤ m N2 を満たす最小の m（二分法）
        def ok(m_):
            A_ub = np.array([[-m_, 1, 0, 0], [-m_, -1, 0, 0], [0, 0, -m_, 1], [0, 0, -m_, -1]])
            r = linprog(np.zeros(4), A_ub=A_ub, b_ub=np.zeros(4), A_eq=A_eq, b_eq=b_eq,
                        bounds=[(0, None), (None, None), (0, None), (None, None)], method="highs")
            return r.status == 0
        if not ok(10.0):
            return False, np.inf
        lo, hi = 0.0, 10.0
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if ok(mid) else (mid, hi)
        worst = max(worst, hi)
    return worst <= mu, worst


ARM_POSES = {"arm stowed back-up (COM rear)": RAD([120.0, 60.0]),
             "arm down (neutral)": RAD([-90.0, 0.0]),
             "arm forward (COM front)": RAD([0.0, 0.0])}


def tank_hmax(com, mass, s=1.0, mu=MU):
    """越えられる段差（前端の高さの限界と、力の限界の小さい方）。力の限界は二分法で求める"""
    lo, hi = 0.0, 0.79 * L_TRACK * s
    if climb_feasible(hi, com, mass, mu, s)[0]:
        h_force = hi
    else:
        for _ in range(12):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if climb_feasible(mid, com, mass, mu, s, n=30)[0] else (lo, mid)
        h_force = lo
    return min(h_force, H_NOSE * s), h_force


print(f"=== Part A: 戦車型（履帯接地長 {L_TRACK} m, 前端の高さ {H_NOSE} m, μ = {MU}）===")
resA = {}
for name, q in ARM_POSES.items():
    com, mass = tank_com(q)
    hmax, hforce = tank_hmax(com, mass)
    _, need = climb_feasible(0.20, com, mass)
    resA[name] = (com, hmax, hforce, need)
    print(f"  {name:30s}: 重心 x {com[0]:+.3f} m, 高さ {com[1]:.3f} m → 力の限界 {hforce:.2f} m,"
          f" 越えられる段差 {hmax:.2f} m（20 cm の段差に必要な摩擦係数 {need:.2f}）")

# =============================================================================
# Part B: 二本脚
# =============================================================================
L_TH, L_SH = 0.45, 0.45
HIP_H = 0.90               # 足首から股関節までの高さ（直立時）
M_TOT = 97.0
M_SHANK_FOOT = 4.0         # 支持脚の下腿＋足（膝より下）
HIP_FLEX, KNEE_FLEX = RAD(120), RAD(140)
TAU_KNEE = 250.0           # 膝伸展トルクの上限の目安 [N m]（人の成人男性の最大値は 200〜300 N m 程度）
STEP_FWD, CLEAR = 0.30, 0.05


def leg_ik(x, z, l1=L_TH, l2=L_SH):
    """股関節から見た足首の位置 (x 前方, z 上) → 股関節屈曲角（下向きから前へ）, 膝屈曲角"""
    d = np.hypot(x, z)
    if d > l1 + l2 or d < abs(l1 - l2):
        return None
    knee = np.pi - np.arccos((l1**2 + l2**2 - d**2) / (2 * l1 * l2))
    a = np.arctan2(x, -z)                                # 下向きからの前方角
    b = np.arccos((l1**2 + d**2 - l2**2) / (2 * l1 * d))
    return a + b, knee                                   # 膝を前に出す解


TAU_HIP = 250.0            # 股関節伸展トルクの上限の目安 [N m]
X_TOE = 0.15               # 足首から見たつま先の前方位置（重心を前に移せる上限）[m]
M_TRUNK = 56.0             # 股関節より上（上体・腕・頭）


def biped_limits(s=1.0, tau_knee=TAU_KNEE, tau_hip=TAU_HIP, lean=True):
    l1, l2, H = L_TH * s, L_SH * s, HIP_H * s
    M = M_TOT * s**3; m_above = M - M_SHANK_FOOT * s**3; m_trunk = M_TRUNK * s**3
    # 可動域の限界: 足を (前方 STEP_FWD, 高さ h + CLEAR) に置けるか（股関節は床から H の高さのまま）
    def rom_ok(h):
        sol = leg_ik(STEP_FWD * s, -(H - h - CLEAR * s), l1, l2)
        return sol is not None and sol[0] <= HIP_FLEX and sol[1] <= KNEE_FLEX
    hs = np.linspace(0.0, H, 400)
    h_rom = max([h for h in hs if rom_ok(h)], default=0.0)

    # トルクの限界: 股関節が段の上の足首の真上（股関節と足首の距離 H − h）、膝は前方に d はみ出す
    def torques(h, xc):
        Hh = H - h
        d = np.sqrt(max(l1**2 - (Hh / 2)**2, 0.0))
        t_knee = m_above * G * max(d - xc, 0.0)          # 重心が膝より後ろにある分だけ膝が支える
        t_hip = m_trunk * G * max(xc * m_above / m_trunk, 0.0)   # 重心を前へ移すために上体を前に出す分
        return t_knee, t_hip

    def best(h):
        xs = np.linspace(0.0, X_TOE * s, 61) if lean else np.array([0.0])
        return min(max(tk / tau_knee, th / tau_hip) for tk, th in (torques(h, x) for x in xs))

    hs2 = np.linspace(0.0, H * 0.999, 400)
    feas = [h for h in hs2 if best(h) <= 1.0]
    h_tau = max(feas) if feas else 0.0
    knee_tau = lambda h: torques(h, 0.0)[0]
    return h_rom, h_tau, knee_tau


h_rom, h_tau_up, knee_tau = biped_limits(lean=False)
_, h_tau, _ = biped_limits(lean=True)
print(f"\n=== Part B: 二本脚（股関節高さ {HIP_H} m, 大腿・下腿 {L_TH}/{L_SH} m, 膝・股関節トルク上限 {TAU_KNEE:.0f}/{TAU_HIP:.0f} N m）===")
print(f"  可動域の限界（足を段の上に置けるか）: {h_rom:.2f} m")
print(f"  トルクの限界: 上体を立てたまま {h_tau_up:.2f} m / 上体の前傾で重心を前へ移す {h_tau:.2f} m")
for h in (0.15, 0.20, 0.30, 0.40):
    print(f"    段差 {h:.2f} m で上体を立てたままの膝トルク {knee_tau(h):6.0f} N m")
h_biped = min(h_rom, h_tau)
print(f"  → 越えられる段差 {h_biped:.2f} m（上体の前傾あり。ゆっくり上る場合で、蹴り出しや勢いは使わない）")

# =============================================================================
# Part C: 相似拡大
# =============================================================================
S_LIST = np.array([0.5, 1, 2, 5, 10])
rowsC = []
com_fwd, mass_fwd = tank_com(ARM_POSES["arm forward (COM front)"])
print("\n=== Part C: 相似拡大（長さ s 倍, 質量 s³ 倍, トルク上限 ∝ s³）での越えられる段差 / 機体の大きさ ===")
for s in S_LIST:
    com_s, mass_s = tank_com(ARM_POSES["arm forward (COM front)"], s)
    ht, _ = tank_hmax(com_s, mass_s, s)
    hr, hq, _ = biped_limits(s, TAU_KNEE * s**3, TAU_HIP * s**3)
    hb = min(hr, hq)
    rowsC.append((s, ht, hb, ht / s, hb / s))
    print(f"  s = {s:4.1f}（全高 約 {1.8 * s:4.1f} m）: 戦車 {ht:6.2f} m（大きさ比 {ht / s:.2f}）,"
          f" 二本脚 {hb:6.2f} m（大きさ比 {hb / s:.2f}, 可動域 {hr / s:.2f} / トルク {hq / s:.2f}）")
rowsC = np.array(rowsC)
np.savetxt(DATA / "sim10_scaling.csv", rowsC, delimiter=",", comments="",
           header="scale,tank_hmax[m],biped_hmax[m],tank_hmax/s,biped_hmax/s")
with open(DATA / "sim10_tank.csv", "w") as fh:
    fh.write("arm_pose,com_x[m],com_z[m],h_max[m],h_force_limit[m],mu_needed_at_0.20m\n")
    for k, (com, hm, hf, need) in resA.items():
        fh.write(f"{k},{com[0]:.4f},{com[1]:.4f},{hm:.3f},{hf:.3f},{need:.3f}\n")

# ---- 図 ------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
hh = np.linspace(0.02, 0.6, 30)
for (name, q), c in zip(ARM_POSES.items(), ["C3", "C7", "C2"]):
    com, mass = tank_com(q)
    need = [climb_feasible(h, com, mass)[1] for h in hh]
    ax[0].plot(hh, np.minimum(need, 3), c, label=name)
ax[0].axhline(MU, color="k", ls=":", label=f"available friction μ = {MU}")
ax[0].axvline(H_NOSE, color="gray", ls="--", label=f"track nose height {H_NOSE} m")
ax[0].set_ylim(0, 2); ax[0].set_xlabel("step height h [m]"); ax[0].set_ylabel("friction coefficient needed")
ax[0].set_title("A: tank climbing a step (arm pose shifts COM)"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
hb_ = np.linspace(0.0, 0.6, 61)
ax[1].plot(hb_, [knee_tau(h) for h in hb_], "C0", label="knee torque, trunk upright")
ax[1].axvline(h_tau, color="C2", ls="-.", label=f"limit with trunk lean {h_tau:.2f} m")
ax[1].axhline(TAU_KNEE, color="k", ls=":", label=f"knee torque limit {TAU_KNEE:.0f} N m")
ax[1].axvline(h_rom, color="gray", ls="--", label=f"hip/knee range-of-motion limit {h_rom:.2f} m")
ax[1].set_xlabel("step height h [m]"); ax[1].set_ylabel("[N m]")
ax[1].set_title("B: biped stepping up (quasi-static)"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
ax[2].semilogx(rowsC[:, 0], rowsC[:, 3], "o-C0", label="tank (tracks)")
ax[2].semilogx(rowsC[:, 0], rowsC[:, 4], "s-C3", label="biped (legs)")
ax[2].set_xlabel("length scale s (s=1: human, s=10: 18 m class)"); ax[2].set_ylabel("max step height / s [m]")
ax[2].set_title("C: scaling (actuator torque ∝ s³)"); ax[2].legend(); ax[2].grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig(FIG / "sim10_steps.png", dpi=150)
