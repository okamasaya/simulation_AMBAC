"""Sim08: 肩の 3 次元運動 — AMBAC から人体工学へ.

Part A: コッドマンのパラドックス
  腕をひねらずに（各瞬間の回転軸が腕に垂直）動かして元の向きに戻すと、上腕は長軸まわりにひねられている。
  そのひねり角 = 腕の向きが球面上で囲んだ立体角（ガウス・ボンネの定理）。
  → AMBAC の「ループが囲む曲率の面積 = 正味の回転」（01 章）と同じ構造（単位球面は曲率 1 の空間）。
  2 軸ジンバルの肩（Sim07）では、姿勢が関節角の関数なのでパラドックスは起きない（代わりにひねりを選べない）。
Part B: 腕組みは可能か — 肩の機構と胸板の厚さ
  右手を左上腕（肘の上）に置き、前腕を胸の前で水平に交差させる姿勢を、逆運動学（最適化）で探す。
  機構: 2 軸ジンバル / 3 自由度の球関節（人の可動域） / 球関節＋上肢帯の前方突出 / 可動域制限なし（幾何のみ）
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rt

from common import DATA, FIG
from ambac.shoulder import DOWN, UpperBody, no_spin_transport, slerp_path, swing_twist

D = np.degrees
RAD = np.radians

# =============================================================================
# Part A: コッドマンのパラドックス
# =============================================================================
print("=== Part A: コッドマンのパラドックス（ひねらずに腕を動かして元の向きに戻す）===")
fwd, side, up = np.array([1, 0, 0.]), np.array([0, -1, 0.]), np.array([0, 0, 1.])


def loop_twist(path):
    R = no_spin_transport(path)
    assert np.allclose(R.apply(path[0]), path[-1], atol=1e-6)
    return swing_twist(R, axis=DOWN if np.allclose(path[0], DOWN) else path[0])


def solid_angle(path):
    """閉曲線が球面上で囲む立体角（下向き DOWN を内側とする, 測地三角形の和）"""
    c = path.mean(0); c /= np.linalg.norm(c)
    tot = 0.0
    for a, b in zip(path[:-1], path[1:]):
        num = c @ np.cross(a, b); den = 1 + c @ a + a @ b + b @ c
        tot += 2 * np.arctan2(num, den)
    return tot


cases = {}
p = np.vstack([slerp_path(DOWN, fwd), slerp_path(fwd, side)[1:], slerp_path(side, DOWN)[1:]])
cases["下→前→横→下（古典的なコッドマンの動作）"] = p
for beta in (20, 45, 90):
    a = DOWN; e1 = fwd; e2 = np.cross(a, e1)
    ph = np.linspace(0, 2 * np.pi, 721)
    ring = np.array([np.cos(RAD(beta)) * a + np.sin(RAD(beta)) * (np.cos(t) * e1 + np.sin(t) * e2) for t in ph])
    # 下ろした腕から円錐の縁まで上げ、1 周して、下ろす
    path = np.vstack([slerp_path(DOWN, ring[0]), ring[1:], slerp_path(ring[-1], DOWN)[1:]])
    cases[f"腕で半頂角 {beta}° の円錐を描いて下ろす"] = path
rowsA = []
for name, path in cases.items():
    tw = loop_twist(path); om = solid_angle(path)
    tw = om + (tw - om + np.pi) % (2 * np.pi) - np.pi      # ひねりは 360° の違いを区別できない → 立体角に揃える
    rowsA.append((name, D(tw), D(om)))
    print(f"  {name:28s}: 上腕のひねり {D(tw):+7.2f}°  / 囲んだ立体角 {D(om):+7.2f}°（ラジアンを度に換算）")

# 2 軸ジンバル（Sim07）の場合: 姿勢は (ψ, el) の関数 → 元の関節角に戻れば必ずひねりはゼロ
print("  2 軸ジンバルの肩: 関節角が元に戻れば姿勢も元に戻る（ひねり 0°）。代わりに、ひねりを独立に選べない。")

# =============================================================================
# Part B: 腕組み
# =============================================================================
TARGET = np.array([0.20, 0.15, 0.00])        # 右手を置く位置（左上腕の肘の少し上、胸の前方・左寄り）
TOL = 0.015                                  # 到達とみなす誤差 [m]


def solve(body: UpperBody, design: str, n_start=40, seed=0):
    rng = np.random.default_rng(seed)

    def unpack(x):
        if design == "gimbal":
            psi, el, elb = x
            R = Rt.from_rotvec([0, 0, psi]) * Rt.from_rotvec([0, -el, 0])
            return R, elb, 0.0
        if design in ("ball", "free"):
            return Rt.from_rotvec(x[:3]), x[3], 0.0
        return Rt.from_rotvec(x[:3]), x[3], x[4]

    def obj(x):
        R, elb, prot = unpack(x)
        H = body.chain(R, elb, prot)[2]
        return np.sum((H - TARGET)**2) * 1e3

    def cons(x):
        R, elb, prot = unpack(x)
        S, E, H, u, f = body.chain(R, elb, prot)
        c = list(body.collision(S, E, H))
        c += [elb, body.elbow_max - elb]
        if design in ("ball", "ball+sg"):
            # 水平内転の限界: 腕の向きの水平成分が正中線側へ hadd_max を越えない
            c.append(np.sin(body.hadd_max) * u[0] - np.cos(body.hadd_max) * u[1] + 0.0)
            tw = swing_twist(R)
            c += [tw + body.twist_int, body.twist_ext - tw]   # 内旋は負のツイスト
        if design == "ball+sg":
            c += [prot, body.prot_max - prot]
        return np.array(c)

    nx = {"gimbal": 3, "ball": 4, "free": 4, "ball+sg": 5}[design]
    best = None
    for k in range(n_start):
        if design == "gimbal":
            x0 = np.r_[rng.uniform(-np.pi, np.pi), rng.uniform(0, np.pi / 2), rng.uniform(0.5, 2.4)]
        else:
            x0 = np.r_[rng.normal(0, 0.8, 3), rng.uniform(0.5, 2.4)]
            if design == "ball+sg":
                x0 = np.r_[x0, rng.uniform(0, body.prot_max)]
        r = minimize(obj, x0, method="SLSQP", constraints=[{"type": "ineq", "fun": cons}], options=dict(maxiter=400))
        if cons(r.x).min() > -1e-4 and (best is None or r.fun < best.fun):
            best = r
    R, elb, prot = unpack(best.x)
    S, E, H, u, f = body.chain(R, elb, prot)
    err = np.linalg.norm(H - TARGET)
    hadd = D(np.arctan2(u[1], u[0])) if np.hypot(u[0], u[1]) > 1e-6 else 0.0
    return dict(ok=err < TOL, err=err, S=S, E=E, H=H, elbow=D(elb), twist=D(swing_twist(R)),
                hadd=hadd, prot=D(prot), x=best.x)


DESIGNS = {"gimbal": "2 軸ジンバル（Sim07）", "ball": "球関節（人の可動域）",
           "ball+sg": "球関節＋上肢帯の前方突出", "free": "球関節（可動域制限なし）"}
CHESTS = {"細身 (奥行 0.18 m)": 0.09, "標準 (0.22 m)": 0.11, "厚い胸板・装甲 (0.26 m)": 0.13}
print(f"\n=== Part B: 腕組み（右手を左上腕へ, 目標 {TARGET} m, 許容誤差 {TOL * 100:.1f} cm）===")
resB = {}
for cname, bx in CHESTS.items():
    body = UpperBody(torso_half=(bx, 0.14))
    for d, dname in DESIGNS.items():
        r = solve(body, d)
        resB[(cname, d)] = r
        extra = f"水平内転 {r['hadd']:5.1f}°, ひねり {r['twist']:+6.1f}°（負=内旋）, 肘 {r['elbow']:5.1f}°"
        if d == "ball+sg":
            extra += f", 前方突出 {r['prot']:4.1f}°"
        print(f"  [{cname:16s}] {dname:20s}: {'可能' if r['ok'] else '不可'}（誤差 {r['err'] * 100:5.1f} cm）  {extra}")

# ---- B2: 腕組みに最低限必要な内旋量（ひねりの大きさを最小化）---------------------------
#   内旋の可動域は姿勢によって変わり（70° は腕を 90° 外転した姿勢での標準値）、
#   さらに「ひねり」の数値はどの基準姿勢から測るかで変わる（Part A の帰結）。
#   そこで可動域で合否を決める代わりに、必要なひねりの最小値を求め、人の値と並べる。


def required_twist(body, with_sg, n_start=40, seed=1):
    rng = np.random.default_rng(seed)

    def unpack(x):
        return Rt.from_rotvec(x[:3]), x[3], (x[4] if with_sg else 0.0)

    def cons(x):
        R, elb, prot = unpack(x)
        S, E, H, u, f = body.chain(R, elb, prot)
        c = list(body.collision(S, E, H)) + [elb, body.elbow_max - elb,
                                             np.sin(body.hadd_max) * u[0] - np.cos(body.hadd_max) * u[1],
                                             (TOL * 0.9)**2 - np.sum((H - TARGET)**2)]
        if with_sg:
            c += [prot, body.prot_max - prot]
        return np.array(c)

    best = None
    for k in range(n_start):
        x0 = np.r_[rng.normal(0, 0.8, 3), rng.uniform(0.5, 2.4)]
        if with_sg:
            x0 = np.r_[x0, rng.uniform(0, body.prot_max)]
        r = minimize(lambda x: swing_twist(unpack(x)[0])**2, x0, method="SLSQP",
                     constraints=[{"type": "ineq", "fun": cons}], options=dict(maxiter=400))
        if cons(r.x).min() > -1e-5 and (best is None or r.fun < best.fun):
            best = r
    if best is None:
        return None
    R, elb, prot = unpack(best.x)
    return dict(twist=D(swing_twist(R)), elbow=D(elb), prot=D(prot))


print("\n=== Part B2: 腕組みに最低限必要な内旋（ひねり）の大きさ（水平内転 ≤ 45°, 前方突出 ≤ 25° の下で）===")
print("  参考: 人の内旋の可動域はおよそ 70〜90°（姿勢と測り方に依存）")
req = {}
for cname, bx in CHESTS.items():
    body = UpperBody(torso_half=(bx, 0.14))
    for sg in (False, True):
        r = required_twist(body, sg)
        req[(cname, sg)] = r
        lab = "球関節＋上肢帯" if sg else "球関節のみ    "
        if r is None:
            print(f"  [{cname:16s}] {lab}: 解なし")
        else:
            print(f"  [{cname:16s}] {lab}: 必要な内旋 {-r['twist']:5.1f}°（肘 {r['elbow']:5.1f}°"
                  + (f", 前方突出 {r['prot']:4.1f}°）" if sg else "）"))

import csv
with open(DATA / "sim08_required_twist.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["chest", "shoulder_girdle", "required_internal_rotation_deg", "elbow_deg", "protraction_deg"])
    for (c, sg), r in req.items():
        w.writerow([c, sg] + ([f"{-r['twist']:.1f}", f"{r['elbow']:.1f}", f"{r['prot']:.1f}"] if r else ["", "", ""]))
with open(DATA / "sim08_arm_crossing.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["chest", "design", "feasible", "hand_error_m", "horizontal_adduction_deg", "twist_deg",
                "elbow_deg", "protraction_deg"])
    for (c, d), r in resB.items():
        w.writerow([c, d, r["ok"], f"{r['err']:.4f}", f"{r['hadd']:.1f}", f"{r['twist']:.1f}",
                    f"{r['elbow']:.1f}", f"{r['prot']:.1f}"])
with open(DATA / "sim08_codman.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["path", "twist_deg", "solid_angle_deg"])
    for row in rowsA:
        w.writerow([row[0], f"{row[1]:.3f}", f"{row[2]:.3f}"])

# ---- 図 ------------------------------------------------------------------------
fig = plt.figure(figsize=(21, 5.2))
axA = fig.add_subplot(1, 4, 1, projection="3d")
uu, vv = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
axA.plot_wireframe(np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv), color="gray", lw=.3, alpha=.4)
for (name, path), c in zip(cases.items(), ["C3", "C0", "C1", "C2"]):
    axA.plot(*path.T, c, lw=2)
axA.set_title("A: arm-direction loops on the unit sphere\ntwist = enclosed solid angle")
axA.set_box_aspect((1, 1, 1)); axA.view_init(elev=-20, azim=-60)
axA2 = fig.add_subplot(1, 4, 2)
tw = [r[1] for r in rowsA]; om = [r[2] for r in rowsA]
axA2.plot([-200, 200], [-200, 200], "k:", lw=.8)
axA2.scatter(om, tw, c=["C3", "C0", "C1", "C2"], s=60, zorder=3)
for (n, t, o), lab in zip(rowsA, ["down-front-side-down (Codman)", "cone 20°", "cone 45°", "cone 90° (360° ≡ 0°)"]):
    axA2.annotate(lab, (o, t), fontsize=7, xytext=(4, -10), textcoords="offset points")
axA2.set_xlabel("enclosed solid angle [deg]"); axA2.set_ylabel("humeral twist after the loop [deg]")
axA2.set_title("A: Codman's paradox = geometric phase"); axA2.grid(alpha=.3)
axB = fig.add_subplot(1, 4, 3)
body = UpperBody()
th = np.linspace(0, 2 * np.pi, 200)
for (cname, bx), ls in zip(CHESTS.items(), [":", "-", "--"]):
    axB.plot(bx * np.cos(th), 0.14 * np.sin(th), "k", ls=ls, lw=.8)
for d, c in zip(DESIGNS, ["C0", "C1", "C2", "C7"]):
    r = resB[("標準 (0.22 m)", d)]
    P = np.array([r["S"], r["E"], r["H"]])
    axB.plot(P[:, 0], P[:, 1], "o-", color=c, label=f"{d}: {'OK' if r['ok'] else 'NG'} ({r['err'] * 100:.1f} cm)")
axB.plot(*TARGET[:2], "k*", ms=12, label="target (left upper arm)")
axB.set_aspect("equal"); axB.set_xlabel("x (front) [m]"); axB.set_ylabel("y (left) [m]")
axB.set_title("B: arm-crossing pose, top view (standard chest)"); axB.legend(fontsize=7, loc="lower right")
axB.grid(alpha=.3)
axC = fig.add_subplot(1, 4, 4)
xs = np.arange(len(CHESTS))
for k, (sg, lab) in enumerate(((False, "ball joint only"), (True, "ball + shoulder girdle"))):
    vals = [(-req[(c, sg)]["twist"]) if req[(c, sg)] else np.nan for c in CHESTS]
    axC.bar(xs + (k - 0.5) * 0.35, vals, 0.35, label=lab)
axC.axhspan(70, 90, color="gray", alpha=.2, label="human internal rotation range (approx.)")
axC.set_xticks(xs, ["slim\n0.18 m", "standard\n0.22 m", "thick/armored\n0.26 m"])
axC.set_ylabel("required internal rotation [deg]"); axC.set_title("B2: internal rotation needed to cross arms")
axC.legend(fontsize=7); axC.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(FIG / "sim08_shoulder.png", dpi=150)
