"""詳細版の技術報告書のための追加の図（figures/detail/）と数値（data/detail_summary.txt）を作る.

既存のモデル（src/ambac）とシミュレーションスクリプトの関数を使う。各図は独立に作れる:
    python detail_figures.py            # すべて
    python detail_figures.py D1 D5      # 指定したものだけ
"""
import json
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from common import DATA, FIG, human_like, smooth_step
from ambac.planar import Limb, PlanarRobot

D = np.degrees
RAD = np.radians
OUT = FIG / "detail"
OUT.mkdir(exist_ok=True)
LOG = []


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)


def section(src_file, start, end=None):
    """シミュレーションスクリプトの一部（関数定義など）を取り出して実行し、名前空間を返す"""
    src = open(src_file, encoding="utf-8").read()
    i = src.index(start) if start else 0
    j = src.index(end, i) if end else len(src)
    ns = {"__name__": "detail"}
    exec(compile(src[:i] if start is None else src[i:j], src_file, "exec"), ns)
    return ns


def header(src_file, end):
    src = open(src_file, encoding="utf-8").read()
    ns = {"__name__": "detail"}
    exec(compile(src[:src.index(end)], src_file, "exec"), ns)
    return ns


# =============================================================================
def D1():
    """閉じた形の A(q), B(q), Γ(q) と数値計算の比較（第 1 章）"""
    r = human_like()            # 腕のみ
    mb, Ib = r.base_mass, r.base_inertia
    lb = r.limbs[0]; ml, Il = lb.mass, lb.inertia
    a = np.array(lb.attach); la = np.linalg.norm(a); phi = np.arctan2(a[1], a[0]); lc = lb.length / 2
    mu = mb * ml / (mb + ml)
    q = np.linspace(-np.pi, np.pi, 361)
    A = Ib + Il + mu * (la**2 + lc**2 + 2 * lc * la * np.cos(q - phi))
    B = Il + mu * (lc**2 + lc * la * np.cos(q - phi))
    num = np.array([r.coupling([x]) for x in q], dtype=object)
    An = np.array([x[0] for x in num]); Bn = np.array([x[1][0] for x in num])
    err = max(np.abs(A - An).max(), np.abs(B - Bn).max())
    log(f"[D1] 閉じた形と数値計算の最大差 {err:.2e}（A の範囲 {A.min():.3f}〜{A.max():.3f}, Γ の範囲 {(B/A).min():.4f}〜{(B/A).max():.4f}）")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(D(q), A, label="A(q) closed form"); ax[0].plot(D(q), B, label="B(q) closed form")
    ax[0].plot(D(q[::12]), An[::12], "k.", label="numerical"); ax[0].plot(D(q[::12]), Bn[::12], "k.")
    ax[0].set_xlabel("arm angle q [deg]"); ax[0].set_ylabel("[kg m²]"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[0].set_title("A(q), B(q): closed form vs numerical")
    ax[1].plot(D(q), B / A); ax[1].set_xlabel("arm angle q [deg]"); ax[1].set_ylabel("Γ = B/A")
    ax[1].set_title("mechanical connection of a single arm"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "D1_closed_form.png", dpi=150); plt.close(fig)


# =============================================================================
def D2():
    """腕 1 本の振り: 始点と終点ごとの胴体の回転（可逆）（第 2 章）"""
    r = human_like()
    q = np.linspace(-np.pi, np.pi, 181)
    G = np.array([r.connection([x])[0] for x in q])
    cum = np.concatenate([[0], np.cumsum((G[1:] + G[:-1]) / 2 * np.diff(q))])   # θ(q) = -cum
    th = -(cum[None, :] - cum[:, None])                                          # 始点 i → 終点 j
    log(f"[D2] 腕 1 本で得られる最大の首振り: {D(np.abs(th).max()):.1f}°（下 → 上の 180° で {D(-(np.interp(np.pi/2, q, cum) - np.interp(-np.pi/2, q, cum))):.1f}°）")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(D(q), D(q), D(th).T, cmap="RdBu_r", shading="auto")
    ax.set_xlabel("start arm angle [deg]"); ax.set_ylabel("end arm angle [deg]")
    ax.set_title("body rotation by a single arm swing [deg]\n(path-independent: depends only on start and end)")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(OUT / "D2_swing_map.png", dpi=150); plt.close(fig)


# =============================================================================
def curvature(robot, lim1, lim2, n=91):
    g1 = np.linspace(*lim1, n); g2 = np.linspace(*lim2, n)
    G = np.array([[robot.connection([a, b]) for a in g1] for b in g2])
    F = np.gradient(G[..., 1], g1, axis=1) - np.gradient(G[..., 0], g2, axis=0)
    h = (g1[1] - g1[0]) * (g2[1] - g2[0])
    return g1, g2, F, max(-F[F < 0].sum() * h, F[F > 0].sum() * h)


LIM_ARM, LIM_LEG = RAD([-150, 150]), RAD([-120, 30])


def D3():
    """手足の質量による曲率マップの違い（第 2 章・第 9 章）"""
    cases = [(4, 6), (4, 12), (4, 24), (8, 12)]
    fig, ax = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    vmax = None
    res = []
    for (ma, ml), a in zip(cases, ax.flat):
        r = PlanarRobot(60.0 + (16 - ma - ml), (60.0 + 16 - ma - ml) * (0.35**2 + 1) / 12,
                        [Limb((0, 0.35), ma, 0.7), Limb((0, -0.15), ml, 0.9)])
        g1, g2, F, b = curvature(r, LIM_ARM, LIM_LEG)
        vmax = vmax or np.abs(F).max()
        im = a.pcolormesh(D(g1), D(g2), F, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        a.contour(D(g1), D(g2), F, levels=[0], colors="k", linewidths=.6, linestyles="--")
        a.set_title(f"arm {ma} kg, leg {ml} kg: bound {D(b):.2f}°/cycle", fontsize=10); a.set_xlabel("arm q1 [deg]"); a.set_ylabel("leg q2 [deg]")
        res.append((ma, ml, D(b)))
    fig.colorbar(im, ax=ax.ravel().tolist(), shrink=.8, label="curvature F")
    fig.savefig(OUT / "D3_curvature_mass.png", dpi=150); plt.close(fig)
    log("[D3] 1 周あたりの上界（総質量 76 kg）: " + ", ".join(f"腕{a}kg・脚{l}kg → {b:.2f}°" for a, l, b in res))


# =============================================================================
def D4():
    """リアクションホイール: 旋回時間・ホイールの慣性と最大回転数（第 3 章）"""
    from ambac.planar import reaction_wheel
    T = np.geomspace(2, 120, 60)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mw, rr in ((0.5, 0.08), (2.0, 0.12), (8.0, 0.18)):
        r = human_like(4.0, 12.0); r.limbs.append(reaction_wheel(mw, rr))
        A, B = r.coupling([-np.pi / 2, -np.pi / 2, 0.0])
        dphi = np.pi / 2 * A / B[2]
        wpk = 1.875 * dphi / T * 60 / (2 * np.pi)
        ax.loglog(T, wpk, label=f"wheel {mw} kg, r={rr} m (I={0.5*mw*rr**2:.4f})")
        log(f"[D4] RW {mw} kg: 90° に必要なホイール回転 {dphi/(2*np.pi):.0f} 回, 10 s で旋回するときの最大 {1.875*dphi/10*60/(2*np.pi):.0f} rpm")
    ax.axhline(6000, color="k", ls=":", label="typical max ~6000 rpm")
    ax.axvline(13.2, color="C3", ls="--", label="AMBAC 90° time (Sim04, human)")
    ax.set_xlabel("maneuver time for 90° [s]"); ax.set_ylabel("peak wheel speed [rpm]"); ax.legend(fontsize=7); ax.grid(alpha=.3, which="both")
    ax.set_title("reaction wheel: speed needed for a 90° rest-to-rest turn (human-scale body)")
    fig.tight_layout(); fig.savefig(OUT / "D4_rw_tradeoff.png", dpi=150); plt.close(fig)


# =============================================================================
def D5():
    """トルク ∝ f², エネルギー ∝ f（第 4 章）"""
    ns = header("sim04_dynamics_scaling.py", "# =============================================================================\n# Part A")
    hum = ns["scaled_human"](1.0)
    fs = np.geomspace(0.1, 2.0, 9)
    tk = []; Epos = []
    for f in fs:
        t1, pr = ns["cycle_profile"](hum, f, npts=200)
        tk.append(np.abs(pr[:, 3:5]).max(0))
        cycles = (np.pi / 2) / abs(ns["DTH_C"])
        Epos.append(np.clip(pr[:, 5:7], 0, None).mean(0).sum() / f * cycles)
    tk = np.array(tk); Epos = np.array(Epos)
    s1 = np.polyfit(np.log(fs), np.log(tk[:, 1]), 1)[0]; s2 = np.polyfit(np.log(fs), np.log(Epos), 1)[0]
    log(f"[D5] 股関節トルク ∝ f^{s1:.2f}, 90° 旋回の正仕事 ∝ f^{s2:.2f}（0.1 Hz で {Epos[0]:.1f} J, 2 Hz で {Epos[-1]:.0f} J）")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].loglog(fs, tk[:, 0], "o-", label="shoulder"); ax[0].loglog(fs, tk[:, 1], "s-", label="hip")
    ax[0].axhline(40, color="C0", ls=":"); ax[0].axhline(120, color="C1", ls=":")
    ax[0].set_xlabel("loop frequency f [Hz]"); ax[0].set_ylabel("peak joint torque [N m]"); ax[0].legend(); ax[0].grid(alpha=.3, which="both")
    ax[0].set_title(f"torque ∝ f^{s1:.2f} (dotted: limits)")
    ax[1].loglog(fs, Epos, "o-C2"); ax[1].set_xlabel("loop frequency f [Hz]"); ax[1].set_ylabel("positive work for 90° [J]")
    ax[1].set_title(f"energy for 90° ∝ f^{s2:.2f} (no regeneration)"); ax[1].grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(OUT / "D5_torque_energy.png", dpi=150); plt.close(fig)


# =============================================================================
def D6():
    """学習した軌道生成器型方策の中身（第 5 章）"""
    ns = header("sim05a_learn_ambac.py", "# ---- 学習")
    P = np.load(DATA / "sim05a_policy_tg.npy")
    c, A, dlt, W = ns["tg_unpack"](P[None])
    e = np.linspace(-np.pi, np.pi, 361)
    u = ns["TG_NET"](W, np.stack([np.sin(e), np.cos(e)], -1)[None])[0, :, 0]
    psi = np.linspace(0, 2 * np.pi, 400)
    loop = ns["tg_target"](c[0], A[0], dlt[0], psi)
    zero = e[np.argmin(np.abs(u[(np.abs(e) < 1.0)]) + np.argmax(np.abs(e) < 1.0))]
    log(f"[D6] 学習した位相速度 u(e): e=+90° で {np.interp(np.pi/2, e, u):+.2f}, e=−90° で {np.interp(-np.pi/2, e, u):+.2f}, e=0 付近で {np.interp(0, e, u):+.3f}")
    g1, g2, F, _ = curvature(ns["robot"], LIM_ARM, LIM_LEG, n=81)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(D(e), u); ax[0].axhline(0, color="k", lw=.6); ax[0].axvline(0, color="k", lw=.6)
    ax[0].set_xlabel("attitude error e = target − θ [deg]"); ax[0].set_ylabel("phase rate command u (−1..1)")
    ax[0].set_title("learned feedback: loop speed & direction vs error"); ax[0].grid(alpha=.3)
    vm = np.abs(F).max()
    ax[1].pcolormesh(D(g1), D(g2), F, cmap="RdBu_r", vmin=-vm, vmax=vm, shading="auto")
    ax[1].contour(D(g1), D(g2), F, levels=[0], colors="k", linewidths=.6, linestyles="--")
    ax[1].plot(D(loop[:, 0]), D(loop[:, 1]), "g-", lw=2, label="learned target loop (clipped by joint limits)")
    ax[1].set_xlabel("arm q1 [deg]"); ax[1].set_ylabel("leg q2 [deg]"); ax[1].legend(fontsize=7, loc="lower right")
    ax[1].set_title("learned loop vs curvature")
    fig.tight_layout(); fig.savefig(OUT / "D6_learned_policy.png", dpi=150); plt.close(fig)


# =============================================================================
def D7():
    """推進剤最小の配分: 旋回時間ごとの計画（第 5 章）"""
    ns = header("sim05b_allocation.py", "# =============================================================================\n# Part B")
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=False)
    for T, c in ((20, "C0"), (60, "C1"), (110, "C2")):
        p = ns["plan_lp"](T, use_amb=True, soft=False)
        t = np.arange(len(p["tau"])) * p["dt"]
        ax[0].step(t / T, p["tau"] / ns["TAU_TH"], where="post", color=c, label=f"T={T} s ({p['prop']:.2f} kg)")
        # 平均化モデルでの周回平均姿勢の時間変化（計画どおりに実行した場合）
        L = np.concatenate([[0], np.cumsum(p["tau"] * p["dt"])])[:-1]
        thd = L / ns["A_BAR"] + p["f"] * ns["DTH"]
        th = np.concatenate([[0], np.cumsum(thd * p["dt"])])
        ax[1].plot(np.arange(len(th)) * p["dt"] / T, D(th), color=c)
        log(f"[D7] T={T} s: 推進剤 {p['prop']:.2f} kg, スラスタの噴射は最初と最後の各 {np.sum(np.abs(p['tau'])>1)//2 * p['dt']:.1f} s 程度, AMBAC は全区間で最大周波数（平均 f/fmax {p['f'].mean()/ns['F_MAX']:.2f}）")
    ax[0].set_ylabel("thruster τ / τmax"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].set_ylabel("cycle-averaged attitude θ̄ [deg]"); ax[1].set_xlabel("normalized time t/T"); ax[1].grid(alpha=.3)
    ax[1].text(0.02, 80, "AMBAC runs at f_max during the whole maneuver;\nthrusters add only the momentum it cannot supply", fontsize=8)
    ax[0].set_title("propellant-optimal plans (thrusters + AMBAC, 18 m class)")
    fig.tight_layout(); fig.savefig(OUT / "D7_lp_plans.png", dpi=150); plt.close(fig)


# =============================================================================
def D8():
    """フルード則: 規格化した ZMP が 1 本の曲線に重なる（第 6 章）"""
    from ambac.ground import TankArm, min_jerk
    r0 = TankArm()
    q0, q1 = RAD([-20, -50]), RAD([130, 60]); tau = np.linspace(0, 1, 401)
    qp = q0 + (q1 - q0) * min_jerk(tau)[:, None]
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for s, g, c in ((1, 9.81, "C0"), (10, 9.81, "C1"), (1, 1.625, "C2"), (5, 3.72, "C3")):
        r = r0.scaled(s, g); T = 1.616 * np.sqrt(s * 9.80665 / g)
        p, N, Fx = r.reactions(qp, T)
        ax.plot(tau, p / s, color=c, lw=2 if s == 1 and g > 9 else 1, ls="-" if s == 1 and g > 9 else "--",
                label=f"s={s}, g={g} m/s², T={T:.2f} s")
    ax.axhline(0.4, color="k", lw=.8); ax.axhline(-0.4, color="k", lw=.8)
    ax.set_xlabel("normalized time τ"); ax.set_ylabel("ZMP / s [m]"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax.set_title("Froude scaling: with T ∝ √(s/g) all ZMP curves coincide")
    fig.tight_layout(); fig.savefig(OUT / "D8_froude_collapse.png", dpi=150); plt.close(fig)
    log("[D8] T = 1.616√(s·g_E/g) で、4 条件の規格化 ZMP が一致（図）")


# =============================================================================
def D9():
    """スティックスリップ: 摩擦係数と振りの速さ（月面）（第 6 章）"""
    ns = header("sim06_ground_tank.py", 'print("\\n=== Part C')
    mus = np.linspace(0.2, 1.0, 9); TF = np.geomspace(0.3, 2.5, 12)
    Z = np.array([[ns["yaw_cycle"](T, 1.625, mu=m, dt=5e-4) for T in TF] for m in mus])
    log(f"[D9] 月面: μ=0.2 で 0.3 s の振り → {D(Z[0,0]):.1f}°/往復, μ=1.0 で → {D(Z[-1,0]):.1f}°/往復")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.pcolormesh(TF, mus, D(Z), cmap="viridis_r", shading="auto"); ax.set_xscale("log")
    ax.set_xlabel("fast-stroke time T_fast [s]"); ax.set_ylabel("friction coefficient μ")
    ax.set_title("Moon: net base yaw per cycle [deg] (single turret joint)"); fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(OUT / "D9_stickslip_map.png", dpi=150); plt.close(fig)


# =============================================================================
def D10():
    """円錐の軸の向きと 1 周あたりの回転（第 7 章）"""
    from sim07_helpers import cone_holonomy
    from ambac.spatial import Arm3D, FloatingBody3D, sph
    body = FloatingBody3D(60.0, (0.25, 0.4, 1.0), Arm3D(shoulder=(0.0, -0.2, 0.35), mass=4.0, length=0.7))
    PS = np.arange(-180, 181, 30); AL = np.arange(-60, 61, 30)
    Z = np.zeros((len(AL), len(PS)))
    for i, al in enumerate(AL):
        for j, ps in enumerate(PS):
            Z[i, j] = D(np.linalg.norm(cone_holonomy(body, sph(RAD(ps), RAD(al)), RAD(30), n=800)))
    k = np.unravel_index(Z.argmax(), Z.shape)
    log(f"[D10] 半頂角 30° の円錐: 1 周あたり {Z.min():.1f}〜{Z.max():.1f}°（最大は軸 ψ={PS[k[1]]}°, α={AL[k[0]]}°）")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    im = ax.pcolormesh(PS, AL, Z, cmap="magma", shading="auto")
    ax.set_xlabel("cone axis azimuth ψ0 [deg]"); ax.set_ylabel("cone axis elevation α0 [deg]")
    ax.set_title("body rotation per arm-cone cycle (β = 30°) vs cone axis direction [deg]"); fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(OUT / "D10_cone_axes.png", dpi=150); plt.close(fig)


# =============================================================================
def D11():
    """腕組みに必要な内旋と胸板の厚さ（第 8 章）"""
    src = open("sim08_shoulder.py", encoding="utf-8").read()
    ns = {"__name__": "detail"}
    pre = src[:src.index("# =============================================================================\n# Part A")]
    defs = src[src.index("TARGET = np.array"):src.index("DESIGNS = {")]
    rt = src[src.index("def required_twist"):src.index('print("\\n=== Part B2')]
    exec(pre + "\n" + defs + "\n" + rt, ns)
    depths = np.linspace(0.08, 0.14, 7)
    rows = []
    for bx in depths:
        body = ns["UpperBody"](torso_half=(bx, 0.14))
        r0 = ns["required_twist"](body, False, n_start=40); r1 = ns["required_twist"](body, True, n_start=20)
        rows.append((bx, -r0["twist"] if r0 else np.nan, -r1["twist"] if r1 else np.nan, r1["prot"] if r1 else np.nan))
    rows = np.array(rows)
    log("[D11] 胸の奥行 → 必要な内旋（球関節のみ / ＋上肢帯, 前方突出）: " +
        ", ".join(f"{2*b:.2f} m → {x:.0f}° / {y:.0f}° ({p:.0f}°)" for b, x, y, p in rows))
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(2 * rows[:, 0], rows[:, 1], "o-", label="ball joint only")
    ax.plot(2 * rows[:, 0], rows[:, 2], "s-", label="ball + shoulder girdle")
    ax2 = ax.twinx(); ax2.plot(2 * rows[:, 0], rows[:, 3], "^:", color="C2"); ax2.set_ylabel("protraction used [deg] (green)")
    ax.axhspan(70, 90, color="gray", alpha=.2, label="human internal rotation (approx.)")
    ax.set_xlabel("chest depth [m]"); ax.set_ylabel("required internal rotation [deg]"); ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=.3)
    ax.set_title("arm crossing: internal rotation needed vs chest depth")
    fig.tight_layout(); fig.savefig(OUT / "D11_chest_sweep.png", dpi=150); plt.close(fig)


# =============================================================================
def D12():
    """脚の質量の割合と 1 周あたりの回転の上界（第 9 章）"""
    fr = np.linspace(0.02, 0.4, 12); res = []
    for x in fr:
        ml = 76 * x
        r = PlanarRobot(76 - 4 - ml, (76 - 4 - ml) * (0.35**2 + 1) / 12, [Limb((0, 0.35), 4.0, 0.7), Limb((0, -0.15), ml, 0.9)])
        res.append(curvature(r, LIM_ARM, LIM_LEG, n=61)[3])
    res = np.array(res)
    log(f"[D12] 脚の質量の割合 {fr[0]*100:.0f}% → {D(res[0]):.1f}°/周, {fr[-1]*100:.0f}% → {D(res[-1]):.1f}°/周（人間型 16% 付近 ≈ 9.6°）")
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(fr * 100, D(res), "o-"); ax.axvline(12 / 76 * 100, color="gray", ls="--", label="human-like (12 kg leg)")
    ax.set_xlabel("leg mass / total mass [%] (total 76 kg, arm 4 kg)"); ax.set_ylabel("max rotation per cycle [deg]")
    ax.set_title("space: heavier legs make AMBAC stronger"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "D12_leg_mass.png", dpi=150); plt.close(fig)


# =============================================================================
def D13():
    """段差: 戦車型の摩擦係数・前端高さ、二本脚の膝トルク（第 10 章）"""
    src = open("sim10_steps.py", encoding="utf-8").read()
    ns = {"__name__": "detail"}
    exec(src[:src.index('print(f"=== Part A')] + "\n" +
         src[src.index("L_TH, L_SH"):src.index("h_rom, h_tau_up, knee_tau")], ns)
    com, mass = ns["tank_com"](ns["ARM_POSES"]["arm forward (COM front)"])
    mus = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 1.0])
    hf = []
    for m in mus:
        lo, hi = 0.0, 0.6
        for _ in range(10):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if ns["climb_feasible"](mid, com, mass, mu=m, n=25)[0] else (lo, mid)
        hf.append(lo)
    taus = np.linspace(100, 400, 13)
    hb = [ns["biped_limits"](1.0, t, t)[1] for t in taus]
    log("[D13] 戦車型の力の限界（腕を前）: " + ", ".join(f"μ={m} → {h:.2f} m" for m, h in zip(mus, hf)))
    log("[D13] 二本脚の限界（上体の前傾あり）: " + ", ".join(f"{t:.0f} N m → {h:.2f} m" for t, h in zip(taus[::3], hb[::3])))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(mus, hf, "o-", label="force limit (arm forward)"); ax[0].axhline(0.25, color="gray", ls="--", label="track nose height 0.25 m")
    ax[0].set_xlabel("friction coefficient μ"); ax[0].set_ylabel("max step [m]"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[0].set_title("tank: step limit vs friction")
    ax[1].plot(taus, hb, "s-C3"); ax[1].axhline(0.63, color="gray", ls="--", label="range-of-motion limit 0.63 m")
    ax[1].set_xlabel("knee & hip torque limit [N m]"); ax[1].set_ylabel("max step [m]"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[1].set_title("biped: step limit vs joint torque (with trunk lean)")
    fig.tight_layout(); fig.savefig(OUT / "D13_step_sensitivity.png", dpi=150); plt.close(fig)


# =============================================================================
def loop_rotation(robot, c1, c2, R, n=361):
    """関節空間の円ループ 1 周の正味の回転（線積分）"""
    ph = np.linspace(0, 2 * np.pi, n)
    q = np.c_[c1 + R * np.cos(ph), c2 + R * np.sin(ph)]
    dq = np.diff(q, axis=0); mid = (q[1:] + q[:-1]) / 2
    return -np.sum(np.sum(np.array([robot.connection(m) for m in mid]) * dq, 1))


def D14():
    """AMBAC ループによる慣性パラメータの自己同定の可能性（第 2 段階の計画）"""
    def make(p):
        ma, ml, kI = p
        mb = 76.0 - ma - ml
        return PlanarRobot(mb, kI * mb * (0.35**2 + 1) / 12, [Limb((0, 0.35), ma, 0.7), Limb((0, -0.15), ml, 0.9)])
    p0 = np.array([4.0, 12.0, 1.0])            # 腕の質量, 脚の質量, 胴体の慣性の倍率
    names = ["arm mass", "leg mass", "torso inertia"]
    loops = [(-20, -40, 70), (-60, -60, 40), (40, -60, 40), (0, -90, 25), (-90, -20, 30), (60, -20, 30)]
    loops = [(RAD(a), RAD(b), RAD(c)) for a, b, c in loops]
    th0 = np.array([loop_rotation(make(p0), *L) for L in loops])
    J = np.zeros((len(loops), 3))
    for k in range(3):
        dp = np.zeros(3); dp[k] = 0.01 * p0[k]
        J[:, k] = (np.array([loop_rotation(make(p0 + dp), *L) for L in loops]) - th0) / (0.01 * p0[k])  # 相対 1 あたり
    sig = RAD(0.1); Nrep = 10                   # 1 周あたりの姿勢の計測誤差 0.1°, 各ループ 10 周
    cov = sig**2 / Nrep * np.linalg.inv(J.T @ J)
    rel = np.sqrt(np.diag(cov))
    cond = np.linalg.cond(J)
    log("[D14] 6 種類のループの 1 周あたりの回転: " + ", ".join(f"{D(x):+.2f}°" for x in th0))
    log("[D14] 相対誤差 1% あたりの回転の変化 [°]: " + "; ".join(f"{n}: " + ", ".join(f"{D(J[i,k])*0.01:+.3f}" for i in range(len(loops))) for k, n in enumerate(names)))
    log(f"[D14] 計測誤差 0.1°・各 10 周で推定した場合の相対標準偏差: 腕 {rel[0]*100:.1f}%, 脚 {rel[1]*100:.1f}%, 胴体の慣性 {rel[2]*100:.1f}%（条件数 {cond:.1f}）")
    # 周回中の姿勢の波形（各ループ 36 点）を使う場合
    def wave(robot, c1, c2, R, n=36):
        ph = np.linspace(0, 2 * np.pi, 361)
        q = np.c_[c1 + R * np.cos(ph), c2 + R * np.sin(ph)]
        dq = np.diff(q, axis=0); mid = (q[1:] + q[:-1]) / 2
        th = np.concatenate([[0], np.cumsum(-np.sum(np.array([robot.connection(m) for m in mid]) * dq, 1))])
        return th[::10][1:n + 1]
    w0 = np.concatenate([wave(make(p0), *L) for L in loops])
    Jw = np.zeros((len(w0), 3))
    for k in range(3):
        dp = np.zeros(3); dp[k] = 0.01 * p0[k]
        Jw[:, k] = (np.concatenate([wave(make(p0 + dp), *L) for L in loops]) - w0) / (0.01 * p0[k])
    covw = sig**2 / Nrep * np.linalg.inv(Jw.T @ Jw)
    relw = np.sqrt(np.diag(covw))
    log(f"[D14] 周回中の姿勢の波形（各ループ 36 点）も使う場合の相対標準偏差: 腕 {relw[0]*100:.2f}%, 脚 {relw[1]*100:.2f}%, 胴体の慣性 {relw[2]*100:.2f}%（条件数 {np.linalg.cond(Jw):.1f}）")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(loops)); w = 0.27
    for k, n in enumerate(names):
        ax[0].bar(x + (k - 1) * w, D(J[:, k]) * 0.01, w, label=n)
    ax[0].set_xticks(x, [f"L{i+1}" for i in x]); ax[0].set_ylabel("change of rotation per cycle [deg] per +1%")
    ax[0].set_title("sensitivity of loop rotation to inertial parameters"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis="y")
    sigs = np.array([0.02, 0.05, 0.1, 0.2, 0.5])
    for k, n in enumerate(names):
        ax[1].loglog(sigs, rel[k] * sigs / 0.1 * 100, "o-", color=f"C{k}", label=f"{n} (net rotation only)")
        ax[1].loglog(sigs, relw[k] * sigs / 0.1 * 100, "s--", color=f"C{k}", label=f"{n} (in-cycle waveform)")
    ax[1].set_xlabel("attitude measurement error per cycle [deg]"); ax[1].set_ylabel("estimated relative std [%] (10 cycles/loop)")
    ax[1].set_title("identifiability from 6 AMBAC loops"); ax[1].legend(fontsize=6); ax[1].grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(OUT / "D14_self_identification.png", dpi=150); plt.close(fig)


ALL = [D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14]
if __name__ == "__main__":
    pick = [f for f in ALL if len(sys.argv) == 1 or f.__name__ in sys.argv[1:]]
    for f in pick:
        t0 = time.time(); f(); print(f"   ({f.__name__}: {time.time() - t0:.0f} s)", flush=True)
    path = DATA / "detail_summary.txt"
    old = path.read_text(encoding="utf-8").splitlines() if path.exists() and len(sys.argv) > 1 else []
    keep = [l for l in old if not any(l.startswith(f"[{f.__name__}]") for f in pick)]
    path.write_text("\n".join(sorted(keep + LOG, key=lambda s: int(s[2:s.index("]")]))) + "\n", encoding="utf-8")
