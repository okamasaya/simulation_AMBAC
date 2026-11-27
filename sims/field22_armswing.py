"""Field22: 歩くときに腕を振って、足のねじれの滑り（ヨー）を打ち消す（下半身の段 5、上半身の AMBAC を歩行へ）.

段階 21 では、ZMP の計画で開ループのまま歩けたが、1 歩 0.48〜0.60 s では支持する足が鉛直軸まわりに滑って向きがずれた（F37）。
振り出す脚の前後の加速が、上体の鉛直軸まわりの角運動量を変え、その反作用（ヨーのモーメント）を足裏の摩擦が支えきれない。
人は腕を脚と逆に振って、この角運動量を打ち消している（腕の振りの役目として知られる説: Collins ら 2009、Herr & Popovic 2008）。
ここでは段階 21 の機体に腕（肩の前後の振り、片腕 4 kg・0.6 m、上体 56 kg の中から）を付け、

  1 層目: 計画（関節角の列）から全身の鉛直軸まわりの角運動量 H_z と、足圧中心まわりのヨーのモーメント M_z を計算し、
          M_z の 2 乗和を最小にする腕の振り幅 A*（右腕の角 = A × (左足 − 右足) / 歩幅）を最小二乗で求める
  2 層目: MuJoCo で歩かせ、床から受けるヨーのモーメント（足圧中心まわり）、最後の向きのずれ、歩ける 1 歩の時間、
          歩くのに必要な摩擦係数 μ を、腕なし・腕を固定・A* の半分・A*・1.5 倍で比べる

実行: python field22_armswing.py [main|errors|figure]
"""
import json
import sys
import time

import matplotlib.pyplot as plt
import mujoco
import numpy as np

import field21_walk as W
from common import DATA

T0 = time.time()
TS = W.TS_ERR                     # 0.32〜0.80 s、0.04 s 刻み
OUT = W.OUT


def cfg(T=0.52, A=None, **kw):
    ex = dict(kw.pop("extra", {}))
    if A is not None:
        ex["arms"] = {**ex.get("arms", {}), "A": float(A)}
    return W.Cfg(T_step=T, extra=ex, **kw)


def plan_yaw(c):
    """1 層目: 計画の関節角の列から H_z（重心まわり）と足圧中心（ZMP の目標）まわりの M_z"""
    P = W.plan(c); rb = P["rb"]; m, d = rb.m, rb.d
    t = P["t"]; dt = t[1] - t[0]; Q = P["Q"]
    H = np.zeros(len(t)); C = np.zeros((len(t), 3)); Mtot = mujoco.mj_getTotalmass(m)
    v = np.zeros(m.nv)
    for k in range(len(t)):
        a, b = max(k - 1, 0), min(k + 1, len(t) - 1)
        mujoco.mj_differentiatePos(m, v, (b - a) * dt, Q[a], Q[b])
        d.qpos[:] = Q[k]; d.qvel[:] = v
        mujoco.mj_forward(m, d); mujoco.mj_subtreeVel(m, d)
        H[k] = d.subtree_angmom[1][2]; C[k] = d.subtree_com[1]
    dH = np.gradient(H, dt)
    acc = np.gradient(np.gradient(C[:, :2], dt, axis=0), dt, axis=0)
    p = P["zref"]                                   # 足圧中心 ≒ ZMP の目標
    r = C[:, :2] - p
    Mz = dH + Mtot * (r[:, 0] * acc[:, 1] - r[:, 1] * acc[:, 0])
    single = np.array([s_ != "LR" for s_ in P["support"]])
    return dict(t=t, H=H, Mz=Mz, single=single, P=P)


def best_A(T, A_probe=30.0, **kw):
    """M_z(A) ≒ M_z(0) + A × g を仮定して、片脚支持の区間の M_z の 2 乗和を最小にする A*"""
    y0 = plan_yaw(cfg(T, 0.0, **kw)); y1 = plan_yaw(cfg(T, A_probe, **kw))
    g = (y1["Mz"] - y0["Mz"]) / A_probe; sm = y0["single"]
    A = -np.sum(g[sm] * y0["Mz"][sm]) / np.sum(g[sm] ** 2)
    yA = plan_yaw(cfg(T, A, **kw))
    rms = lambda y: float(np.sqrt(np.mean(y["Mz"][y["single"]] ** 2)))
    pk = lambda y: float(np.abs(y["Mz"][y["single"]]).max())
    return dict(T=T, A=float(A), rms0=rms(y0), rmsA=rms(yA), pk0=pk(y0), pkA=pk(yA), lin_err=float(rms(yA) / max(rms(y0), 1e-9)))


def run(args):
    """MuJoCo で 1 回: (T, A or None, 上書き) → 要約（床のヨーのモーメントを含む）"""
    T, A, over = args
    c = cfg(T, A, **over)
    try:
        P = W.plan(c); ex = W.execute(c, P, record=True)
        r = ex.pop("rec")
        sing = np.array([P["support"][min(int(tt / (P["t"][1] - P["t"][0])), len(P["t"]) - 1)] != "LR" for tt in r["t"]])
        mz = np.asarray(r["mz"]); ok_ = sing & np.isfinite(mz)
        ex["mz_rms"] = float(np.sqrt(np.mean(mz[ok_] ** 2))) if ok_.any() else np.nan
        ex["mz_pk"] = float(np.percentile(np.abs(mz[ok_]), 99)) if ok_.any() else np.nan
        tau = np.abs(np.asarray(r["tau"]))
        ex["sh_tau"] = float(tau[:, len(W.JOINTS):].max()) if tau.shape[1] > len(W.JOINTS) else 0.0
    except Exception as e:  # noqa: BLE001
        ex = dict(ok=False, fell=True, err=str(e))
    ex.update(T=T, A=A, over=over)
    return ex


def mu_min(args):
    """歩ける最小の摩擦係数（二分法、[0.1, 1.6]）。単調でなければ注意を出す"""
    T, A, over, it = args
    okf = lambda mu: run((T, A, {**over, "mu": mu}))["ok"]
    lo, hi = 0.1, 1.6
    if not okf(hi):
        return (np.nan, hi)
    for _ in range(it):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if okf(mid) else (mid, hi)
    return (hi, lo)


def seq(rs):
    return "".join("○" if r["ok"] else "×" for r in rs)


def main():
    from multiprocessing import Pool
    out = {}
    print("=== 1 層目: 計画から求めた最適な腕の振り幅 A*（片脚支持の区間の足圧中心まわりのヨーのモーメント M_z を最小に）===")
    bests = []
    for T in (0.40, 0.52, 0.64, 0.80):
        b = best_A(T); bests.append(b)
        print(f"  T {T:.2f} s: A* = {b['A']:.1f}°、M_z の RMS {b['rms0']:.1f} → {b['rmsA']:.1f} N·m、最大 {b['pk0']:.1f} → {b['pkA']:.1f} N·m", flush=True)
    out["best"] = bests
    A_star = float(np.round(np.mean([b["A"] for b in bests]), 1))
    out["A_star"] = A_star
    print(f"  → 以下、A* = {A_star}°（4 つの T の平均）を使う")
    cases = [("腕なし（段階 21）", None), ("腕を固定（A 0）", 0.0), ("A*/2", A_star / 2), ("A*", A_star), ("1.5 A*", 1.5 * A_star), ("逆向き −A*", -A_star)]
    with Pool(4) as pool:
        res = pool.map(run, [(float(T), A, {}) for _, A in cases for T in TS], chunksize=2)
        mus = pool.map(mu_min, [(T, A, {}, 6) for T in (0.52, 0.80) for _, A in cases[1:5]])
    n = len(TS); out["scan"] = []
    print(f"=== 2 層目: MuJoCo、歩幅 0.25 m、T_step {TS[0]}〜{TS[-1]} s（0.04 s 刻み、左が速い） ===")
    print("| 腕 | 成功の並び | 成功の割合 | 最後の向きのずれ（T 0.52 / 0.80） | 床のヨーのモーメント RMS（T 0.52 / 0.80）| 肩のトルクの最大 |")
    for i, (name, A) in enumerate(cases):
        rs = res[i * n:(i + 1) * n]
        g = lambda T, k: next(r.get(k, np.nan) for r in rs if abs(r["T"] - T) < 1e-6)
        print(f"| {name} | {seq(rs)} | {np.mean([r['ok'] for r in rs]):.2f} | {g(0.52, 'yaw'):.1f}° / {g(0.80, 'yaw'):.1f}° | "
              f"{g(0.52, 'mz_rms'):.1f} / {g(0.80, 'mz_rms'):.1f} N·m | {max(r.get('sh_tau', 0) for r in rs):.1f} N·m |", flush=True)
        out["scan"].append(dict(name=name, A=A, res=[{k: v for k, v in r.items() if k != "over"} for r in rs]))
    print("=== 歩くのに必要な最小の摩擦係数 μ（二分法 6 回、±0.012） ===")
    out["mu"] = []
    for (T, (name, A)), r in zip([(T, ca) for T in (0.52, 0.80) for ca in cases[1:5]], mus):
        print(f"  T {T:.2f} s、{name}: μ_min = {r[0]:.3f}", flush=True)
        out["mu"].append(dict(T=T, name=name, A=A, mu=r))
    json.dump(out, open(DATA / "field22_armswing.json", "w"), indent=1, ensure_ascii=False, default=float)
    print(f"[{time.time() - T0:.0f} s]")


VARIANTS = {
    "基準": {},
    "計算: 時間刻み 0.25 ms": {"dt": 2.5e-4},
    "系統: 腕の質量 3 kg": {"extra": {"arms": {"m": 3.0}}},
    "系統: 腕の質量 5 kg": {"extra": {"arms": {"m": 5.0}}},
    "系統: 肩のサーボ kp 1e3": {"extra": {"arms": {"kp": 1e3}}},
    "系統: 接触を柔らかく（solref 0.02）": {"solref": "0.02 1"},
    "系統: 骨盤の高さ 0.88": {"z_hip": 0.88},
    "系統: 摩擦 μ 0.5": {"mu": 0.5},
}


def errors_main():
    from multiprocessing import Pool
    A_star = json.load(open(DATA / "field22_armswing.json"))["A_star"]
    rng = np.random.default_rng(2)
    mc = [{"trunk": 1 + rng.uniform(-.1, .1), "thigh": 1 + rng.uniform(-.1, .1), "shank": 1 + rng.uniform(-.1, .1),
           "foot": 1 + rng.uniform(-.15, .15)} for _ in range(8)]
    sets = list(VARIANTS.items()) + [(f"入力: 質量 {i}", {"extra": {"mass": m_}}) for i, m_ in enumerate(mc)]
    jobs = []
    for _, ov in sets:
        for A in (0.0, A_star):
            ov2 = json.loads(json.dumps(ov)); ex = ov2.pop("extra", {})
            arms = {**ex.get("arms", {}), "A": A}
            jobs += [(float(T), None, {**ov2, "extra": {**ex, "arms": arms}}) for T in TS]
    with Pool(4) as pool:
        res = pool.map(run, jobs, chunksize=2)
    n = len(TS); out = []
    print(f"=== 誤差の評価: 腕を固定（A 0）と A* = {A_star}° の成功の並び（T {TS[0]}〜{TS[-1]} s、左が速い） ===")
    for i, (name, ov) in enumerate(sets):
        r0 = res[(2 * i) * n:(2 * i + 1) * n]; r1 = res[(2 * i + 1) * n:(2 * i + 2) * n]
        f0, f1 = np.mean([r["ok"] for r in r0]), np.mean([r["ok"] for r in r1])
        print(f"| {name} | {seq(r0)} {f0:.2f} | {seq(r1)} {f1:.2f} | {f1 - f0:+.2f} |", flush=True)
        out.append(dict(name=name, over=ov, f0=f0, f1=f1, s0=seq(r0), s1=seq(r1)))
    d_mc = np.array([o["f1"] - o["f0"] for o in out if o["name"].startswith("入力")])
    print(f"入力（質量 8 回）: 成功の割合の増え {d_mc.mean():+.2f} ± {d_mc.std(ddof=1):.2f}")
    json.dump(out, open(DATA / "field22_errors.json", "w"), indent=1, ensure_ascii=False, default=float)
    print(f"[{time.time() - T0:.0f} s]")


def figure():
    res = json.load(open(DATA / "field22_armswing.json")); A_star = res["A_star"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for A, col, lab in ((0.0, "tab:gray", "arms fixed"), (A_star, "tab:orange", f"arm swing A*={A_star:.0f}°")):
        y = plan_yaw(cfg(0.52, A))
        ax[0].plot(y["t"], np.where(y["single"], y["Mz"], np.nan), color=col, lw=1, label=f"plan, {lab}")
        ex = W.execute(cfg(0.52, A), y["P"], record=True); r = ex["rec"]
        ax[0].plot(r["t"], r["mz"], color=col, lw=0.5, alpha=0.5, label=f"MuJoCo, {lab}")
    ax[0].set_ylim(-150, 150); ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("yaw moment about CoP M_z [N·m]")
    ax[0].set_title("T_step 0.52 s: yaw moment the feet must hold", fontsize=9); ax[0].legend(fontsize=7)
    names = [s_["name"] for s_ in res["scan"]]
    lab_en = {"腕なし（段階 21）": "no arms (stage 21)", "腕を固定（A 0）": "arms fixed", "A*/2": "A*/2", "A*": "A*", "1.5 A*": "1.5 A*", "逆向き −A*": "reversed −A*"}
    M = np.array([[1 if r["ok"] else 0 for r in s_["res"]] for s_ in res["scan"]])
    ax[1].imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels([lab_en.get(n_, n_) for n_ in names], fontsize=8)
    ax[1].set_xticks(range(len(TS))); ax[1].set_xticklabels([f"{t:.2f}" for t in TS], fontsize=7)
    ax[1].set_xlabel("step time T_step [s] (L 0.25 m)"); ax[1].set_title("8 steps OK (green) / fail (red)", fontsize=9)
    for s_, col in zip(res["scan"], ("k", "tab:gray", "tab:blue", "tab:orange", "tab:green", "tab:red")):
        ax[2].plot(TS, [abs(r.get("yaw", np.nan)) if not r.get("fell") else np.nan for r in s_["res"]], "o-", color=col, ms=3, label=lab_en.get(s_["name"], s_["name"]))
    ax[2].set_xlabel("step time T_step [s]"); ax[2].set_ylabel("|heading error| after 8 steps [deg]"); ax[2].legend(fontsize=7)
    ax[2].set_title("heading drift (yaw slip of stance foot)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "field22_armswing.png", dpi=130)
    print("図 → figures/field/field22_armswing.png")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "main"
    {"main": main, "errors": errors_main, "figure": figure}[arg]()
