"""Field23: 歩行の閉ループの安定化（下半身の段 6）.

段階 21・22 の歩行は開ループ（計画した関節角を位置サーボで追うだけ）。歩幅 0.35 m 以上を速く歩くと倒れ、外から押されても直せない。
ここでは、歩行の安定化の標準的な考え方の 1 つ、
  DCM（発散成分、capture point）ξ = c + ċ/ω（ω = √(g/z_c)）のずれを見て直す（Englsberger ら 2015、Kajita ら 2010 の安定化と同じ型）
を、位置サーボのままで入れる:
  10 ms ごとに MuJoCo の重心の位置・速度から ξ を測り、計画の ξ_ref とのずれ e に比例して骨盤を水平に Δ = −K e ずらす
  （足は床に固定のまま、Δ に対する脚の関節角の変化 S = ∂q/∂Δ を計画のときに逆運動学で求めておく。|Δ| ≤ 8 cm）。
骨盤を後ろへずらす動きの反作用で、ZMP が前へ動き、前へずれた重心を押し戻す。
腕は段階 22 の振り（A* = 24.6°）をつけたまま。

比べること
  (1) 歩幅 L × 1 歩の時間 T の成功の地図: K = 0（開ループ）, 1, 3, 6
  (2) 押されたときの回復: 4 歩目の片脚支持の中ごろに、上体を前・横に 0.1 s 押す。倒れない最大の力
  (3) 誤差: 制御の遅れ、測定の雑音、制御の周期、時間刻み、接触、質量
実行: python field23_stabilizer.py [main|errors|push|noise|figure]
"""
import json
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

import field21_walk as W
from common import DATA

T0 = time.time()
OUT = W.OUT
A_STAR = 24.6
KS = [0.0, 1.0, 3.0, 6.0]
LS = [0.25, 0.35, 0.45]
TS = [1.0, 0.8, 0.7, 0.6, 0.52, 0.44, 0.40, 0.36]
FS = [0, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1200]


def run(args):
    """(上書き, 安定化の設定 or None, 押す力 or None) → 要約"""
    over, st, push = args
    over = json.loads(json.dumps(over))
    ex = over.pop("extra", {})
    ex = {**ex, "arms": {**ex.get("arms", {}), "A": ex.get("arms", {}).get("A", A_STAR)}}
    if st is not None:
        ex["stab"] = st
    if push is not None:
        ex["push"] = push
    c = W.Cfg(extra=ex, **over)
    try:
        r = W.execute(c)
    except Exception as e:  # noqa: BLE001
        r = dict(ok=False, fell=True, err=str(e))
    r.update(over=over, st=st, push=push)
    return r


def push_time(T, L=0.25):
    """4 歩目の片脚支持の中ごろ"""
    c = W.Cfg(T_step=T, L=L)
    steps, _, _ = W.footsteps(c)
    t0, t1 = steps[3][0], steps[3][1]
    return float((t0 + t1) / 2 - 0.05)


def grid_jobs(over, K, st_extra=None):
    return [({**over, "L": L, "T_step": T}, {"K": K, **(st_extra or {})} if K is not None else None, None) for L in LS for T in TS]


def push_jobs(over, K, T=0.8, st_extra=None):
    tp = push_time(T)
    return [({**over, "T_step": T}, {"K": K, **(st_extra or {})}, {"t": tp, "F": [F * np.cos(a), F * np.sin(a)], "dur": 0.1})
            for a in (0.0, np.pi / 2) for F in FS]


def push_limit(rs):
    """押す力を小さい方から並べて、最初に失敗する手前の力（それ以下はすべて成功）"""
    lim = 0
    for F, r in zip(FS, rs):
        if not r["ok"]:
            break
        lim = F
    return lim, "".join("○" if r["ok"] else "×" for r in rs)


def seq(rs):
    return "".join("○" if r["ok"] else "×" for r in rs)


def main():
    from multiprocessing import Pool
    jobs = []
    for K in KS:
        jobs += grid_jobs({}, K)
    for K in KS:
        jobs += push_jobs({}, K)
    with Pool(4) as pool:
        res = pool.map(run, jobs, chunksize=2)
    ng = len(LS) * len(TS); npu = 2 * len(FS)
    out = {"grid": [], "push": [], "TS": TS, "LS": LS, "FS": FS, "KS": KS}
    print(f"=== 成功の地図（腕の振り A* = {A_STAR}°）。行 = 歩幅 L、並び = T {TS}（左が遅い） ===")
    for a, K in enumerate(KS):
        rs = res[a * ng:(a + 1) * ng]
        print(f"K = {K}: 成功の割合 {np.mean([r['ok'] for r in rs]):.2f}")
        for b, L in enumerate(LS):
            rr = rs[b * len(TS):(b + 1) * len(TS)]
            dc = np.nanmean([r.get("dcm_rms", np.nan) for r in rr if not r.get("fell")]) if any(not r.get("fell") for r in rr) else np.nan
            print(f"   L {L:.2f}: {seq(rr)}  最速の成功 {min([T for T, r in zip(TS, rr) if r['ok']], default=np.nan):.2f} s、DCM のずれの RMS {dc * 100:.1f} cm", flush=True)
        out["grid"].append(dict(K=K, ok=[[r["ok"] for r in rs[b * len(TS):(b + 1) * len(TS)]] for b in range(len(LS))],
                                dcm=[r.get("dcm_rms", np.nan) for r in rs]))
    print(f"=== 押されたときの回復（T 0.8 s、L 0.25 m、4 歩目の片脚支持の中ごろに 0.1 s、力 {FS} N）===")
    base = len(KS) * ng
    for a, K in enumerate(KS):
        rs = res[base + a * npu: base + (a + 1) * npu]
        fx, sx = push_limit(rs[:len(FS)]); fy, sy = push_limit(rs[len(FS):])
        print(f"K = {K}: 前へ {fx} N（{fx * 0.1:.0f} N·s） {sx}  /  横へ {fy} N（{fy * 0.1:.0f} N·s） {sy}", flush=True)
        out["push"].append(dict(K=K, fwd=fx, lat=fy, sfwd=sx, slat=sy))
    fr = [np.mean(np.array(g["ok"], float)) for g in out["grid"]]
    out["K_best"] = float(KS[int(np.argmax(fr))])
    json.dump(out, open(DATA / "field23_stabilizer.json", "w"), indent=1, ensure_ascii=False, default=float)
    print(f"[{time.time() - T0:.0f} s]")


VARIANTS = {
    "基準（K 1、周期 10 ms、遅れ 0）": ({}, {}),
    "計算: 時間刻み 0.25 ms": ({"dt": 2.5e-4}, {}),
    "系統: 制御の遅れ 10 ms": ({}, {"delay": 0.01}),
    "系統: 制御の遅れ 30 ms": ({}, {"delay": 0.03}),
    "系統: 制御の周期 5 ms / 20 ms": None,
    "系統: 制御の周期 2 ms": ({}, {"dt_c": 0.002}),
    "参考: K 3・周期 2 ms": ({}, {"dt_c": 0.002, "K": 3.0}),
    "参考: K 6・周期 2 ms": ({}, {"dt_c": 0.002, "K": 6.0}),
    "系統: 測定の雑音（位置 5 mm、速度 5 cm/s）": ({}, {"noise": 0.005, "noise_v": 0.05}),
    "系統: 接触を柔らかく（solref 0.02）": ({"solref": "0.02 1"}, {}),
    "系統: サーボ kp 1e4": ({"kp": 1e4}, {}),
    "系統: 摩擦 μ 0.5": ({"mu": 0.5}, {}),
    "系統: 腕を振らない": ({"extra": {"arms": {"A": 0.0}}}, {}),
}


def errors_main():
    from multiprocessing import Pool
    K = json.load(open(DATA / "field23_stabilizer.json")).get("K_best", 3.0)
    items = []
    for name, v in VARIANTS.items():
        if v is None:
            items += [(name + " 5 ms", {}, {"dt_c": 0.005}), (name + " 20 ms", {}, {"dt_c": 0.02})]
        else:
            items.append((name, v[0], v[1]))
    rng = np.random.default_rng(3)
    for i in range(8):
        ms = {"trunk": 1 + rng.uniform(-.1, .1), "thigh": 1 + rng.uniform(-.1, .1), "shank": 1 + rng.uniform(-.1, .1), "foot": 1 + rng.uniform(-.15, .15)}
        items.append((f"入力: 質量 {i}", {"extra": {"mass": ms}}, {}))
    jobs = []
    for _, ov, se in items:
        jobs += grid_jobs(ov, K, se) + push_jobs(ov, K, st_extra=se)
    with Pool(4) as pool:
        res = pool.map(run, jobs, chunksize=2)
    ng = len(LS) * len(TS); npu = 2 * len(FS); nb = ng + npu
    print(f"=== 誤差の評価（K = {K}）: 地図の成功の割合、押されて倒れない最大の力（前 / 横） ===")
    out = []
    for i, (name, ov, se) in enumerate(items):
        rs = res[i * nb:(i + 1) * nb]
        fr = np.mean([r["ok"] for r in rs[:ng]])
        fx, _ = push_limit(rs[ng:ng + len(FS)]); fy, _ = push_limit(rs[ng + len(FS):])
        print(f"| {name} | {fr:.2f} | {fx} / {fy} N | " + " ".join(seq(rs[b * len(TS):(b + 1) * len(TS)]) for b in range(len(LS))), flush=True)
        out.append(dict(name=name, frac=fr, fwd=fx, lat=fy))
    mc = [o for o in out if o["name"].startswith("入力")]
    f_ = np.array([o["frac"] for o in mc]); x_ = np.array([o["fwd"] for o in mc]); y_ = np.array([o["lat"] for o in mc])
    print(f"入力（質量 8 回）: 成功の割合 {f_.mean():.2f} ± {f_.std(ddof=1):.2f}、前 {x_.mean():.0f} ± {x_.std(ddof=1):.0f} N、横 {y_.mean():.0f} ± {y_.std(ddof=1):.0f} N")
    json.dump(out, open(DATA / "field23_errors.json", "w"), indent=1, ensure_ascii=False, default=float)
    print(f"[{time.time() - T0:.0f} s]")


DIRS = {"前 +x": (1, 0), "後ろ −x": (-1, 0), "横（次の足の側）+y": (0, 1), "横（外側）−y": (0, -1)}


def predict_push():
    """1 層目（LIPM の capture point）: 押された直後の DCM のずれ J/(M ω) が、ZMP の目標から足の端までの距離 d を超えると
    足首（足裏の中の ZMP）だけでは戻せない → J_max = M ω d"""
    import mujoco
    c = W.Cfg(extra={"arms": {"A": A_STAR}}); P = W.plan(c)
    M = mujoco.mj_getTotalmass(P["rb"].m); om = np.sqrt(W.G / P["zc"])
    xc = (W.TOE - W.HEEL) / 2                     # ZMP の目標（足首から前へ）
    d = {"前 +x": W.TOE - xc, "後ろ −x": W.HEEL + xc, "横（次の足の側）+y": W.FOOT_W / 2, "横（外側）−y": W.FOOT_W / 2}
    return {k: float(M * om * v) for k, v in d.items()}, float(M), float(om)


def push_main():
    """押す向き 4 つ × K（0、最良）。1 層目の予測と比べる"""
    from multiprocessing import Pool
    Kb = json.load(open(DATA / "field23_stabilizer.json")).get("K_best", 1.0)
    tp = push_time(0.8)
    jobs = [({"T_step": 0.8}, {"K": K}, {"t": tp, "F": [dx * F, dy * F], "dur": 0.1}) for K in (0.0, Kb) for dx, dy in DIRS.values() for F in FS]
    with Pool(4) as pool:
        res = pool.map(run, jobs, chunksize=2)
    pred, M, om = predict_push()
    print(f"=== 押されたときの回復（T 0.8 s、L 0.25 m、4 歩目＝左足を振り出す間、右足で立つ。0.1 s）: 倒れない最大の力積 [N·s]（分解能 10 N·s）===")
    print(f"1 層目の予測 J_max = M ω d（M {M:.1f} kg、ω {om:.2f} 1/s、d = ZMP の目標から足の端まで）")
    out = {"pred": pred, "res": []}
    n = len(FS)
    for a, K in enumerate((0.0, Kb)):
        for b, name in enumerate(DIRS):
            rs = res[(a * len(DIRS) + b) * n:(a * len(DIRS) + b + 1) * n]
            lim, sq = push_limit(rs)
            print(f"  K {K}: {name}: {lim * 0.1:.0f}〜{(FS[FS.index(lim) + 1] if lim < FS[-1] else lim) * 0.1:.0f} N·s  {sq}  （予測 {pred[name]:.1f} N·s）", flush=True)
            out["res"].append(dict(K=K, dir=name, lim=lim * 0.1, seq=sq))
    json.dump(out, open(DATA / "field23_push.json", "w"), indent=1, ensure_ascii=False, default=float)


def noise_main():
    """測定の雑音と低域通過フィルタ（K 1、3 つの歩き方）"""
    from multiprocessing import Pool
    cases = [{"noise": 0.001, "noise_v": 0.01}, {"noise": 0.005, "noise_v": 0.0}, {"noise": 0.0, "noise_v": 0.05},
             {"noise": 0.005, "noise_v": 0.05}, {"noise": 0.005, "noise_v": 0.05, "lpf": 0.03}, {"noise": 0.005, "noise_v": 0.05, "lpf": 0.1}, {"lpf": 0.03}]
    walks = ((0.25, 0.8), (0.35, 0.6), (0.45, 0.7))
    jobs = [({"L": L, "T_step": T}, {"K": 1.0, **cs}, None) for cs in cases for L, T in walks]
    with Pool(4) as pool:
        res = pool.map(run, jobs)
    print(f"=== 測定の雑音（白色、10 ms ごと）と低域通過フィルタ。K 1、歩き方 (L, T) = {walks} ===")
    out = []
    for i, cs in enumerate(cases):
        rs = res[i * len(walks):(i + 1) * len(walks)]
        print(f"  {cs}: {seq(rs)}", flush=True)
        out.append(dict(case=cs, ok=[r["ok"] for r in rs]))
    json.dump(out, open(DATA / "field23_noise.json", "w"), indent=1, ensure_ascii=False, default=float)


def figure():
    res = json.load(open(DATA / "field23_stabilizer.json"))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    M = np.array([np.array(g["ok"], float).ravel() for g in res["grid"]])
    ax[0].imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax[0].set_yticks(range(len(res["KS"]))); ax[0].set_yticklabels([f"K = {k:g}" + (" (open loop)" if k == 0 else "") for k in res["KS"]])
    lab = [f"{L:.2f}/{T:.2f}" for L in res["LS"] for T in res["TS"]]
    ax[0].set_xticks(range(len(lab))); ax[0].set_xticklabels(lab, rotation=90, fontsize=6)
    for b in range(1, len(res["LS"])):
        ax[0].axvline(b * len(res["TS"]) - 0.5, color="k", lw=1)
    ax[0].set_xlabel("step length L [m] / step time T [s]"); ax[0].set_title("8 steps OK (green) / fail (red), arm swing on", fontsize=9)
    pu = json.load(open(DATA / "field23_push.json")); names = list(DIRS); en = ["forward", "backward", "lateral (to next foot)", "lateral (outward)"]
    xx = np.arange(len(names)); Ks2 = sorted({r["K"] for r in pu["res"]})
    for j, K in enumerate(Ks2):
        v = [next(r["lim"] for r in pu["res"] if r["K"] == K and r["dir"] == nm) for nm in names]
        ax[1].bar(xx + (j - 0.5) * 0.35, v, 0.35, label=f"MuJoCo K = {K:g}" + (" (open loop)" if K == 0 else ""), yerr=[[0] * len(v), [10] * len(v)], capsize=2)
    ax[1].plot(xx, [pu["pred"][nm] for nm in names], "k_", ms=30, mew=2, label="layer 1: M ω d (ankle strategy limit)")
    ax[1].set_xticks(xx); ax[1].set_xticklabels(en, fontsize=7); ax[1].set_ylabel("max impulse survived [N·s] (0.1 s push)"); ax[1].legend(fontsize=7)
    ax[1].set_title("push recovery (T 0.8 s, L 0.25 m, mid single support on right foot)", fontsize=9)
    for K, col in zip((0.0, res.get("K_best", 3.0)), ("tab:gray", "tab:blue")):
        tp = push_time(0.8); F = 200.0
        c = W.Cfg(T_step=0.8, extra={"arms": {"A": A_STAR}, "stab": {"K": K}, "push": {"t": tp, "F": [0.0, -F], "dur": 0.1}})
        r = W.execute(c, record=True)["rec"]
        ax[2].plot(r["t"], r["com"][:, 1], color=col, label=f"K = {K:g}, outward push {F:.0f} N × 0.1 s")
    ax[2].axvspan(tp, tp + 0.1, color="tab:red", alpha=0.2)
    ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("CoM y [m]"); ax[2].legend(fontsize=7); ax[2].set_ylim(-0.4, 0.4)
    ax[2].set_title("lateral CoM after an outward push (−y, 20 N·s)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "field23_stabilizer.png", dpi=130)
    print("図 → figures/field/field23_stabilizer.png")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "main"
    {"main": main, "errors": errors_main, "push": push_main, "noise": noise_main, "figure": figure}[arg]()
