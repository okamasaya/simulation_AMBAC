"""Field24: 押されたら足を踏み出す（足の置き場所の戦略、下半身の段 7）.

段階 23 の安定化（DCM のずれ → 骨盤をずらす）は、押されたときの前後の回復が capture point の M ω d（足首の戦略の限界）で止まった。
それより強く押されたら、人は足を踏み出す。ここでは
  DCM のずれ e が 3 cm を超えたら、次に着地する足を e × exp(ω × 着地までの時間) だけずらして（最大 0.3 m、脚は交差させない）、
  その場で歩行の計画（ZMP プレビュー制御・逆運動学）を作り直す。着地の 0.15 s 前まで 50 ms ごとに足し直し、
  着地したら次の足で続けて直す（複数歩で止まる）
を、段階 23 の安定化（K 1）に加える。計画し直す計算の時間は 0 とみなした（実機では未確認）。

1 層目（LIPM の capture point、1 歩で止まる条件）:
  足首で ZMP を足の端 d まで動かしながら、着地までの Δt の間に DCM のずれが育つ: e(t) = d + (e0 − d) e^{ωt}
  着地のときのずれが d + 踏み出し δ_max 以下なら止まれる → e0 ≤ d + δ_max e^{−ωΔt} → J_max = M ω (d + δ_max e^{−ωΔt})
実行: python field24_stepping.py [main|errors|from_state|figure]
"""
import json
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

import field21_walk as W
import field23_stabilizer as F23
from common import DATA

T0 = time.time()
OUT = W.OUT
FS = [300, 400, 500, 600, 700]
DIRS = {"前 +x": (1, 0), "後ろ −x": (-1, 0), "横（振り出す足の側）+y": (0, 1)}
STEP = {"K": 1.0, "step": True, "step_max": 0.30, "g_step": 1.0}


def jobs_for(st, over=None, fs=FS, dirs=DIRS):
    tp = F23.push_time(0.8)
    return [({"T_step": 0.8, **(over or {})}, st, {"t": tp, "F": [dx * f, dy * f], "dur": 0.1}) for dx, dy in dirs.values() for f in fs]


def limit(rs, fs=FS):
    lim = None
    for f, r in zip(fs, rs):
        if not r["ok"]:
            break
        lim = f
    return lim, "".join("○" if r["ok"] else ("△" if not r.get("fell") else "×") for r in rs)


def predict():
    import mujoco
    c = W.Cfg(extra={"arms": {"A": F23.A_STAR}}); P = W.plan(c)
    M = mujoco.mj_getTotalmass(P["rb"].m); om = np.sqrt(W.G / P["zc"])
    tp = F23.push_time(0.8); steps, _, _ = W.footsteps(c); Tds = c.ds * c.T_step
    t_land = steps[3][1] - Tds / 2
    dt = t_land - (tp + 0.05)                         # 押す力の中ごろから着地まで
    xc = (W.TOE - W.HEEL) / 2
    dd = {"前 +x": W.TOE - xc, "後ろ −x": W.HEEL + xc, "横（振り出す足の側）+y": W.FOOT_W / 2}
    out = {}
    for k, d in dd.items():
        out[k] = dict(ankle=float(M * om * d), step=float(M * om * (d + STEP["step_max"] * np.exp(-om * dt))))
    return out, float(M), float(om), float(dt)


def main():
    from multiprocessing import Pool
    pred, M, om, dt = predict()
    j0 = jobs_for({"K": 1.0}); j1 = jobs_for(STEP)
    with Pool(4) as pool:
        res = pool.map(F23.run, j0 + j1, chunksize=1)
    n = len(FS); out = {"pred": pred, "FS": FS, "res": []}
    print(f"=== 押されたときの回復（T 0.8 s、L 0.25 m、4 歩目の片脚支持の中ごろ、0.1 s）。力 {FS} N。○ 歩けた、△ 倒れないが位置が 10 cm 以上ずれた、× 転倒 ===")
    print(f"1 層目: M {M:.1f} kg、ω {om:.2f} 1/s、押してから着地まで {dt:.2f} s、踏み出しの最大 {STEP['step_max']} m")
    for b, name in enumerate(DIRS):
        r0 = res[b * n:(b + 1) * n]; r1 = res[len(j0) + b * n: len(j0) + (b + 1) * n]
        l0, s0 = limit(r0); l1, s1 = limit(r1)
        print(f"  {name}: 足首だけ（段階 23）{s0} → {l0} N、踏み出しあり {s1} → {l1} N"
              f"（予測: 足首 {pred[name]['ankle']:.1f} N·s、1 歩 {pred[name]['step']:.1f} N·s）", flush=True)
        for f, r in zip(FS, r1):
            print(f"      {f} N: 足のずらし {r.get('step', {}).get('offs')}", flush=True)
        out["res"].append(dict(dir=name, ankle=l0, step=l1, s0=s0, s1=s1, offs=[r.get("step", {}).get("offs") for r in r1]))
    json.dump(out, open(DATA / "field24_stepping.json", "w"), indent=1, ensure_ascii=False, default=str)
    print(f"[{time.time() - T0:.0f} s]")


def errors_main():
    """押す力 500 N（50 N·s）で、踏み出しの設定・質量を変える"""
    from multiprocessing import Pool
    items = [("基準", {}, {}), ("系統: 踏み出しの最大 0.2 m", {}, {"step_max": 0.2}), ("系統: 踏み出しの最大 0.4 m", {}, {"step_max": 0.4}),
             ("系統: 踏み出しのゲイン 0.7", {}, {"g_step": 0.7}), ("系統: 判定のしきい値 5 cm", {}, {"e_th": 0.05}),
             ("系統: 制御の遅れ 10 ms", {}, {"delay": 0.01})]
    rng = np.random.default_rng(4)
    for i in range(4):
        ms = {"trunk": 1 + rng.uniform(-.1, .1), "thigh": 1 + rng.uniform(-.1, .1), "shank": 1 + rng.uniform(-.1, .1), "foot": 1 + rng.uniform(-.15, .15)}
        items.append((f"入力: 質量 {i}", {"extra": {"mass": ms}}, {}))
    jobs = []
    for _, ov, se in items:
        jobs += jobs_for({**STEP, **se}, ov, fs=[500])
    with Pool(4) as pool:
        res = pool.map(F23.run, jobs, chunksize=1)
    nd = len(DIRS)
    print("=== 誤差の評価: 50 N·s で押したとき（前 / 後ろ / 横）。○ 歩けた、△ 位置のずれ、× 転倒 ===")
    out = []
    for i, (name, ov, se) in enumerate(items):
        rs = res[i * nd:(i + 1) * nd]
        sq = " / ".join("○" if r["ok"] else ("△" if not r.get("fell") else "×") for r in rs)
        print(f"| {name} | {sq} |", flush=True)
        out.append(dict(name=name, seq=sq))
    json.dump(out, open(DATA / "field24_errors.json", "w"), indent=1, ensure_ascii=False, default=str)
    print(f"[{time.time() - T0:.0f} s]")


def from_state_main():
    """F44: 計画を測った状態から作り直す版（from_state）。前・横に 30 / 50 / 70 N·s"""
    from multiprocessing import Pool
    tp = F23.push_time(0.8)
    jobs = [({"T_step": 0.8}, {**STEP, "from_state": True}, {"t": tp, "F": [dx * f, dy * f], "dur": 0.1}) for dx, dy in ((1, 0), (0, 1)) for f in (300, 500, 700)]
    with Pool(4) as pool:
        res = pool.map(F23.run, jobs, chunksize=1)
    print("=== 計画を測った状態から作り直す版: 前 30/50/70 N·s、横 30/50/70 N·s ===")
    for j, r in zip(jobs, res):
        print(f"  {j[2]['F']}: {'○' if r['ok'] else ('△' if not r.get('fell') else '×')}  足のずらし {r.get('step', {}).get('offs')}", flush=True)


def figure():
    res = json.load(open(DATA / "field24_stepping.json"))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    names = list(DIRS); en = ["forward", "backward", "lateral (swing side)"]; xx = np.arange(len(names))
    p23 = json.load(open(DATA / "field23_push.json"))
    lat23 = next(r["lim"] for r in p23["res"] if r["K"] == 1.0 and r["dir"].startswith("横（次"))   # 段階 23 の横（300 N より弱い側）
    a0 = [r["ankle"] / 10 if r["ankle"] else lat23 for r in res["res"]]; a1 = [r["step"] / 10 if r["step"] else 0 for r in res["res"]]
    ax[0].bar(xx - 0.18, a0, 0.36, label="ankle + pelvis (stage 23)", yerr=[[0] * 3, [10] * 3], capsize=2)
    ax[0].bar(xx + 0.18, a1, 0.36, label="+ stepping (stage 24)", yerr=[[0] * 3, [10] * 3], capsize=2)
    ax[0].plot(xx - 0.18, [res["pred"][n_]["ankle"] for n_ in names], "k_", ms=22, mew=2, label="layer 1: ankle M ω d")
    ax[0].plot(xx + 0.18, [res["pred"][n_]["step"] for n_ in names], "r_", ms=22, mew=2, label="layer 1: one step M ω (d + δ e^{−ωΔt})")
    lat = res["res"][2]
    if not lat["step"]:
        ax[0].text(2 + 0.18, 3, "not monotonic:\n" + lat["s1"].replace("○", "o").replace("×", "x").replace("△", "~") + "\n(30–70 N·s)", ha="center", fontsize=7)
    ax[0].set_xticks(xx); ax[0].set_xticklabels(en); ax[0].set_ylabel("max impulse survived [N·s]"); ax[0].legend(fontsize=7)
    ax[0].set_title("push recovery, T 0.8 s, L 0.25 m (0.1 s push, mid single support)", fontsize=9)
    tp = F23.push_time(0.8)
    for stp, col, lab in ((None, "tab:gray", "ankle + pelvis only"), (STEP, "tab:blue", "with stepping")):
        st = dict(stp) if stp else {"K": 1.0}
        c = W.Cfg(T_step=0.8, extra={"arms": {"A": F23.A_STAR}, "stab": st, "push": {"t": tp, "F": [0.0, 500.0], "dur": 0.1}})
        r = W.execute(c, record=True)["rec"]
        ax[1].plot(r["t"], r["com"][:, 1], color=col, label=f"{lab}: CoM y")
    ax[1].axvspan(tp, tp + 0.1, color="tab:red", alpha=0.2); ax[1].set_ylim(-0.5, 0.6)
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("lateral CoM [m]"); ax[1].legend(fontsize=7)
    ax[1].set_title("lateral push 50 N·s toward the swing side", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "field24_stepping.png", dpi=130)
    print("図 → figures/field/field24_stepping.png")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "main"
    {"main": main, "errors": errors_main, "from_state": from_state_main, "figure": figure}[arg]()
