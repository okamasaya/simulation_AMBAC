"""Field26: 脚のオドメトリ（下半身の段 9）.

段階 25 では、重心の位置の推定が 3 cm ほどずれていた（F47）。原因は「立っている足は計画どおりの場所にある」とおいたこと。
実際の足は計画から数 cm ずれて着き、ねじれて滑る。ここでは、着いた足の場所を運動学で測って積み上げる（脚のオドメトリ）。

  接地の判定: 足裏の力センサの鉛直の力が体重の odo_th（0.1）を超えたら接地、半分を切ったら離地
  着いた足 sd の場所 = もう一方の足 o の推定の場所 ＋ 運動学（雑音あり）で測った (p_sd − p_o)
      着いた瞬間に置き、両足が接地しているあいだはその平均で更新する。片脚で立つあいだは動かさない
  重心の速度: 段階 25 の LIPM＋足圧中心のカルマンフィルタ（sw 10）。雑音は段階 25 と同じ

比べること: 足の場所が「計画」「オドメトリ」「正しい」のときの歩ける範囲（段階 23 と同じ 24 通り）、押されたときの回復、推定の誤差
系統の確認: 接地の判定を計画の時刻にする / 閾値 0.3 / 雑音 3 倍 / IMU の傾きの偏り 0.01 rad
実行: python field26_odometry.py [main|figure]
"""
import json
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

import field21_walk as W
import field23_stabilizer as F23
import field25_estimation as F25
from common import DATA

T0 = time.time()
OUT = W.OUT
NOISE, NOISE3 = F25.NOISE, F25.NOISE3
KF = {"type": "lipm", "sw": 10}
CASES = [
    ("真の値、K 1（段階 23）", None),
    ("真の値、K 0.3", {"_truth": True, "_K": 0.3}),
    ("足の場所は計画、K 1（段階 25）", {**KF, "anchor": "plan", **NOISE}),
    ("オドメトリ、K 1", {**KF, "anchor": "odo", **NOISE}),
    ("足の場所は正しい、K 1（段階 25）", {**KF, "anchor": "true", **NOISE}),
    ("足の場所は計画、K 0.3（段階 25）", {**KF, "anchor": "plan", "_K": 0.3, **NOISE}),
    ("オドメトリ、K 0.3", {**KF, "anchor": "odo", "_K": 0.3, **NOISE}),
    ("足の場所は正しい、K 0.3", {**KF, "anchor": "true", "_K": 0.3, **NOISE}),
    ("オドメトリ、K 0.3、接地は計画の時刻", {**KF, "anchor": "odo", "odo_plan": True, "_K": 0.3, **NOISE}),
    ("オドメトリ、K 0.3、接地の閾値 0.3", {**KF, "anchor": "odo", "odo_th": 0.3, "_K": 0.3, **NOISE}),
    ("オドメトリ、K 0.3、雑音 3 倍", {**KF, "anchor": "odo", "_K": 0.3, **NOISE3}),
    ("オドメトリ、K 0.3、傾きの偏り 0.01 rad", {**KF, "anchor": "odo", "_K": 0.3, "sthb": 0.01, **NOISE}),
]
LS, TS = F23.LS, F23.TS
FS = F25.FS
PCASES = [0, 1, 2, 3, 5, 6, 7]


def run(args):
    over, est, push = args
    st = {"K": 1.0}
    if est is not None:
        est = dict(est); st["K"] = est.pop("_K", 1.0)
        if not est.pop("_truth", False):
            st["est"] = est
    ex = {"arms": {"A": F23.A_STAR}, "stab": st}
    if push is not None:
        ex["push"] = push
    try:
        r = W.execute(W.Cfg(extra=ex, **over))
    except Exception as e:  # noqa: BLE001
        r = dict(ok=False, fell=True, err=str(e))
    if "anc_err" in r:     # 最後（8 歩のあと、両足で立っている）の足の場所の誤差 [m]
        a = r.pop("anc_err"); r["anc_end"] = float(np.mean([np.hypot(a[-1, 1], a[-1, 2]), np.hypot(a[-1, 3], a[-1, 4])]))
    return {k: r[k] for k in ("ok", "est_rms", "anc_end", "err") if k in r}


def seq(rs):
    return "".join("○" if r["ok"] else "×" for r in rs)


def main():
    from multiprocessing import Pool
    tp = F23.push_time(0.8)
    jobs = []
    for _, est in CASES:
        jobs += [({"L": L, "T_step": T}, est, None) for L in LS for T in TS]
    for ci in PCASES:
        jobs += [({"T_step": 0.8}, CASES[ci][1], {"t": tp, "F": [dx * f, dy * f], "dur": 0.1}) for dx, dy in ((1, 0), (0, 1)) for f in FS]
    with Pool(4) as pool:
        res = pool.map(run, jobs, chunksize=2)
    ng = len(LS) * len(TS); out = {"grid": [], "push": [], "LS": LS, "TS": TS, "FS": FS}
    print(f"=== 歩ける範囲（腕の振りあり）。行 = 歩幅 {LS}、並び = T {TS}（左が遅い）。推定の誤差の RMS は歩けた歩き方の平均"
          "（c_x, c_y [mm]、v_x, v_y [cm/s]）、足の場所の誤差は 8 歩のあと [mm] ===")
    for a, (name, est) in enumerate(CASES):
        rs = res[a * ng:(a + 1) * ng]
        er = np.array([r["est_rms"] for r in rs if r.get("ok") and "est_rms" in r])
        ae = [r["anc_end"] for r in rs if r.get("ok") and "anc_end" in r]
        es = f"{er[:, 0].mean() * 1e3:.1f}, {er[:, 1].mean() * 1e3:.1f} mm、{er[:, 2].mean() * 100:.1f}, {er[:, 3].mean() * 100:.1f} cm/s" if len(er) else "−"
        aes = f"{np.mean(ae) * 1e3:.0f} mm（最大 {np.max(ae) * 1e3:.0f}）" if ae else "−"
        fr = np.mean([r["ok"] for r in rs])
        print(f"| {name} | {' '.join(seq(rs[b * len(TS):(b + 1) * len(TS)]) for b in range(len(LS)))} | {fr:.2f} | {es} | {aes} |", flush=True)
        out["grid"].append(dict(name=name, est=est, frac=float(fr), ok=[r["ok"] for r in rs], est_rms=er.mean(0).tolist() if len(er) else None,
                                anc_end=float(np.mean(ae)) if ae else None))
    base = len(CASES) * ng; n = len(FS)
    print(f"=== 押されたときの回復（T 0.8 s、0.1 s、力 {FS} N）: 前 / 横（振り出す足の側） ===")
    for j, ci in enumerate(PCASES):
        rr = res[base + j * 2 * n: base + (j + 1) * 2 * n]
        lims = []
        for b in range(2):
            lim = 0
            for f, r in zip(FS, rr[b * n:(b + 1) * n]):
                if not r["ok"]:
                    break
                lim = f
            lims.append(lim)
        print(f"| {CASES[ci][0]} | 前 {lims[0] / 10:.0f} N·s {seq(rr[:n])} | 横 {lims[1] / 10:.0f} N·s {seq(rr[n:])} |", flush=True)
        out["push"].append(dict(name=CASES[ci][0], fwd=lims[0] / 10, lat=lims[1] / 10))
    json.dump(out, open(DATA / "field26_odometry.json", "w"), indent=1, ensure_ascii=False, default=float)
    print(f"[{time.time() - T0:.0f} s]")


def figure():
    res = json.load(open(DATA / "field26_odometry.json"))
    en = ["true state, K 1 (stage 23)", "true state, K 0.3", "plan anchor, K 1 (stage 25)", "odometry, K 1", "true anchor, K 1",
          "plan anchor, K 0.3 (stage 25)", "odometry, K 0.3", "true anchor, K 0.3", "odometry, K 0.3, plan timing",
          "odometry, K 0.3, threshold 0.3", "odometry, K 0.3, noise x3", "odometry, K 0.3, tilt bias 0.01 rad"]
    col = ["k", "gray", "tab:red", "tab:blue", "tab:green", "tab:red", "tab:blue", "tab:green", "tab:cyan", "tab:cyan", "tab:purple", "tab:purple"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    fr = [g["frac"] for g in res["grid"]]; yy = np.arange(len(fr))
    ax[0].barh(yy, fr, color=col)
    ax[0].axvline(0.58, color="gray", ls=":", lw=1); ax[0].text(0.59, len(fr) - 0.6, "open loop", fontsize=7, color="gray")
    ax[0].set_yticks(yy); ax[0].set_yticklabels(en, fontsize=8); ax[0].invert_yaxis(); ax[0].set_xlabel("fraction of 24 gaits walked (8 steps)")
    ax[0].set_title("stabilizer driven by estimated CoM: where is the stance foot?", fontsize=9)
    cx = [(g["est_rms"][0] * 1e3 if g["est_rms"] else np.nan) for g in res["grid"]]
    ax[1].scatter(cx, fr, c=col)
    for v_, f_, n_ in zip(cx, fr, en):
        if np.isfinite(v_):
            ax[1].annotate(n_, (v_, f_), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax[1].set_xscale("log"); ax[1].set_xlabel("forward CoM position estimation error RMS [mm]"); ax[1].set_ylabel("fraction walked")
    ax[1].set_title("walking success vs position estimation error", fontsize=9)
    P = res["push"]; xx = np.arange(len(P))
    ax[2].bar(xx - 0.2, [p["fwd"] for p in P], 0.4, label="forward"); ax[2].bar(xx + 0.2, [p["lat"] for p in P], 0.4, label="lateral")
    ax[2].set_xticks(xx); ax[2].set_xticklabels([en[i] for i in PCASES], rotation=35, ha="right", fontsize=7)
    ax[2].set_ylabel("recovered impulse [N s]"); ax[2].legend(fontsize=8); ax[2].set_title("push recovery (T 0.8 s)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "field26_odometry.png", dpi=130)
    print("図 → figures/field/field26_odometry.png")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "main"
    {"main": main, "figure": figure}[arg]()
