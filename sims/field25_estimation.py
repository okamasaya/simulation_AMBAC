"""Field25: 重心の状態推定（下半身の段 8）.

段階 23 では、安定化（DCM のずれ → 骨盤をずらす）が、重心の速度に 5 cm/s の白色雑音を入れると全滅した（F42）。
実機で測れるのは重心そのものではなく、関節角（エンコーダ）、上体の姿勢と加速度（IMU）、足裏の力（足圧中心）である。
ここでは、それらに雑音をのせて重心の位置・速度を推定し、推定値で安定化を動かす（field21_walk.Estimator）。

  重心の位置 c_kin: 関節角の運動学で、立っている足から重心までのベクトルを求め、立っている足の場所に足す
      立っている足の場所（anchor）: "plan" 計画どおりに着地したとみなす（実機で使える） / "true" MuJoCo の正しい場所（脚のオドメトリが完全な場合の参照）
  速度: "fd" c_kin の 10 ms の差分 / "lipm" カルマンフィルタ（予測 LIPM c̈ = ω²(c − p)、p は測った足圧中心） / "acc" カルマンフィルタ（予測は IMU の加速度）
  雑音（基準）: 関節角 0.002 rad、上体の傾き 0.002 rad、足圧中心 1 cm、加速度 0.1 m/s²（すべて白色、1 kHz）

比べること: 歩ける範囲（段階 23 と同じ 24 通りの歩き方）、押されたときの回復、推定の誤差の RMS
実行: python field25_estimation.py [main|figure]
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
NOISE = {"sq": 0.002, "sth": 0.002, "sp": 0.01, "sa": 0.1}
NOISE3 = {k: 3 * v for k, v in NOISE.items()}
CASES = [
    ("真の値（段階 23）", None),
    ("差分（雑音あり、足の場所は計画）", {"type": "fd", "anchor": "plan", **NOISE}),
    ("差分（雑音あり、足の場所は正しい）", {"type": "fd", "anchor": "true", **NOISE}),
    ("LIPM＋足圧中心のカルマンフィルタ（計画）", {"type": "lipm", "sw": 10, "anchor": "plan", **NOISE}),
    ("LIPM＋足圧中心のカルマンフィルタ（正しい）", {"type": "lipm", "sw": 10, "anchor": "true", **NOISE}),
    ("加速度計のカルマンフィルタ（計画）", {"type": "acc", "sw": 10, "anchor": "plan", **NOISE}),
    ("加速度計のカルマンフィルタ（正しい）", {"type": "acc", "sw": 10, "anchor": "true", **NOISE}),
    ("LIPM のカルマンフィルタ、雑音 3 倍（計画）", {"type": "lipm", "sw": 10, "anchor": "plan", **NOISE3}),
    ("LIPM のカルマンフィルタ、雑音なし（計画）", {"type": "lipm", "sw": 10, "anchor": "plan"}),
    ("LIPM のカルマンフィルタ（計画）、K 0.5", {"type": "lipm", "sw": 10, "anchor": "plan", "_K": 0.5, **NOISE}),
    ("LIPM のカルマンフィルタ（計画）、K 0.3", {"type": "lipm", "sw": 10, "anchor": "plan", "_K": 0.3, **NOISE}),
]
LS, TS = F23.LS, F23.TS
FS = [100, 200, 300, 400, 500]


def run(args):
    over, est, push = args
    st = {"K": 1.0}
    if est is not None:
        est = dict(est); st["K"] = est.pop("_K", 1.0)
        st["est"] = est
    ex = {"arms": {"A": F23.A_STAR}, "stab": st}
    if push is not None:
        ex["push"] = push
    try:
        r = W.execute(W.Cfg(extra=ex, **over))
    except Exception as e:  # noqa: BLE001
        r = dict(ok=False, fell=True, err=str(e))
    return r


def seq(rs):
    return "".join("○" if r["ok"] else "×" for r in rs)


def main():
    from multiprocessing import Pool
    tp = F23.push_time(0.8)
    jobs = []
    for _, est in CASES:
        jobs += [({"L": L, "T_step": T}, est, None) for L in LS for T in TS]
    pcases = [0, 1, 3, 4, 5, 9, 10]
    for ci in pcases:
        jobs += [({"T_step": 0.8}, CASES[ci][1], {"t": tp, "F": [dx * f, dy * f], "dur": 0.1}) for dx, dy in ((1, 0), (0, 1)) for f in FS]
    with Pool(4) as pool:
        res = pool.map(run, jobs, chunksize=2)
    ng = len(LS) * len(TS); out = {"grid": [], "push": [], "LS": LS, "TS": TS, "FS": FS}
    print(f"=== 歩ける範囲（K 1、腕の振りあり）。行 = 歩幅 {LS}、並び = T {TS}（左が遅い）。推定の誤差の RMS は歩けた歩き方の平均（c_x, c_y [mm]、v_x, v_y [cm/s]） ===")
    for a, (name, est) in enumerate(CASES):
        rs = res[a * ng:(a + 1) * ng]
        er = np.array([r["est_rms"] for r in rs if r.get("ok") and "est_rms" in r])
        es = f"{er[:, 0].mean() * 1e3:.1f}, {er[:, 1].mean() * 1e3:.1f} mm、{er[:, 2].mean() * 100:.1f}, {er[:, 3].mean() * 100:.1f} cm/s" if len(er) else "−"
        fr = np.mean([r["ok"] for r in rs])
        print(f"| {name} | {' '.join(seq(rs[b * len(TS):(b + 1) * len(TS)]) for b in range(len(LS)))} | {fr:.2f} | {es} |", flush=True)
        out["grid"].append(dict(name=name, est=est, frac=float(fr), ok=[r["ok"] for r in rs], est_rms=er.mean(0).tolist() if len(er) else None))
    base = len(CASES) * ng; n = len(FS)
    print(f"=== 押されたときの回復（T 0.8 s、0.1 s、力 {FS} N）: 前 / 横（振り出す足の側） ===")
    for j, ci in enumerate(pcases):
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
    json.dump(out, open(DATA / "field25_estimation.json", "w"), indent=1, ensure_ascii=False, default=float)
    print(f"[{time.time() - T0:.0f} s]")


def figure():
    res = json.load(open(DATA / "field25_estimation.json"))
    en = ["true state (stage 23)", "finite diff (plan anchor)", "finite diff (true anchor)", "LIPM+CoP KF (plan)", "LIPM+CoP KF (true)",
          "accelerometer KF (plan)", "accelerometer KF (true)", "LIPM KF, noise x3 (plan)", "LIPM KF, no noise (plan)",
          "LIPM KF (plan), K 0.5", "LIPM KF (plan), K 0.3"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    fr = [g["frac"] for g in res["grid"]]; yy = np.arange(len(fr))
    ax[0].barh(yy, fr, color=["k"] + ["tab:red"] * 2 + ["tab:blue"] * 2 + ["tab:green"] * 2 + ["tab:purple"] * 2 + ["tab:orange"] * 2)
    ax[0].axvline(0.58, color="gray", ls=":", lw=1); ax[0].text(0.59, len(fr) - 0.6, "open loop (K 0)", fontsize=7, color="gray")
    ax[0].set_yticks(yy); ax[0].set_yticklabels(en, fontsize=8); ax[0].invert_yaxis(); ax[0].set_xlabel("fraction of 24 gaits walked (8 steps)")
    ax[0].set_title("stabilizer driven by estimated CoM (noise: joint 0.002 rad, tilt 0.002 rad, CoP 1 cm, acc 0.1 m/s²)", fontsize=8)
    vv = [(g["est_rms"][2] * 100 if g["est_rms"] else np.nan) for g in res["grid"]]
    ax[1].scatter(vv, fr, c=["k"] + ["tab:red"] * 2 + ["tab:blue"] * 2 + ["tab:green"] * 2 + ["tab:purple"] * 2 + ["tab:orange"] * 2)
    for v_, f_, n_ in zip(vv, fr, en):
        if np.isfinite(v_):
            ax[1].annotate(n_, (v_, f_), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax[1].set_xscale("log"); ax[1].set_xlabel("forward CoM velocity estimation error RMS [cm/s]"); ax[1].set_ylabel("fraction walked")
    ax[1].set_title("walking success vs velocity estimation error", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "field25_estimation.png", dpi=130)
    print("図 → figures/field/field25_estimation.png")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "main"
    {"main": main, "figure": figure}[arg]()
