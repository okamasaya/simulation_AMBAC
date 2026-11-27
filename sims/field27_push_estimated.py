"""Field27: 推定した重心で踏み出す（下半身の段 10）.

段階 24 の踏み出しは重心の真の値を使い、段階 25・26 の推定は踏み出しを使っていなかった。ここでは両方をつなぎ、
実機で測れる量（関節角・IMU・足裏の力、段階 26 の脚のオドメトリ＋LIPM のカルマンフィルタ）だけで、押されたら踏み出せるかを調べる。
また、段階 24 の F43（踏み出しすぎのあと逆向きに踏み出す）について、「最初の踏み出しと逆向きの成分を消す」制限（no_rev）を試す。

比べること（T 0.8 s、L 0.25 m、4 歩目の片脚支持の中ごろに 0.1 s 押す。力 300〜700 N、前・後ろ・横）:
  真の値 K 1 ＋踏み出し（段階 24） / 真の値＋踏み出し＋逆向きの禁止 /
  推定 K 0.3（踏み出しなし） / 推定 K 0.3 ＋踏み出し / 推定 K 0.3 ＋踏み出し＋逆向きの禁止
実行: python field27_push_estimated.py [main|figure]
"""
import json
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

import field21_walk as W
import field23_stabilizer as F23
import field24_stepping as F24
import field25_estimation as F25
from common import DATA

T0 = time.time()
OUT = W.OUT
EST = {"type": "lipm", "sw": 10, "anchor": "odo", **F25.NOISE}
CASES = [
    ("真の値 K 1、踏み出し（段階 24）", F24.STEP),
    ("真の値 K 1、踏み出し、逆向きの禁止", {**F24.STEP, "no_rev": True}),
    ("推定 K 0.3、踏み出しなし（段階 26）", {"K": 0.3, "est": EST}),
    ("推定 K 0.3、踏み出し", {**F24.STEP, "K": 0.3, "est": EST}),
    ("推定 K 0.3、踏み出し、逆向きの禁止", {**F24.STEP, "K": 0.3, "est": EST, "no_rev": True}),
]
FS, DIRS = F24.FS, F24.DIRS


def run(a):
    r = F23.run(a)
    return {k: r.get(k) for k in ("ok", "fell", "step", "pos_err", "err")}


def main():
    from multiprocessing import Pool
    jobs = []
    for _, st in CASES:
        jobs += F24.jobs_for(st)
    with Pool(4) as pool:
        res = pool.map(run, jobs, chunksize=1)
    n = len(FS); nd = len(DIRS); out = {"FS": FS, "dirs": list(DIRS), "res": []}
    print(f"=== 押されたときの回復（T 0.8 s、L 0.25 m、0.1 s）。力 {FS} N。○ 歩けた、△ 倒れないが位置が 10 cm 以上ずれた、× 転倒。耐えた力積 [N·s] ===")
    for a, (name, st) in enumerate(CASES):
        rr = res[a * n * nd:(a + 1) * n * nd]; row = []; lims = []
        for b, dn in enumerate(DIRS):
            l, sq = F24.limit(rr[b * n:(b + 1) * n]); lims.append(l / 10 if l else 0); row.append(f"{dn} {sq} {lims[-1]:.0f}")
        print(f"| {name} | " + " | ".join(row) + " |", flush=True)
        for b, dn in enumerate(DIRS):
            for f, r in zip(FS, rr[b * n:(b + 1) * n]):
                if r.get("step"):
                    print(f"      {dn} {f} N: {'○' if r['ok'] else '×'} 足のずらし {r['step']['offs']}", flush=True)
        out["res"].append(dict(name=name, lims=lims, seq=["".join("○" if r["ok"] else ("△" if not r.get("fell") else "×") for r in rr[b * n:(b + 1) * n]) for b in range(nd)],
                               offs=[[r.get("step", {}) and r["step"]["offs"] for r in rr[b * n:(b + 1) * n]] for b in range(nd)]))
    json.dump(out, open(DATA / "field27_push_estimated.json", "w"), indent=1, ensure_ascii=False, default=str)
    print(f"[{time.time() - T0:.0f} s]")


def threshold():
    """しきい値 6 cm で、推定した重心の行をやり直す（押さないときの DCM のずれの RMS は推定 2.5 cm、data/field27_threshold_check.txt）"""
    from multiprocessing import Pool
    cases = [("推定 K 0.3、踏み出し、しきい値 6 cm", {**F24.STEP, "K": 0.3, "est": EST, "e_th": 0.06}),
             ("推定 K 0.3、踏み出し、しきい値 6 cm、逆向きの禁止", {**F24.STEP, "K": 0.3, "est": EST, "e_th": 0.06, "no_rev": True})]
    jobs = []; tags = []
    for name, st in cases:
        for (over, st_, push), (dn, f) in zip(F24.jobs_for(st), [(d, f) for d in DIRS for f in FS]):
            jobs.append((over, st_, push)); tags.append((name, dn, f))
    out = {"FS": FS, "dirs": list(DIRS), "res": []}
    with Pool(4) as pool:
        res = []
        for tg, r in zip(tags, pool.imap(run, jobs)):
            res.append(r)
            print(f"  {tg[0]} | {tg[1]} {tg[2]} N: {'○' if r['ok'] else ('△' if not r.get('fell') else '×')} 足のずらし {(r.get('step') or {}).get('offs')}", flush=True)
    n = len(FS); nd = len(DIRS)
    print("=== しきい値 6 cm（T 0.8 s、L 0.25 m、0.1 s、30〜70 N·s） ===")
    for a, (name, _) in enumerate(cases):
        rr = res[a * n * nd:(a + 1) * n * nd]; row = []; lims = []
        for b, dn in enumerate(DIRS):
            l, sq = F24.limit(rr[b * n:(b + 1) * n]); lims.append(l / 10 if l else 0); row.append(f"{dn} {sq} {lims[-1]:.0f}")
        print(f"| {name} | " + " | ".join(row) + " |", flush=True)
        out["res"].append(dict(name=name, lims=lims, seq=["".join("○" if r["ok"] else ("△" if not r.get("fell") else "×") for r in rr[b * n:(b + 1) * n]) for b in range(nd)]))
    json.dump(out, open(DATA / "field27_threshold.json", "w"), indent=1, ensure_ascii=False, default=str)
    print(f"[{time.time() - T0:.0f} s]")


def figure():
    res = json.load(open(DATA / "field27_push_estimated.json"))
    en = ["true state K1 + stepping (stage 24)", "true K1 + stepping, no reverse", "estimated K0.3, no stepping (stage 26)",
          "estimated K0.3 + stepping", "estimated K0.3 + stepping, no reverse"]
    dn = ["forward", "backward", "lateral (swing side)"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1.3]})
    xx = np.arange(len(dn)); wd = 0.16
    for a, r in enumerate(res["res"]):
        ax[0].bar(xx + (a - 2) * wd, r["lims"], wd, label=en[a])
    ax[0].axhline(39.5, color="gray", ls=":", lw=1); ax[0].text(2.3, 40.5, "ankle limit Mωd (fwd)", fontsize=7, color="gray", ha="right")
    ax[0].axhline(73.0, color="gray", ls="--", lw=1); ax[0].text(2.3, 74, "1-step capture (fwd)", fontsize=7, color="gray", ha="right")
    ax[0].set_xticks(xx); ax[0].set_xticklabels(dn); ax[0].set_ylabel("largest impulse recovered (continuous from 30) [N s]"); ax[0].legend(fontsize=7)
    ax[0].set_title("push recovery with estimated CoM (T 0.8 s)", fontsize=9)
    sym = {"○": 2, "△": 1, "×": 0}
    M = np.array([[sym[ch] for b in range(3) for ch in r["seq"][b]] for r in res["res"]])
    ax[1].imshow(M, cmap="RdYlGn", vmin=0, vmax=2, aspect="auto")
    ax[1].set_yticks(range(len(en))); ax[1].set_yticklabels(en, fontsize=7)
    F = res["FS"]; ax[1].set_xticks(range(3 * len(F))); ax[1].set_xticklabels([f"{f // 10}" for _ in range(3) for f in F], fontsize=7)
    for b in range(1, 3):
        ax[1].axvline(b * len(F) - 0.5, color="k", lw=1.5)
    for b in range(3):
        ax[1].text(b * len(F) + len(F) / 2 - 0.5, -0.8, dn[b], ha="center", fontsize=8)
    ax[1].set_xlabel("impulse [N s] (green walked, yellow drifted >10 cm, red fell)")
    fig.tight_layout(); fig.savefig(OUT / "field27_push_estimated.png", dpi=130)
    print("図 → figures/field/field27_push_estimated.png")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "main"
    {"main": main, "threshold": threshold, "figure": figure}[arg]()
