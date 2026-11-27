"""Field28: 推定した重心で踏み出す（しきい値の決め直し）と、計画し直しを速くする（下半身の段 11）.

段階 27 では、推定した重心で踏み出すと押される前に踏み出した（F53）。押さずに歩いたときの DCM のずれの RMS は
推定 2.5 cm（真の値 1.3 cm）で、しきい値 3 cm は小さすぎた（data/field27_threshold_check.txt）。しきい値を 6 cm にしてやり直す。
計算は field27_push_estimated.py の threshold（約 20 分）。

計画し直しを速くする（stab の fast、既定は off）:
  ・過ぎた時刻は古い計画をそのまま使い、いまから先だけ逆運動学を解く（field21_walk.plan の reuse）
  ・逆運動学の外積と clip を成分で書く（値は同じ。基準の execute(Cfg()) は変わらない）
実行: python field28_threshold.py figure
"""
import json

import matplotlib.pyplot as plt
import numpy as np

import field21_walk as W
from common import DATA

OUT = W.OUT


def figure():
    a = json.load(open(DATA / "field27_push_estimated.json")); b = json.load(open(DATA / "field27_threshold.json"))
    rows = [("true state K1 + stepping (stage 24)", a["res"][0]), ("estimated K0.3, no stepping", a["res"][2]),
            ("estimated K0.3 + stepping, threshold 3 cm (stage 27)", a["res"][3]),
            ("estimated K0.3 + stepping, threshold 6 cm", b["res"][0]), ("estimated + stepping, 6 cm, no reverse", b["res"][1])]
    dn = ["forward", "backward", "lateral (swing side)"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1, 1.4]})
    xx = np.arange(3); wd = 0.16
    for k, (n, r) in enumerate(rows):
        ax[0].bar(xx + (k - 2) * wd, r["lims"], wd, label=n)
    ax[0].axhline(39.5, color="gray", ls=":", lw=1); ax[0].axhline(73.0, color="gray", ls="--", lw=1)
    ax[0].text(2.4, 40.5, "ankle limit (fwd)", fontsize=7, color="gray", ha="right"); ax[0].text(2.4, 74, "1-step capture (fwd)", fontsize=7, color="gray", ha="right")
    ax[0].set_xticks(xx); ax[0].set_xticklabels(dn); ax[0].set_ylabel("largest impulse recovered (from 30) [N s]"); ax[0].legend(fontsize=6.5, loc="upper right")
    ax[0].set_title("push recovery: stepping with estimated CoM (T 0.8 s)", fontsize=9)
    sym = {"○": 2, "△": 1, "×": 0}
    M = np.array([[sym[ch] for k in range(3) for ch in r["seq"][k]] for _, r in rows])
    ax[1].imshow(M, cmap="RdYlGn", vmin=0, vmax=2, aspect="auto")
    ax[1].set_yticks(range(len(rows))); ax[1].set_yticklabels([n for n, _ in rows], fontsize=7)
    F = a["FS"]; ax[1].set_xticks(range(15)); ax[1].set_xticklabels([f"{f // 10}" for _ in range(3) for f in F], fontsize=7)
    for k in (1, 2):
        ax[1].axvline(k * 5 - 0.5, color="k", lw=1.5)
    for k in range(3):
        ax[1].text(k * 5 + 2, -0.8, dn[k], ha="center", fontsize=8)
    ax[1].set_xlabel("impulse [N s] (green walked, yellow drifted >10 cm, red fell)")
    fig.tight_layout(); fig.savefig(OUT / "field28_threshold.png", dpi=130)
    print("図 → figures/field/field28_threshold.png")


if __name__ == "__main__":
    figure()
