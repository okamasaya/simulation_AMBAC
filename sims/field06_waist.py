"""Field06: 腰の戦略（地上版 AMBAC）を、組み立てた機体に適用する（シミュレーション場の段階 6）.

機体: tank_waist_3axis（戦車型台車＋前後に傾く腰＋3 軸の肩の腕）。荷物運び（前方の地面 → 頭上 → 後部）は段階 5 と同じ。
腕の動き（最小ジャーク）に、次の「補正」を重ねて、傾き 1° 未満・ずれ 5 mm 未満で実行できる最短時間を比べる。
  (0) 補正なし（段階 5 の性能表の値）
  (1) 肘の補正：動作の途中で荷物を体に引き寄せる（第 1 段階の Sim06 の最適軌道に相当）
  (2) 腰の補正：上体を前後に傾けて、腕の反作用を打ち消す（Sim09 の腰の戦略に相当）
  (3) 肘＋腰
補正は τ=t/T の正弦波の和で表し（両端で 0、腰は最後に一定の傾きを残してよい）、
係数を Nelder–Mead 法で最適化する（MuJoCo を評価関数として直接使う：物理の式を使わない探索）。
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import json
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from common import DATA, FIG
from ambac.field.assembly import build, load
from ambac.field.evaluate import SLIDE_LIMIT_MM, TILT_LIMIT_DEG, _min_jerk, _targets
from ambac.field.menu import run_menu

FIGF = FIG / "field"; FIGF.mkdir(exist_ok=True)
ROBOT = load("robots", "tank_waist_3axis")
WAIST_MAX = np.radians(35.0)


def offsets_of(strategy, c):
    """係数 c から補正の関数を作る。肘 2 個（sin πτ, sin 2πτ）、腰 4 個（sin πτ, sin 2πτ, sin 3πτ, 最後に残す傾き）"""
    ce = c[:2] if strategy in ("elbow", "both") else np.zeros(2)
    cw = c[-4:] if strategy in ("waist", "both") else np.zeros(4)

    def f(u):
        o = {}
        if strategy in ("elbow", "both"):
            o["arm_elbow1"] = ce[0] * np.sin(np.pi * u) + ce[1] * np.sin(2 * np.pi * u)
        if strategy in ("waist", "both"):
            w = cw[0] * np.sin(np.pi * u) + cw[1] * np.sin(2 * np.pi * u) + cw[2] * np.sin(3 * np.pi * u) + cw[3] * _min_jerk(u)
            o["waist"] = float(np.clip(w, -WAIST_MAX, WAIST_MAX))
        return o
    return f


NPAR = {"none": 0, "elbow": 2, "waist": 4, "both": 6}


def score(xml, joints, a, b, T, strategy, c):
    _, s = run_menu(xml, joints, a, b, T, _min_jerk, offsets=offsets_of(strategy, c))
    return max(s["max_tilt_deg"] / TILT_LIMIT_DEG, s["slide_mm"] / SLIDE_LIMIT_MM), s


def best_at(xml, joints, a, b, T, strategy, c0, step=0.1):
    """T を固定して補正の係数を最適化する（c0 から出発、初期の単体の大きさ step [rad]）"""
    if NPAR[strategy] == 0:
        J, s = score(xml, joints, a, b, T, strategy, [])
        return J, s, np.zeros(0)
    c0 = np.asarray(c0, float); n = len(c0)
    simplex = np.vstack([c0] + [c0 + step * np.eye(n)[k] for k in range(n)])
    r = minimize(lambda c: np.log(score(xml, joints, a, b, T, strategy, c)[0] + 1e-6), c0, method="Nelder-Mead",
                 options=dict(maxfev=25 * n + 20, xatol=1e-3, fatol=1e-3, initial_simplex=simplex))
    J, s = score(xml, joints, a, b, T, strategy, r.x)
    return J, s, r.x


def safe_time(stage, strategy, t_start, shrink=0.94, refine=2):
    """継続法: 安全な時間 t_start から、T を少しずつ短くしながら、前の係数を初期値にして最適化し直す。
    基準を満たせなくなったら、最後に満たした T との間を二分法で詰める"""
    xml, joints = build(ROBOT, stage); a, b = _targets(joints, "arm")
    J, s, c = best_at(xml, joints, a, b, t_start, strategy, np.zeros(NPAR[strategy]))
    assert J < 1, (strategy, J)
    good = (t_start, s, c); T = t_start
    while True:
        T = T * shrink
        J, s, cn = best_at(xml, joints, a, b, T, strategy, good[2], step=0.05)
        if J < 1:
            good = (T, s, cn)
        else:
            break
        if NPAR[strategy] == 0:
            continue
    lo, hi = T, good[0]
    for _ in range(refine):
        mid = (lo + hi) / 2
        J, s, cm = best_at(xml, joints, a, b, mid, strategy, good[2], step=0.03)
        if J < 1:
            hi = mid; good = (mid, s, cm)
        else:
            lo = mid
    return good


if __name__ == "__main__":
    perf5 = {(r["robot"], r["stage"]): float(r["safe_time_s"]) for r in
             __import__("csv").DictReader(open(DATA / "field_performance.csv", encoding="utf-8"))}
    res = {}
    print("=== 荷物運びの安全な最短時間（tank_waist_3axis、傾き 1° 未満・ずれ 5 mm 未満）===")
    for st in ["practice", "moon"]:
        stage = load("stages", st); base = perf5[("tank_waist_3axis", st)]
        res[st] = {}
        for strat in ["none", "elbow", "waist", "both"]:
            t0 = time.time()
            T, s, c = safe_time(stage, strat, base * 1.02)
            res[st][strat] = dict(T=round(T, 3), reduction_pct=round(100 * (1 - T / base), 1), coef=[round(x, 4) for x in c],
                                  at={k: round(v, 3) for k, v in s.items()})
            print(f"  {st:9s} {strat:6s}: {T:.3f} s（補正なしの性能表 {base:.3f} s から {100 * (1 - T / base):+.0f}%）"
                  f"  傾き {s['max_tilt_deg']:.2f}° ずれ {s['slide_mm']:.1f} mm  [{time.time() - t0:.0f} s]", flush=True)
    json.dump(res, open(DATA / "field06_waist.json", "w"), indent=2, ensure_ascii=False)

    # ---- 図: 練習場で、補正なしと腰の補正を、腰の補正の最短時間で比べる --------------------------
    stage = load("stages", "practice"); xml, joints = build(ROBOT, stage); a, b = _targets(joints, "arm")
    Tw = res["practice"]["both"]["T"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for strat, col in [("none", "0.5"), ("elbow", "tab:blue"), ("waist", "tab:green"), ("both", "tab:red")]:
        c = np.array(res["practice"][strat]["coef"])
        rec, s = run_menu(xml, joints, a, b, Tw, _min_jerk, offsets=offsets_of(strat, c))
        t = rec["t"]; lab = {"none": "no compensation", "elbow": "elbow (pull payload in)", "waist": "waist strategy", "both": "elbow + waist"}[strat]
        ax[0].plot(t, np.degrees(rec["pitch"]), color=col, label=f"{lab}: max {s['max_tilt_deg']:.1f} deg")
        o = offsets_of(strat, c)
        u = np.clip(t / Tw, 0, 1)
        if strat in ("waist", "both"):
            ax[1].plot(t, np.degrees(rec["q"][:, joints.index("waist")]), color=col, label=lab)
        if strat in ("elbow", "both"):
            ax[2].plot(t, np.degrees(rec["q"][:, joints.index("arm_elbow1")]), color=col, label=lab)
    ax[2].plot(t, np.degrees(rec["q"][:, joints.index("arm_pitch")]), "k:", label="shoulder pitch (common)")
    ax[0].axhspan(-1, 1, color="g", alpha=0.08); ax[0].set_ylim(-20, 20)
    ax[0].set_title(f"base pitch, practice stage, T = {Tw:.2f} s"); ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("deg"); ax[0].legend(fontsize=8)
    ax[1].set_title("waist angle (+: lean forward)"); ax[1].set_xlabel("t [s]"); ax[1].legend(fontsize=8)
    ax[2].set_title("arm joints"); ax[2].set_xlabel("t [s]"); ax[2].legend(fontsize=8)
    for x in ax:
        x.axvspan(0, Tw, color="0.9", zorder=0)
    fig.tight_layout(); fig.savefig(FIGF / "field06_waist.png", dpi=130)
    print("図 → figures/field/field06_waist.png")
