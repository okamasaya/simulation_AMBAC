"""Field09: 二本脚で立つ機体を、シミュレーション場で戦車型と比べる（段階 9）.

機体: biped_waist（Sim09 Part B と同じ寸法：脚 24 kg・足裏 0.25 m、股関節 0.95 m の上体 56 kg、肩 1.45 m、総質量 97 kg）。
比較: tank_waist_3axis（戦車型：接地長 0.8 m、腰 0.4 m、総質量 97 kg）の段階 6 の結果。
荷物運び（段階 5・6 と同じ動作）を、傾き 1° 未満・ずれ 5 mm 未満で実行できる最短時間を、補正なし / 肘 / 腰 / 肘＋腰 で求める。
二本脚は補正なしでは実行できないので、Sim09 の解（最後に前へ約 28° 傾く）に近い初期値から、ゆっくりした動作で合格する補正を探し、
そこから段階 6 と同じ継続法で時間を縮める。
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import json
import time

import matplotlib.pyplot as plt
import numpy as np

import field06_waist as f6
from common import DATA, FIG
from ambac.field.assembly import build, load
from ambac.field.evaluate import _min_jerk, _targets, safe_time as eval_safe_time
from ambac.field.menu import run_menu

FIGF = FIG / "field"
BIPED = load("robots", "biped_waist")
f6.ROBOT = BIPED
GUESS = {"elbow": [0.6, 0.0], "waist": [0.0, 0.0, 0.0, np.radians(25)],
         "both": [0.6, 0.0, 0.0, 0.0, 0.0, np.radians(25)]}


def first_feasible(stage, strategy):
    """ゆっくりした動作で、合格する補正を探す（見つかった T と係数）"""
    xml, joints = build(BIPED, stage); a, b = _targets(joints, "arm")
    c = np.array(GUESS[strategy], float)
    for T in [6.0, 9.0, 12.0]:
        J, s, c2 = f6.best_at(xml, joints, a, b, T, strategy, c, step=0.15)
        print(f"    {strategy}: T={T:.0f} s で最良 {J:.2f}（1 未満で合格）", flush=True)
        if J < 1:
            return T, c2
        c = c2
    return None, c


if __name__ == "__main__":
    res = {}
    tank = json.load(open(DATA / "field06_waist.json"))
    for st in ["practice", "moon"]:
        stage = load("stages", st); res[st] = {}
        T0, s0 = eval_safe_time(BIPED, stage)
        res[st]["none"] = dict(T=None if np.isinf(T0) else round(float(T0), 3))
        print(f"=== {st}: 二本脚 ===\n  補正なし: {'実行できない（12 s でも転倒）' if np.isinf(T0) else f'{T0:.3f} s'}", flush=True)
        for strat in ["elbow", "waist", "both"]:
            t0 = time.time()
            Tf, c = first_feasible(stage, strat)
            if Tf is None:
                res[st][strat] = dict(T=None); print(f"  {strat:6s}: 実行できない", flush=True); continue
            # 継続法（段階 6 と同じ）: 初期値を差し替えるため、best_at の出発点を c にする
            xml, joints = build(BIPED, stage); a, b = _targets(joints, "arm")
            good = (Tf, None, c); T = Tf
            while True:
                T *= 0.85 if T > 2 else 0.94
                J, s, cn = f6.best_at(xml, joints, a, b, T, strat, good[2], step=0.05)
                if J < 1:
                    good = (T, s, cn)
                else:
                    break
            lo, hi = T, good[0]
            for _ in range(2):
                mid = (lo + hi) / 2
                J, s, cm = f6.best_at(xml, joints, a, b, mid, strat, good[2], step=0.03)
                if J < 1:
                    hi = mid; good = (mid, s, cm)
                else:
                    lo = mid
            T, s, c = good
            if s is None:
                _, s = f6.score(xml, joints, a, b, T, strat, c)
            tk = tank[st].get(strat, {}).get("T")
            res[st][strat] = dict(T=round(T, 3), coef=[round(x, 4) for x in c], at={k: round(v, 3) for k, v in s.items()}, tank_T=tk)
            print(f"  {strat:6s}: {T:.3f} s（戦車型 {tk} s）  傾き {s['max_tilt_deg']:.2f}°  ずれ {s['slide_mm']:.1f} mm  "
                  f"肩の最大トルク {s['peak_torque_Nm']:.0f} N·m  [{time.time() - t0:.0f} s]", flush=True)
    json.dump(res, open(DATA / "field09_biped.json", "w"), indent=2, ensure_ascii=False)

    # 図: 練習場、腰の補正の最短時間での、腰の角度と傾き（二本脚と戦車型）
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, robot, r, col in [("biped", BIPED, res["practice"].get("waist", {}), "tab:red"),
                                ("tank", load("robots", "tank_waist_3axis"), tank["practice"]["waist"], "tab:blue")]:
        if not r.get("T"):
            continue
        xml, joints = build(robot, load("stages", "practice")); a, b = _targets(joints, "arm")
        rec, s = run_menu(xml, joints, a, b, r["T"], _min_jerk, offsets=f6.offsets_of("waist", np.array(r["coef"])))
        t = rec["t"] / r["T"]
        ax[0].plot(t, np.degrees(rec["q"][:, joints.index("waist")]), color=col, label=f"{name}: T = {r['T']:.2f} s")
        ax[1].plot(t, np.degrees(rec["pitch"]), color=col, label=f"{name}: max tilt {s['max_tilt_deg']:.2f} deg")
    ax[0].set_title("waist angle (+: lean forward)"); ax[0].set_xlabel("t / T"); ax[0].set_ylabel("deg"); ax[0].legend()
    ax[1].set_title("base pitch"); ax[1].set_xlabel("t / T"); ax[1].axhspan(-1, 1, color="g", alpha=0.08); ax[1].legend()
    for x in ax:
        x.axvspan(0, 1, color="0.92", zorder=0)
    fig.tight_layout(); fig.savefig(FIGF / "field09_biped.png", dpi=130)
    print("図 → figures/field/field09_biped.png")
