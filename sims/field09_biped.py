"""Field09: 二本脚で立つ機体を、シミュレーション場で戦車型と比べる（段階 9）.

機体: biped_waist（Sim09 Part B と同じ寸法：脚 24 kg・足裏 0.25 m、股関節 0.95 m の上体 56 kg、肩 1.45 m、総質量 97 kg）。
比較: tank_waist_3axis（戦車型：接地長 0.8 m、腰 0.4 m、総質量 97 kg）の段階 6 の結果。
荷物運び（段階 5・6 と同じ動作）を、傾き 1° 未満・ずれ 5 mm 未満で実行できる最短時間を求める。

二本脚は、腕だけ（補正なし・肘の補正）では、12 s かけても転倒して実行できない。
そこで、腰の動きを 2 段階で作る。
  (1) 静的なつり合い: 腕の各姿勢で、全体の重心が足裏の中央に来る腰の角度（前もって計算する、フィードフォワード）
  (2) 動的な補正: (1) に正弦波の和（3 個）を重ね、MuJoCo を評価関数として最適化する（段階 6 と同じ継続法）
  (3) (2) に肘の補正（2 個）を加える
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import json
import time

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.optimize import brentq

import field06_waist as f6
from common import DATA, FIG
from ambac.field.assembly import build, load
from ambac.field.evaluate import _min_jerk, _targets, safe_time as eval_safe_time
from ambac.field.menu import run_menu

FIGF = FIG / "field"
R = np.radians
BIPED = load("robots", "biped_waist")
U = np.linspace(0, 1, 41)


def balance_profile(robot, stage):
    """腕の各姿勢（τ = 0〜1）で、重心が支持の中央（台車の原点の真上）に来る腰の角度"""
    xml, joints = build(robot, stage); a, b = _targets(joints, "arm")
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    adr = {x: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, x)] for x in joints}
    lo, hi = [R(v) for v in robot["waist"]["range"]]

    def comx(u, w):
        d.qpos[:] = m.qpos0
        for x in joints:
            d.qpos[adr[x]] = a.get(x, 0) + (b.get(x, 0) - a.get(x, 0)) * _min_jerk(u)
        d.qpos[adr["waist"]] = w; mujoco.mj_forward(m, d)
        return d.subtree_com[1][0] - d.xpos[1][0]
    return np.array([brentq(lambda w: comx(u, w), lo, hi) for u in U])


def make_offsets(prof):
    def offsets_of(strategy, c):
        c = np.asarray(c, float)
        ce = c[:2] if strategy == "static+dyn+elbow" else np.zeros(2)
        cw = c[-3:] if strategy in ("static+dyn", "static+dyn+elbow") else np.zeros(3)

        def f(u):
            w = np.interp(u, U, prof) + cw[0] * np.sin(np.pi * u) + cw[1] * np.sin(2 * np.pi * u) + cw[2] * np.sin(3 * np.pi * u)
            o = {"waist": float(np.clip(w, R(-45), R(45)))}
            if strategy == "static+dyn+elbow":
                o["arm_elbow1"] = ce[0] * np.sin(np.pi * u) + ce[1] * np.sin(2 * np.pi * u)
            return o
        return f
    return offsets_of


STRATS = {"static": 0, "static+dyn": 3, "static+dyn+elbow": 5}


def shortest(stage, strategy, xml, joints, a, b):
    """静的なつり合いだけ: 二分法。補正あり: 静的なつり合いの最短時間から継続法"""
    if STRATS[strategy] == 0:
        ok = lambda T: f6.score(xml, joints, a, b, T, strategy, [])
        lo, hi = 0.2, 12.0
        J, s = ok(hi)
        if J >= 1:
            return None, s, np.zeros(0)
        best = (hi, s)
        for _ in range(9):
            mid = (lo + hi) / 2; J, s = ok(mid)
            if J < 1:
                hi = mid; best = (mid, s)
            else:
                lo = mid
        return best[0], best[1], np.zeros(0)
    T = res_static_T * 1.02
    c = np.zeros(STRATS[strategy])
    J, s, c = f6.best_at(xml, joints, a, b, T, strategy, c, step=0.1)
    if J >= 1:
        return None, s, c
    good = (T, s, c)
    while True:
        T *= 0.94
        J, s, cn = f6.best_at(xml, joints, a, b, T, strategy, good[2], step=0.05)
        if J < 1:
            good = (T, s, cn)
        else:
            break
    lo, hi = T, good[0]
    for _ in range(2):
        mid = (lo + hi) / 2
        J, s, cm = f6.best_at(xml, joints, a, b, mid, strategy, good[2], step=0.03)
        if J < 1:
            hi = mid; good = (mid, s, cm)
        else:
            lo = mid
    return good


if __name__ == "__main__":
    res = {}
    tank = json.load(open(DATA / "field06_waist.json"))
    for st in ["practice", "moon"]:
        stage = load("stages", st); res[st] = {}
        xml, joints = build(BIPED, stage); a, b = _targets(joints, "arm")
        T0, _ = eval_safe_time(BIPED, stage)
        res[st]["none"] = dict(T=None if np.isinf(T0) else round(float(T0), 3))
        prof = balance_profile(BIPED, stage)
        res[st]["balance_profile_deg"] = [round(float(np.degrees(p)), 2) for p in prof]
        f6.offsets_of = make_offsets(prof); f6.NPAR.update(STRATS)
        print(f"=== {st}: 二本脚（戦車型は段階 6 の値）===", flush=True)
        print(f"  補正なし          : {'実行できない（12 s でも転倒）' if np.isinf(T0) else f'{T0:.3f} s'}（戦車型 {tank[st]['none']['T']} s）")
        print(f"  静的なつり合いの腰: 最初 {np.degrees(prof[0]):+.1f}° → 最後 {np.degrees(prof[-1]):+.1f}°（Sim09 の最適解 −40° → +28°）", flush=True)
        for strat in STRATS:
            t0 = time.time()
            T, s, c = shortest(stage, strat, xml, joints, a, b)
            if strat == "static":
                res_static_T = T
            ref = {"static": "waist", "static+dyn": "waist", "static+dyn+elbow": "both"}[strat]
            tk = tank[st][ref]["T"]
            if T is None:
                res[st][strat] = dict(T=None); print(f"  {strat:17s}: 実行できない", flush=True); continue
            res[st][strat] = dict(T=round(T, 3), coef=[round(x, 4) for x in c], at={k: round(v, 3) for k, v in s.items()}, tank_T=tk)
            print(f"  {strat:17s}: {T:.3f} s（戦車型の{'腰' if ref == 'waist' else '肘＋腰'} {tk} s）  傾き {s['max_tilt_deg']:.2f}°  "
                  f"ずれ {s['slide_mm']:.1f} mm  最大トルク {s['peak_torque_Nm']:.0f} N·m  [{time.time() - t0:.0f} s]", flush=True)
    json.dump(res, open(DATA / "field09_biped.json", "w"), indent=2, ensure_ascii=False)

    # 図: 練習場での腰の角度（静的なつり合い、動的な補正の最短時間）と傾き
    stage = load("stages", "practice"); xml, joints = build(BIPED, stage); a, b = _targets(joints, "arm")
    prof = np.radians(res["practice"]["balance_profile_deg"]); f6.offsets_of = make_offsets(prof)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(U, np.degrees(prof), "k--", label="static balance (feedforward)")
    for strat, col in [("static", "tab:gray"), ("static+dyn", "tab:green"), ("static+dyn+elbow", "tab:red")]:
        r = res["practice"].get(strat, {})
        if not r.get("T"):
            continue
        rec, s = run_menu(xml, joints, a, b, r["T"], _min_jerk, offsets=f6.offsets_of(strat, np.array(r["coef"])))
        t = rec["t"] / r["T"]
        ax[0].plot(t, np.degrees(rec["q"][:, joints.index("waist")]), color=col, label=f"{strat}: T = {r['T']:.2f} s")
        ax[1].plot(t, np.degrees(rec["pitch"]), color=col, label=f"{strat}: max {s['max_tilt_deg']:.2f} deg")
    ax[0].set_title("biped: waist angle (+: lean forward)"); ax[0].set_xlabel("t / T"); ax[0].set_ylabel("deg"); ax[0].legend(fontsize=8)
    ax[1].set_title("biped: foot pitch"); ax[1].set_xlabel("t / T"); ax[1].axhspan(-1, 1, color="g", alpha=0.08); ax[1].legend(fontsize=8)
    for x in ax:
        x.axvspan(0, 1, color="0.92", zorder=0); x.set_xlim(-0.5, 2.0)
    fig.tight_layout(); fig.savefig(FIGF / "field09_biped.png", dpi=130)
    print("図 → figures/field/field09_biped.png")
