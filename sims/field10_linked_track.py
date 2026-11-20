"""Field10: 板をつないだ履帯（正確なモデル）で、段差・坂の走行をやり直す（段階 10）.

段階 8 は、履帯を「転輪の列」で近似したため、段の角が転輪の間に挟まって止まる場合があった。
ここでは、履板（ピッチ 4 cm、厚さ 2 cm、0.3 kg）57 枚をヒンジでつないだ輪を、起動輪（後ろ、駆動）・転輪 5 個・誘導輪（前、高さ 0.25 m、進入角 40°）に巻き付ける。
地面・段に触れるのは履板だけで、駆動は起動輪と履板の摩擦で伝える。
機体: linked_track_arm（車体 80 kg、接地長 0.8 m、腕＋荷物 10 kg）。駆動力の上限 2000 N、周速 0.15 m/s、μ = 0.6。
Part A: 段差 0.05〜0.30 m × 腕の姿勢（下ろす / 前 / 後ろ上方 / 能動：機首が上がるほど腕を前へ）
Part B: 上り坂 15〜34° × 腕の姿勢
4 並列で計算する（1 回の試行は実時間の約 2.6 倍）。
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import json
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

from common import DATA, FIG
from ambac.field.assembly import build, load, mass_properties
from ambac.field.drive import run_drive

FIGF = FIG / "field"
R = np.radians
ROBOT = load("robots", "linked_track_arm")
RS = ROBOT["base"]["track"]["sprocket_radius"]; SPEED = 0.15; T_RUN = 30.0
POSES = {"down": {"arm_pitch": R(-90)}, "forward": {"arm_pitch": R(0)},
         "back": {"arm_pitch": R(135), "arm_elbow1": R(-100)}}


def active(t, st):
    up = max(0.0, -st["pitch"])
    world = min(0.0, R(-90) + 4.0 * up)
    return {"arm_pitch": world - up}


def stage_of(kind, v):
    if kind == "step":
        return {"name": f"step{v}", "gravity": 9.80665, "friction": 0.6, "obstacles": [{"type": "step", "x": 0.9, "height": v}]}
    return {"name": f"slope{v}", "gravity": 9.80665, "friction": 0.6, "obstacles": [{"type": "slope", "x": 0.7, "angle": v, "length": 4.0}]}


def trial(args):
    kind, v, strat = args
    xml, joints = build(ROBOT, stage_of(kind, v))
    rec, s = run_drive(xml, joints, POSES.get(strat, POSES["down"]), SPEED, T_RUN, RS,
                       pose_fun=active if strat == "active" else None, record_every=200)
    if kind == "step":
        ok = s["progress_m"] > 1.7 and not s["tipped"]
    else:
        ok = s["final_z_m"] > 1.5 * np.sin(R(v)) * 0.9 and not s["tipped"]
    tag = "OK" if ok else ("転倒" if s["tipped"] else ("引っかかり" if kind == "step" else "滑落/停止"))
    return (kind, v, strat), dict(s, result=tag), {k: rec[k].tolist() for k in ("t", "x", "z", "pitch")}


if __name__ == "__main__":
    xml, _ = build(ROBOT, stage_of("step", 0.1)); M, com = mass_properties(xml)
    print(f"機体: 総質量 {M:.0f} kg、重心の高さ {com[2]:.2f} m（腕を下ろす前の初期姿勢）")
    H = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]; A = [15, 20, 25, 28, 31, 34]
    STR = ["down", "forward", "back", "active"]
    jobs = [("step", h, s) for s in STR for h in H] + [("slope", a, s) for s in ["down", "forward", "back"] for a in A]
    t0 = time.time()
    with Pool(4) as pool:
        out = pool.map(trial, jobs)
    res = {f"{k[0]}_{k[1]}_{k[2]}": dict(s) for k, s, _ in out}
    traj = {f"{k[0]}_{k[1]}_{k[2]}": r for k, _, r in out}
    print(f"（{len(jobs)} 試行、{time.time() - t0:.0f} s）")
    s8 = json.load(open(DATA / "field08_steps.json"))
    print("=== Part A: 段差（OK=段の上を走り抜けた）。括弧内は段階 8（転輪の列の近似）===")
    for st in STR:
        row = []
        for h in H:
            r = res[f"step_{h}_{st}"]; old = s8["steps"].get(f"{st}_{h}", {}).get("result", "-")
            row.append(f"{h:.2f}:{r['result']}({old})")
        best = max([h for h in H if res[f'step_{h}_{st}']['result'] == 'OK'], default=0.0)
        print(f"  {st:8s} 最大 {best:.2f} m | " + " ".join(row))
    print("=== Part B: 上り坂（1.5 m 上れたか）===")
    for st in ["down", "forward", "back"]:
        print(f"  {st:8s} " + " ".join(f"{a}°:{res[f'slope_{a}_{st}']['result']}" for a in A))
    print(f"  参考: 摩擦の限界 atan(μ) = {np.degrees(np.arctan(0.6)):.1f}°")
    json.dump(dict(results=res), open(DATA / "field10_linked_track.json", "w"), indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for st, col in [("down", "tab:gray"), ("forward", "tab:blue"), ("back", "tab:orange"), ("active", "tab:red")]:
        r = traj[f"step_0.15_{st}"]
        ax[0].plot(r["x"], -np.degrees(np.array(r["pitch"])), color=col, label=f"arm {st}: {res[f'step_0.15_{st}']['result']}")
        ax[1].plot(r["t"], r["x"], color=col, label=f"arm {st}")
    ax[0].set_xlabel("base x [m]"); ax[0].set_ylabel("nose-up pitch [deg]"); ax[0].set_title("linked-plate track, 0.15 m step"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("base x [m]"); ax[1].set_title("progress"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGF / "field10_linked_track.png", dpi=130)
    print("図 → figures/field/field10_linked_track.png")
