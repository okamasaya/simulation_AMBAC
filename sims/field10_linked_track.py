"""Field10: 板をつないだ履帯（正確なモデル）で、段差・坂の走行をやり直す（段階 10）.

段階 8 は、履帯を「転輪の列」で近似したため、段の角が転輪の間に挟まって止まる場合があった。
ここでは、履板（ピッチ 4 cm、厚さ 2 cm、0.3 kg）をヒンジでつないだ輪を、起動輪（後ろ、駆動、地面から 1 cm 浮かせる）・転輪 5 個（サスペンション付き）・誘導輪（前、高さ 0.25 m、進入角 40°、張り装置付き）に巻き付ける。
地面・段に触れるのは履板だけで、駆動は起動輪と履板の摩擦で伝える。
機体: linked_track_arm（車体 80 kg、接地長 0.8 m、腕＋荷物 10 kg）。駆動力の上限 2000 N、周速 0.15 m/s、μ = 0.6。
Part A: 段差 0.05〜0.30 m × 腕の姿勢（下ろす / 前 / 後ろ上方 / 能動：機首が上がるほど腕を前へ）
Part B: 上り坂 15〜34° × 腕の姿勢
Part C: 機体の形を変える（重心を下げる 0.30 → 0.15 m、履帯を長くする 0.8 → 1.2 m、両方）× 段差 0.20〜0.30 m × 腕（下ろす / 能動）
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
EN = {"OK": "climbed", "転倒": "tipped over", "引っかかり": "stuck", "滑落/停止": "slid back", "発散": "diverged"}
R = np.radians
ROBOT = load("robots", "linked_track_arm")
RS = ROBOT["base"]["track"]["sprocket_radius"]; SPEED = 0.15; T_RUN = 30.0
# 腕を下ろす: 荷物が床のすぐ上（前方 0.47 m、高さ 0.08 m）に来る姿勢。以前は肩 −90° で、腕と荷物が床の下に入り込んでいた
POSES = {"down": {"arm_pitch": R(-20), "arm_elbow1": R(-70)}, "forward": {"arm_pitch": R(0), "arm_elbow1": 0.0},
         "back": {"arm_pitch": R(135), "arm_elbow1": R(-100)}}


def active(t, st):
    """機首が上がる（pitch < 0）ほど、腕を「下ろす」姿勢から「水平に前へ伸ばす」姿勢へ移す（機首の上がり 22.5° で伸ばし切る）。
    角度は世界座標で決め、台車の座標に直して指令する"""
    up = max(0.0, -st["pitch"])
    s = min(1.0, up / R(22.5))
    world = R(-20) * (1 - s)
    return {"arm_pitch": world - up, "arm_elbow1": R(-70) * (1 - s)}


def stage_of(kind, v):
    if kind == "step":
        return {"name": f"step{v}", "gravity": 9.80665, "friction": 0.6, "obstacles": [{"type": "step", "x": 0.9, "height": v}]}
    return {"name": f"slope{v}", "gravity": 9.80665, "friction": 0.6, "obstacles": [{"type": "slope", "x": 0.7, "angle": v, "length": 4.0}]}


def variant(name):
    r = json.loads(json.dumps(ROBOT))
    if "lowcom" in name:
        r["base"]["com"] = [0, 0, 0.15]
    if "long" in name:
        r["base"]["size"][0] = 1.2
    return r


def trial(args):
    kind, v, strat = args[:3]
    shape = args[3] if len(args) > 3 else "base"
    xml, joints = build(variant(shape), stage_of(kind, v))
    rec, s = run_drive(xml, joints, POSES.get(strat, POSES["down"]), SPEED, T_RUN, RS,
                       pose_fun=active if strat == "active" else None, record_every=200)
    if kind == "step":
        ok = s["progress_m"] > 1.7 and not s["tipped"]
    else:
        ok = s["final_z_m"] > 1.5 * np.sin(R(v)) * 0.9 and not s["tipped"]
    tag = "発散" if s["diverged"] else ("OK" if ok else ("転倒" if s["tipped"] else ("引っかかり" if kind == "step" else "滑落/停止")))
    return tuple(args), dict(s, result=tag), {k: rec[k].tolist() for k in ("t", "x", "z", "pitch")}


if __name__ == "__main__":
    xml, _ = build(ROBOT, stage_of("step", 0.1)); M, com = mass_properties(xml)
    print(f"機体: 総質量 {M:.0f} kg、重心の高さ {com[2]:.2f} m（腕を下ろす前の初期姿勢）")
    H = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]; A = [15, 20, 25, 28, 31, 34]
    STR = ["down", "forward", "back", "active"]
    SHAPES = ["lowcom", "long", "long+lowcom"]
    jobs = [("step", h, s) for s in STR for h in H] + [("slope", a, s) for s in ["down", "forward", "back"] for a in A]
    jobs += [("step", h, s, sh) for sh in SHAPES for s in ["down", "active"] for h in [0.20, 0.25, 0.30]]
    t0 = time.time()
    with Pool(4) as pool:
        out = pool.map(trial, jobs)
    key = lambda k: "_".join(str(x) for x in k)
    res = {key(k): dict(s) for k, s, _ in out}
    traj = {key(k): r for k, _, r in out}
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
    print("=== Part C: 機体の形を変えた場合（段差 0.20 / 0.25 / 0.30 m、元の形は Part A）===")
    for sh, lab in [("lowcom", "重心を下げる（0.30→0.15 m）"), ("long", "履帯を長く（0.8→1.2 m）"), ("long+lowcom", "長く＋重心を下げる")]:
        for st in ["down", "active"]:
            print(f"  {lab:18s} {st:7s} " + " ".join(f"{h:.2f}:{res[f'step_{h}_{st}_{sh}']['result']}" for h in [0.20, 0.25, 0.30]))
    json.dump(dict(results=res), open(DATA / "field10_linked_track.json", "w"), indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for st, col in [("down", "tab:gray"), ("forward", "tab:blue"), ("back", "tab:orange"), ("active", "tab:red")]:
        r = traj[f"step_0.15_{st}"]
        ax[0].plot(r["x"], -np.degrees(np.array(r["pitch"])), color=col, label=f"arm {st}: {EN[res[f'step_0.15_{st}']['result']]}")
        ax[1].plot(r["t"], r["x"], color=col, label=f"arm {st}")
    ax[0].set_xlabel("base x [m]"); ax[0].set_ylabel("nose-up pitch [deg]"); ax[0].set_title("linked-plate track, 0.15 m step"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("base x [m]"); ax[1].set_title("progress"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGF / "field10_linked_track.png", dpi=130)
    print("図 → figures/field/field10_linked_track.png")
