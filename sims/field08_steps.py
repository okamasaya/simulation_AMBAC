"""Field08: 履帯の台車で段差・坂を走る — 腕の姿勢と、腕を能動的に動かす重心移動（地上版 AMBAC）の効果（段階 8）.

機体: tracked_arm（Sim06・10 の戦車型と同じ大きさ：接地長 0.8 m、前端の高さ 0.25 m、腕＋荷物 10 kg）。
履帯は、駆動する転輪を密に並べ、すべて同じ回転にそろえて近似する（ベルト）。駆動力の上限 2000 N（重量の約 1.1 倍）、周速 0.15 m/s、μ = 0.6、転輪 3 kg。
Part A: 段差（高さ 0.05〜0.25 m）を越えられるか。腕の姿勢: 下ろす / 前に伸ばす / 後ろ上方にたたむ / 能動（機首が上がったら腕を前へ）
Part B: 上り坂（10〜35°）を 1.5 m 上れるか。腕: 下ろす / 前 / 後ろ
第 1 段階の Sim10（準静的な見積もり）: 越えられる段差 0.25 m（形の限界）、腕を前に伸ばすと力の限界が上がる。
注意: 転輪の列で履帯を近似しているため、段の角が転輪の間に挟まると、駆動力によらず止まる場合がある（モデルの限界）。
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import copy
import json

import matplotlib.pyplot as plt
import numpy as np

from common import DATA, FIG
from ambac.field.assembly import build, load
from ambac.field.drive import run_drive

FIGF = FIG / "field"
R = np.radians
ROBOT = load("robots", "tracked_arm"); ROBOT["base"]["track"]["drive_force"] = 2000.0   # 重量（187 kg）の約 1.1 倍
WR = ROBOT["base"]["track"]["wheel_radius"]; SPEED = 0.15

POSES = {"down": {"arm_pitch": R(-90)}, "forward": {"arm_pitch": R(0)},
         "back": {"arm_pitch": R(135), "arm_elbow1": R(-100)}}


def active(t, st):
    """機首が上がる（pitch < 0）ほど腕を前へ。世界座標で真下（−90°）→ 水平前方（0°）へ、基準は台車の座標"""
    up = max(0.0, -st["pitch"])
    world = min(0.0, R(-90) + 4.0 * up)
    return {"arm_pitch": world - up}


def step_stage(h):
    return {"name": f"step{h}", "gravity": 9.80665, "friction": 0.6, "obstacles": [{"type": "step", "x": 0.8, "height": h}]}


def slope_stage(a):
    return {"name": f"slope{a}", "gravity": 9.80665, "friction": 0.6, "obstacles": [{"type": "slope", "x": 0.5, "angle": a, "length": 4.0}]}


def trial(stage, strat, T):
    xml, joints = build(ROBOT, stage)
    pose = POSES.get(strat, POSES["down"])
    return run_drive(xml, joints, pose, SPEED, T, WR, pose_fun=active if strat == "active" else None)


if __name__ == "__main__":
    res = {"steps": {}, "slopes": {}}
    H = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
    print("=== Part A: 段差（OK=段の上まで上がって走り抜けた / 引っかかり=止まった / 転倒=後ろに転倒）===")
    for strat in ["down", "forward", "back", "active"]:
        row = []
        for h in H:
            rec, s = trial(step_stage(h), strat, 25.0)
            ok = s["progress_m"] > 1.7 and not s["tipped"]
            tag = "OK" if ok else ("転倒" if s["tipped"] else "引っかかり")
            res["steps"][f"{strat}_{h}"] = dict(s, result=tag)
            row.append(f"{h:.3f}:{tag}")
        best = max([h for h in H if res["steps"][f"{strat}_{h}"]["result"] == "OK"], default=0.0)
        print(f"  {strat:8s} 越えられた最大 {best:.3f} m | " + " ".join(row), flush=True)

    print("\n=== Part B: 上り坂（1.5 m 上れたか）===")
    A = [10, 20, 25, 28, 31, 34]
    for strat in ["down", "forward", "back"]:
        row = []
        for a in A:
            rec, s = trial(slope_stage(a), strat, 16.0)
            along = s["progress_m"] / np.cos(R(a))
            ok = s["final_z_m"] > 1.5 * np.sin(R(a)) * 0.9 and not s["tipped"]
            tag = "OK" if ok else ("転倒" if s["tipped"] else "滑落/停止")
            res["slopes"][f"{strat}_{a}"] = dict(s, result=tag)
            row.append(f"{a}°:{tag}")
        print(f"  {strat:8s} " + " ".join(row), flush=True)
    print(f"  参考: 摩擦の限界 atan(μ) = {np.degrees(np.arctan(0.6)):.1f}°")
    json.dump(res, open(DATA / "field08_steps.json", "w"), indent=2, ensure_ascii=False)

    # 図: 0.15 m の段差で、腕を下ろしたまま（シーソーで停止）と能動の腕（乗り越え）
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for strat, col in [("down", "tab:gray"), ("forward", "tab:blue"), ("back", "tab:orange"), ("active", "tab:red")]:
        rec, s = trial(step_stage(0.15), strat, 25.0)
        ax[0].plot(rec["x"], -np.degrees(rec["pitch"]), color=col, label=f"arm {strat}")
        ax[1].plot(rec["t"], rec["x"], color=col, label=f"arm {strat}")
    ax[0].axvline(0.8 - 0.4 - 0.2 - WR, color="k", ls=":", lw=0.8)
    ax[0].set_xlabel("base x [m]"); ax[0].set_ylabel("nose-up pitch [deg]"); ax[0].set_title("0.15 m step: pitch vs position"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("base x [m]"); ax[1].set_title("progress"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGF / "field08_steps.png", dpi=130)
    print("図 → figures/field/field08_steps.png")
