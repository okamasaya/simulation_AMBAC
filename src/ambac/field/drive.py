"""走行の試験メニュー: 履帯の台車（base.type = "tracked"）を、腕の姿勢を保ったまま一定の速さで走らせる."""
from __future__ import annotations

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None

from .ground import pitch_yaw


def run_drive(xml, joints, pose, speed, T, wheel_radius, settle=0.5, record_every=20, pose_fun=None):
    """腕・腰を pose（{関節名: 角度}）に保ち、転輪の周速を speed [m/s] にして T 秒走る。
    pose_fun(t, state) を渡すと、走行中に姿勢を変えられる（state: {"x", "z", "pitch"}）。
    戻り値: 時系列（t, 車体の位置 x・z, pitch）と要約（進んだ距離、最大の pitch、最終の高さ、転倒したか）"""
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in joints]
    act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in joints]
    drv = [i for i in range(m.nu) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i).startswith("drive_")]
    q = np.array([pose.get(j, 0.0) for j in joints])
    d.qpos[adr] = q; d.ctrl[act] = q
    mujoco.mj_forward(m, d)
    rec = {k: [] for k in ("t", "x", "z", "pitch", "com_z")}
    n = int(round((settle + T) / m.opt.timestep))
    for k in range(n + 1):
        t = d.time - settle
        d.ctrl[drv] = (speed / wheel_radius) if t > 0 else 0.0
        if pose_fun is not None and t > 0:
            p, _ = pitch_yaw(d.qpos)
            newpose = pose_fun(t, {"x": d.qpos[0], "z": d.qpos[2], "pitch": p})
            d.ctrl[act] = [newpose.get(j, pose.get(j, 0.0)) for j in joints]
        if k % record_every == 0:
            p, _ = pitch_yaw(d.qpos)
            rec["t"].append(t); rec["x"].append(d.qpos[0]); rec["z"].append(d.qpos[2]); rec["pitch"].append(p)
            rec["com_z"].append(d.subtree_com[1][2])
        mujoco.mj_step(m, d)
        if abs(pitch_yaw(d.qpos)[0]) > np.radians(80):
            break
    rec = {k: np.array(v) for k, v in rec.items()}
    mv = rec["t"] >= 0
    s = dict(progress_m=float(rec["x"][-1] - rec["x"][mv][0]), max_pitch_deg=float(np.degrees(np.abs(rec["pitch"][mv]).max())),
             final_z_m=float(rec["z"][-1]), tipped=bool(np.degrees(np.abs(rec["pitch"][-1])) > 60))
    return rec, s
