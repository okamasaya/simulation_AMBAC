"""試験メニュー: 組み立てた機体を、決まった関節の動きで動かして計測する."""
from __future__ import annotations

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None

from .ground import pitch_yaw, roll


def run_menu(xml, joints, q_start, q_end, T, profile, settle=0.5, hold=1.0, record_every=4):
    """関節を q_start → q_end へ、profile(τ)（0→1）に沿って T 秒で動かす。
    q_start, q_end は {関節名: 角度[rad]}（省略した関節は 0）。
    戻り値: 時系列（t, q, 台車の pitch・roll・yaw・位置・高さ, 接触点の数）と、要約の指標"""
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    free = m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in joints]
    a = np.array([q_start.get(j, 0.0) for j in joints]); b = np.array([q_end.get(j, 0.0) for j in joints])
    d.qpos[adr] = a; d.ctrl[:] = a
    mujoco.mj_forward(m, d)
    rec = {k: [] for k in ("t", "q", "pitch", "roll", "yaw", "xyz", "ncon")}
    n = int(round((settle + T + hold) / m.opt.timestep))
    for k in range(n + 1):
        t = d.time - settle
        d.ctrl[:] = a + (b - a) * profile(np.clip(t / T, 0, 1))
        if k % record_every == 0:
            rec["t"].append(t); rec["q"].append(d.qpos[adr].copy()); rec["ncon"].append(d.ncon)
            if free:
                p, y = pitch_yaw(d.qpos)
                rec["pitch"].append(p); rec["roll"].append(roll(d.qpos)); rec["yaw"].append(y); rec["xyz"].append(d.qpos[0:3].copy())
        mujoco.mj_step(m, d)
    rec = {k: np.array(v) for k, v in rec.items()}
    mv = rec["t"] >= 0
    s = {"track_err_deg": float(np.degrees(np.abs(rec["q"][-1] - b).max()))}
    if free and len(rec["pitch"]):
        i0 = np.argmin(abs(rec["t"]))
        s.update(max_tilt_deg=float(np.degrees(max(np.abs(rec["pitch"][mv]).max(), np.abs(rec["roll"][mv]).max()))),
                 net_yaw_deg=float(np.degrees(rec["yaw"][-1] - rec["yaw"][i0])),
                 slide_mm=float(np.linalg.norm(rec["xyz"][-1, :2] - rec["xyz"][i0, :2]) * 1000),
                 airborne_s=float(np.sum(rec["ncon"][mv] == 0) * (rec["t"][1] - rec["t"][0])),
                 final_tilt_deg=float(np.degrees(max(abs(rec["pitch"][-1]), abs(rec["roll"][-1])))))
    return rec, s


def snapshot(xml, joints, q, size=(360, 480), distance=3.0, azimuth=120, elevation=-20, lookat=(0.4, 0, 0.6)):
    """関節角 q（辞書）での機体の画像（画面のない環境では MUJOCO_GL=osmesa）"""
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    for j in joints:
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] = q.get(j, 0.0)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, *size)
    cam = mujoco.MjvCamera(); cam.lookat[:] = lookat; cam.distance = distance; cam.azimuth = azimuth; cam.elevation = elevation
    r.update_scene(d, camera=cam); img = r.render(); r.close()
    return img
