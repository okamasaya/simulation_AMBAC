"""試験メニュー: 組み立てた機体を、決まった関節の動きで動かして計測する."""
from __future__ import annotations

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None

from .ground import pitch_yaw, roll


def run_menu(xml, joints, q_start, q_end, T, profile, settle=0.5, hold=1.0, record_every=4, offsets=None):
    """関節を q_start → q_end へ、profile(τ)（0→1）に沿って T 秒で動かす。
    q_start, q_end は {関節名: 角度[rad]}（省略した関節は 0）。
    offsets(τ) を渡すと、τ=t/T（0〜1 に制限）での {関節名: 追加の角度} を目標に足す（腰の戦略など）。
    モデルに speed_limit（rad/s、関節ごと、0 は制限なし）があれば、目標角の変化の速さを制限する（サーボの最高速度）。
    戻り値: 時系列（t, q, 台車の pitch・roll・yaw・位置・高さ, 接触点の数）と、要約の指標"""
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    free = m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in joints]
    jidx = {j: i for i, j in enumerate(joints)}
    act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in joints]   # 関節の位置サーボ（走行用の駆動は除く）

    def extra(tau):
        e = np.zeros(len(joints))
        if offsets is not None:
            for j, v in offsets(tau).items():
                e[jidx[j]] = v
        return e
    a = np.array([q_start.get(j, 0.0) for j in joints]); b = np.array([q_end.get(j, 0.0) for j in joints])
    d.qpos[adr] = a + extra(0.0); d.ctrl[act] = a + extra(0.0)
    mujoco.mj_forward(m, d)
    vmax = None
    for i in range(m.nnumeric):
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_NUMERIC, i) == "speed_limit":
            v = m.numeric_data[m.numeric_adr[i]:m.numeric_adr[i] + m.numeric_size[i]].copy()
            vmax = np.where(v > 0, v, np.inf)
    cmd = d.ctrl[act].copy()
    rec = {k: [] for k in ("t", "q", "pitch", "roll", "yaw", "xyz", "ncon", "tau")}
    n = int(round((settle + T + hold) / m.opt.timestep))
    for k in range(n + 1):
        t = d.time - settle
        tau = np.clip(t / T, 0, 1)
        target = a + (b - a) * profile(tau) + extra(tau)
        if vmax is not None:
            step = vmax * m.opt.timestep
            cmd = cmd + np.clip(target - cmd, -step, step)
        else:
            cmd = target
        d.ctrl[act] = cmd
        if k % record_every == 0:
            rec["t"].append(t); rec["q"].append(d.qpos[adr].copy()); rec["ncon"].append(d.ncon); rec["tau"].append(d.actuator_force[act].copy())
            if free:
                p, y = pitch_yaw(d.qpos)
                rec["pitch"].append(p); rec["roll"].append(roll(d.qpos)); rec["yaw"].append(y); rec["xyz"].append(d.qpos[0:3].copy())
        mujoco.mj_step(m, d)
    rec = {k: np.array(v) for k, v in rec.items()}
    mv = rec["t"] >= 0
    err = np.degrees(np.abs(rec["q"] - (b + extra(1.0))).max(1))
    out = np.nonzero(err >= 1.0)[0]
    done = float(rec["t"][out[-1] + 1]) if len(out) and out[-1] + 1 < len(err) else (float(rec["t"][0]) if not len(out) else np.inf)
    s = {"track_err_deg": float(err[-1]), "done_time_s": max(done, 0.0),
         "peak_torque_Nm": float(np.abs(rec["tau"][mv]).max())}
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
