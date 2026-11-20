"""Field11: 左右 2 本の履帯による旋回（スキッドステア）と、旋回中の腕の作業（段階 11）.

機体: twin_track_arm（板をつないだ履帯 2 本：各幅 0.15 m、中心の間隔 B = 0.45 m、接地長 L = 0.8 m、腕は 2 軸の肩＋肘＋荷物 10 kg）。
基準モデル: ambac.field.skid（接地面の圧力分布とクーロン摩擦から、定常旋回の角速度を求める準静的なモデル）。圧力分布は、一様な場合と、MuJoCo で止まった状態の接地の荷重を測った場合の 2 通り。
Part A: 左右の送りの速さ (v_L, v_R) を変えて、定常旋回の角速度を基準モデルと比べる（L = 0.8 m と 1.2 m）
Part B: その場旋回（v_L = −v_R）を、1 本あたりの駆動力の上限を変えて試す。基準モデルの必要な力 μ W L / (4 B) と比べる
Part C: 止まった状態と直進中に、荷物を持った腕を横に振る（肩の旋回 0 → 180°）。車体の向きの変化と傾きを測る（段階 2・Sim06 の続き）
4 並列で計算する。
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
from ambac.field.skid import pivot_resistance, steady_turn, steady_turn_dist
import mujoco

FIGF = FIG / "field"
R = np.radians
BASE = load("robots", "twin_track_arm")
STAGE = load("stages", "practice")
RS = BASE["base"]["track"]["sprocket_radius"]; B = BASE["base"]["track"]["gauge"]; MU = STAGE["friction"]
POSE = {"arm_pitch": R(-20), "arm_elbow1": R(-70)}   # 荷物が床のすぐ上に来る姿勢（以前の肩 −90° は床の下に入り込んでいた）


def robot(L=0.8, force=None):
    r = json.loads(json.dumps(BASE)); r["base"]["size"][0] = L
    if force:
        r["base"]["track"]["drive_force"] = force
    return r


def weight(r):
    return mass_properties(build(r, STAGE)[0])[0] * 9.80665


def load_distribution(r, bins=16):
    """止まった状態で、履板と床の接触の垂直力を、車体の前後方向の位置ごとに集計する（基準モデルに入れる荷重分布）"""
    xml, joints = build(r, STAGE); m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    for jn, v in POSE.items():
        adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]
        act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, jn); d.qpos[adr] = v; d.ctrl[act] = v
    for _ in range(int(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor"); Rh = d.xmat[1].reshape(3, 3); P = []
    for i in range(d.ncon):
        c = d.contact[i]
        if floor in (c.geom1, c.geom2):
            f = np.zeros(6); mujoco.mj_contactForce(m, d, i, f); P.append(((Rh.T @ (c.pos - d.xpos[1]))[0], f[0]))
    P = np.array(P); L = r["base"]["size"][0]
    h, e = np.histogram(P[:, 0], bins=bins, range=(-L / 2 - 0.05, L / 2 + 0.05), weights=P[:, 1])
    return (e[:-1] + e[1:]) / 2, h / h.sum()


def turn_trial(args):
    L, vl, vr, force = args
    r = robot(L, force); xml, joints = build(r, STAGE)
    rec, s = run_drive(xml, joints, POSE, 0.0, 8.0, RS, speed_fun=lambda t, st: (vl, vr), record_every=100)
    k = rec["t"] > 3.0
    w = float(np.polyfit(rec["t"][k], rec["yaw"][k], 1)[0]) if not s["diverged"] else float("nan")
    return args, w, s


def swing_trial(args):
    drive_v, Tsw = args
    r = robot(); xml, joints = build(r, STAGE)
    pose0 = {"arm_yaw": 0.0, "arm_pitch": R(0), "arm_elbow1": 0.0}

    def pf(t, st):
        u = np.clip((t - 2.0) / Tsw, 0, 1); s = 10 * u**3 - 15 * u**4 + 6 * u**5
        return {"arm_yaw": R(180) * s, "arm_pitch": R(0), "arm_elbow1": 0.0}
    rec, s = run_drive(xml, joints, pose0, drive_v, 2.0 + Tsw + 2.0, RS, pose_fun=pf, record_every=100)
    return args, s, {k: rec[k].tolist() for k in ("t", "yaw", "roll", "pitch")}


if __name__ == "__main__":
    t0 = time.time()
    WL = {L: weight(robot(L)) for L in (0.8, 1.2)}
    DIST = {L: load_distribution(robot(L)) for L in (0.8, 1.2)}
    for L in (0.8, 1.2):
        xs, sh = DIST[L]; on = xs[sh > 0.01]
        print(f"接地の荷重分布（L={L} m）: 荷重を受ける範囲 x = {on.min():+.2f}〜{on.max():+.2f} m、重心位置 {np.sum(xs * sh):+.3f} m")
    VS = [(0.15, -0.15), (0.15, 0.0), (0.20, 0.10), (0.15, 0.05)]
    jobsA = [(L, vl, vr, None) for L in (0.8, 1.2) for vl, vr in VS]
    FORCES = [200, 300, 450, 700, 1000]
    jobsB = [(L, 0.15, -0.15, F) for L in (0.8, 1.2) for F in FORCES]
    jobsC = [(v, T) for v in (0.0, 0.15) for T in (1.0, 3.0)]
    with Pool(4) as pool:
        outA = pool.map(turn_trial, jobsA); outB = pool.map(turn_trial, jobsB); outC = pool.map(swing_trial, jobsC)
    res = {"A": [], "B": [], "C": []}
    print(f"機体の重さ: L=0.8 m で {WL[0.8]:.0f} N、L=1.2 m で {WL[1.2]:.0f} N（{len(jobsA) + len(jobsB) + len(jobsC)} 試行、{time.time() - t0:.0f} s）")
    print("=== Part A: 定常旋回の角速度 [rad/s]（MuJoCo / 基準モデル）===")
    for (L, vl, vr, _), w, s in outA:
        th, eff = steady_turn(vl, vr, L=L, B=B, mu=MU, W=WL[L])
        td = steady_turn_dist(vl, vr, *DIST[L], B=B, mu=MU, W=WL[L])
        ideal = (vr - vl) / B
        res["A"].append(dict(L=L, vL=vl, vR=vr, w_mujoco=round(w, 4), w_theory_uniform=round(float(th[2]), 4),
                             w_theory=round(float(td[2]), 4), w_noslip=round(ideal, 4)))
        print(f"  L={L} m  v_L={vl:+.2f} v_R={vr:+.2f}: MuJoCo {w:+.3f}  基準モデル（実測の荷重分布）{td[2]:+.3f}  （一様な圧力）{th[2]:+.3f}"
              f"  滑りなし {ideal:+.3f}  → MuJoCo/基準 {w / td[2]:.2f}")
    print("=== Part B: その場旋回（v = ±0.15 m/s）と駆動力の上限（1 本あたり）===")
    for L in (0.8, 1.2):
        need = pivot_resistance(L=L, B=B, mu=MU, W=WL[L])
        row = []
        for (LL, _, _, F), w, s in outB:
            if LL == L:
                row.append(f"{F} N: {w:+.3f}")
                res["B"].append(dict(L=L, force=F, w=round(w, 4), need=round(need)))
        print(f"  L={L} m（基準モデルの必要な力 {need:.0f} N）: " + "  ".join(row))
    print("=== Part C: 荷物を持った腕を横に振る（肩の旋回 0 → 180°）===")
    for (v, T), s, rec in outC:
        res["C"].append(dict(drive_v=v, T_swing=T, **{k: round(s[k], 3) for k in ("net_yaw_deg", "max_roll_deg", "max_pitch_deg")}))
        print(f"  {'直進中' if v else '停止中'}（{v} m/s）、振る時間 {T} s: 車体の向きの変化 {s['net_yaw_deg']:+.2f}°  最大の横の傾き {s['max_roll_deg']:.2f}°"
              f"  最大の前後の傾き {s['max_pitch_deg']:.2f}°")
    json.dump(res, open(DATA / "field11_turning.json", "w"), indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for L, mk in ((0.8, "o"), (1.2, "s")):
        a = [r for r in res["A"] if r["L"] == L]
        ax[0].plot([r["w_theory"] for r in a], [r["w_mujoco"] for r in a], mk, label=f"L = {L} m")
    lim = max(abs(r["w_theory"]) for r in res["A"]) * 1.1
    ax[0].plot([-lim, 0], [-lim, 0], "k:", lw=0.8); ax[0].set_xlabel("reference model yaw rate [rad/s]"); ax[0].set_ylabel("MuJoCo yaw rate [rad/s]")
    ax[0].set_title("skid steering: MuJoCo vs reference model"); ax[0].legend()
    for L, col in ((0.8, "tab:blue"), (1.2, "tab:red")):
        b = [r for r in res["B"] if r["L"] == L]
        ax[1].plot([r["force"] for r in b], [-r["w"] for r in b], "o-", color=col, label=f"L = {L} m")
        ax[1].axvline(b[0]["need"], color=col, ls=":", lw=1)
    ax[1].set_xlabel("drive force limit per track [N]  (dotted: mu W L / 4B)"); ax[1].set_ylabel("pivot yaw rate [rad/s]")
    ax[1].set_title("pivot turn vs drive force"); ax[1].legend()
    for (v, T), s, rec in outC:
        ax[2].plot(rec["t"], np.degrees(np.array(rec["yaw"]) - rec["yaw"][0]), label=f"{'driving' if v else 'stopped'}, swing {T} s")
    ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("base yaw change [deg]"); ax[2].set_title("arm swing 0 -> 180 deg with 10 kg payload"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGF / "field11_turning.png", dpi=130)
    print("図 → figures/field/field11_turning.png")
