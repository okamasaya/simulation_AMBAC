"""Field12: 腕の振りと履帯の速度差を組み合わせた、車体の向きの制御（段階 12）.

機体: twin_track_arm（段階 11 と同じ。左右 2 本の履帯、接地長 0.8 m、2 軸の肩＋肘＋荷物 10 kg）。腕は水平に伸ばし、肩の旋回で横に振る。
Part A: 腕を 0 → 180° 振る時間 T を変え、車体の向きの変化を測る。
        簡単な理論: 腕の反作用のトルクの最大値 I·α_max（最小ジャークでは α_max = 5.77 θ / T²）が、
        履帯を止めたまま車体を回し始めるトルク M_s = μ Σ p_i |r_i|（実測の荷重分布）を超えると回り始める → 境目 T* = √(5.77 θ I / M_s)
Part B: 腕を 1 s で振りながら、左右の履帯の速度差で向きを保つ（フィードバック: 向きの誤差と角速度から速度差を決める）。
Part C: 地上版 AMBAC：「速く振って、ゆっくり戻す」腕の往復を繰り返して、車体を回す。
        C1: 腕だけ（履帯は止める）  C2: 長い履帯（1.2 m）で駆動力 300 N（その場旋回できない）に、腕の往復を加える
4 並列で計算する。
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import json
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from common import DATA, FIG
from ambac.field.assembly import build, load
from ambac.field.drive import run_drive
from ambac.field.skid import breakaway_torque
import field11_turning as f11

FIGF = FIG / "field"
R = np.radians
STAGE = load("stages", "practice"); RS = 0.10; B = 0.45; MU = 0.6
ARM0 = {"arm_yaw": 0.0, "arm_pitch": 0.0, "arm_elbow1": 0.0}


def mj(u):
    u = np.clip(u, 0, 1); return 10 * u**3 - 15 * u**4 + 6 * u**5


def arm_inertia_z(r):
    """腕を水平に伸ばしたときの、肩の旋回軸（鉛直）まわりの慣性モーメント"""
    xml, j = build(r, STAGE); m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    sh = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "arm_shoulder")]
    I = 0.0
    for b in range(m.nbody):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if n.startswith("arm_"):
            rr = d.xipos[b] - sh; I += m.body_mass[b] * (rr[0]**2 + rr[1]**2)
            I += m.body_inertia[b].max() if "link" in n else 0.0   # 細い棒の長手に垂直な慣性（上限側の近似）
    return I


def swing_once(args):
    T, hold, L, force, gains = args
    r = f11.robot(L, force); xml, joints = build(r, STAGE)
    t0 = 1.0

    def pf(t, st):
        return {"arm_yaw": R(180) * mj((t - t0) / T), "arm_pitch": 0.0, "arm_elbow1": 0.0}
    sf = None
    if hold:
        kp, kd = gains

        def sf(t, st):
            w_cmd = kp * (0.0 - st["yaw"]) - kd * st["yaw_rate"]
            dv = float(np.clip(w_cmd * B / 2 / 0.8, -0.3, 0.3))      # 0.8: 旋回の効率（段階 11）で割って速度差に直す
            return (-dv, dv)
    rec, s = run_drive(xml, joints, ARM0, 0.0, t0 + T + 2.0, RS, pose_fun=pf, speed_fun=sf, record_every=50)
    y = np.degrees(rec["yaw"] - rec["yaw"][0])
    return args, dict(final_yaw=float(y[-1]), max_abs_yaw=float(np.abs(y).max()), max_roll=s["max_roll_deg"], diverged=s["diverged"]), \
        {"t": rec["t"].tolist(), "yaw": y.tolist()}


def ratchet(args):
    """速く振って（Tf 秒）、ゆっくり戻す（Ts 秒）往復を n 回。tracks: (v_L, v_R) または None"""
    Tf, Ts, n, L, force, tracks, amp = args
    r = f11.robot(L, force); xml, joints = build(r, STAGE)
    period = Tf + Ts + 0.4

    def pf(t, st):
        if Tf <= 0:
            return ARM0
        k = int(t // period); u = t - k * period
        a = mj(u / Tf) if u < Tf + 0.2 else 1 - mj((u - Tf - 0.2) / Ts)
        return {"arm_yaw": R(amp) * a, "arm_pitch": 0.0, "arm_elbow1": 0.0}
    sf = (lambda t, st: tracks) if tracks else None
    rec, s = run_drive(xml, joints, ARM0, 0.0, n * period, RS, pose_fun=pf, speed_fun=sf, record_every=50)
    y = np.degrees(rec["yaw"] - rec["yaw"][0])
    return args, dict(final_yaw=float(y[-1]), per_cycle=float(y[-1] / n), rate_deg_s=float(y[-1] / (n * period)),
                      diverged=s["diverged"]), {"t": rec["t"].tolist(), "yaw": y.tolist()}


if __name__ == "__main__":
    t0 = time.time()
    r08 = f11.robot(0.8); W = f11.weight(r08)
    xs, sh = f11.load_distribution(r08)
    Ms = breakaway_torque(xs, sh, B=B, mu=MU, W=W)
    I = arm_inertia_z(r08); theta = np.pi
    Tstar = np.sqrt(5.7735 * theta * I / Ms)
    TS = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
    jobsA = [(T, False, 0.8, None, None) for T in TS]
    jobsB = [(1.0, True, 0.8, None, g) for g in [(2.0, 0.2), (5.0, 0.5), (10.0, 1.0)]] + [(0.6, True, 0.8, None, (10.0, 1.0))]
    jobsC = [(0.6, 3.0, 4, 0.8, None, None, 120), (0.6, 0.6, 4, 0.8, None, None, 120),     # C1: 非対称 / 対称
             (0.0, 3.0, 4, 1.2, 300, (0.15, -0.15), 120), (0.6, 3.0, 4, 1.2, 300, (0.15, -0.15), 120)]   # C2: 腕なし / 腕あり
    with Pool(4) as pool:
        outA = pool.map(swing_once, jobsA); outB = pool.map(swing_once, jobsB); outC = pool.map(ratchet, jobsC)
    res = dict(theory=dict(weight_N=W, breakaway_torque_Nm=Ms, arm_inertia=I, T_star=Tstar), A=[], B=[], C=[])
    print(f"（{len(jobsA) + len(jobsB) + len(jobsC)} 試行、{time.time() - t0:.0f} s）")
    print(f"理論: 腕の慣性 I = {I:.1f} kg·m²、車体を回し始めるトルク M_s = {Ms:.0f} N·m → 境目 T* = {Tstar:.2f} s（これより速く振ると車体が回る）")
    print("=== Part A: 腕を 0 → 180° 振る時間と、車体の向きの変化（履帯は止めたまま）===")
    for (T, *_), s, _ in outA:
        tq = I * 5.7735 * theta / T**2
        res["A"].append(dict(T=T, reaction_peak_Nm=tq, **s))
        print(f"  T = {T:.1f} s: 反作用の最大 {tq:5.0f} N·m（M_s の {tq / Ms:.2f} 倍）→ 向きの変化 {s['final_yaw']:+6.2f}°、最大の横の傾き {s['max_roll']:.1f}°")
    print("=== Part B: 腕を振りながら、履帯で向きを保つ（フィードバック）===")
    for (T, _, _, _, g), s, _ in outB:
        res["B"].append(dict(T=T, kp=g[0], kd=g[1], **s))
        print(f"  T = {T:.1f} s、ゲイン Kp={g[0]}, Kd={g[1]}: 最大のずれ {s['max_abs_yaw']:.2f}°、最後のずれ {s['final_yaw']:+.2f}°")
    print("=== Part C: 地上版 AMBAC — 腕の往復で車体を回す ===")
    names = ["C1 腕だけ：速く振り（0.6 s）ゆっくり戻す（3 s）", "C1 腕だけ：速く振り速く戻す（対称、0.6 s / 0.6 s）",
             "C2 長い履帯・駆動力 300 N のその場旋回：腕なし", "C2 長い履帯・駆動力 300 N のその場旋回：腕の非対称な往復を加える"]
    for nm, (args, s, _) in zip(names, outC):
        res["C"].append(dict(name=nm, **s))
        print(f"  {nm}: 4 往復で {s['final_yaw']:+.1f}°（1 往復あたり {s['per_cycle']:+.2f}°、平均 {s['rate_deg_s']:+.2f}°/s）")
    json.dump(res, open(DATA / "field12_heading.json", "w"), indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    ax[0].plot([a["T"] for a in res["A"]], [abs(a["final_yaw"]) for a in res["A"]], "o-")
    ax[0].axvline(Tstar, color="r", ls=":", label=f"theory T* = {Tstar:.2f} s")
    ax[0].set_xlabel("arm swing time T [s] (0 -> 180 deg)"); ax[0].set_ylabel("|base heading change| [deg]"); ax[0].set_title("A: arm swing speed vs base rotation"); ax[0].legend()
    for (args, s, rec), lab in zip([outA[TS.index(1.0)]] + outB[:3], ["no control"] + [f"heading hold Kp={g[0]}" for (_, _, _, _, g) in jobsB[:3]]):
        ax[1].plot(rec["t"], rec["yaw"], label=lab)
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("heading [deg]"); ax[1].set_title("B: 1 s arm swing with heading hold by tracks"); ax[1].legend(fontsize=8)
    for (args, s, rec), lab in zip(outC, ["C1 arm only, fast/slow", "C1 arm only, symmetric", "C2 1.2 m, 300 N, no arm", "C2 1.2 m, 300 N, + arm ratchet"]):
        ax[2].plot(rec["t"], rec["yaw"], label=lab)
    ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("heading [deg]"); ax[2].set_title("C: ground AMBAC (asymmetric arm swing)"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGF / "field12_heading.png", dpi=130)
    print("図 → figures/field/field12_heading.png")
