"""Field13: 腕の反作用・AMBAC・車体の摩擦の境目 — 仕組みを段階的に組み込む（段階 13）.

基準（段階 12）: twin_track_arm で、荷物 10 kg を持って水平に伸ばした腕を、肩の旋回で 0 → 180° を 1 s（最小ジャーク）で振ると、車体が 15.7° 回る。
これに、次の仕組みを 1 つずつ組み込み、同じ物差し（車体の向きの変化、車体にかかる反作用の最大値、横の傾き、腕の行き過ぎ、止まるまでの時間）で比べる。
  13-0 基準：硬い肩、最小ジャーク（振る時間 0.6 / 0.8 / 1.0 s）
  13-1 軌道の形：加速度の山を平らにした軌道（台形に近い加速度）。制御だけで、追加の部品なし
  13-2 しなり：肩の旋回にばね＋減衰（直列弾性）。腕の固有振動数 1.5 / 3 Hz、減衰比 0.3 / 0.7
  13-3 逆回転するおもり：車体に付けた円盤（10 kg、慣性 0.31 kg·m²）に、肩のモーターと逆向きのトルクを与える（打ち消す割合 0.5 / 1.0）
  13-4 組み合わせ：しなり＋円盤＋履帯で向きを保つ（段階 12 の Part B）
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
import field11_turning as f11

FIGF = FIG / "field"
R = np.radians
STAGE = load("stages", "practice"); RS = 0.10; B = 0.45
I_ARM = 12.1; M_S = 260.0          # 段階 12：腕の慣性、履帯を止めたまま車体を回し始めるトルク
T0 = 1.0                            # 腕を振り始める時刻
ARM0 = {"arm_yaw": 0.0, "arm_pitch": 0.0, "arm_elbow1": 0.0}


def minjerk(u):
    u = np.clip(u, 0, 1); return 10 * u**3 - 15 * u**4 + 6 * u**5


def _trap_table(r=0.1, n=2001):
    """加速度が「なめらかに立ち上がって平らになり、なめらかに下がる」軌道（加速と減速で対称）。0→1 に正規化"""
    tau = np.linspace(0, 1, n); a = np.zeros(n)
    for i, t in enumerate(tau):
        h = t if t < 0.5 else 1 - t; sgn = 1 if t < 0.5 else -1
        x = h / 0.5                                                   # 半分の中での位置 0〜1
        if x < r:
            g = 0.5 * (1 - np.cos(np.pi * x / r))
        elif x > 1 - r:
            g = 0.5 * (1 - np.cos(np.pi * (1 - x) / r))
        else:
            g = 1.0
        a[i] = sgn * g
    v = np.cumsum(a) / n; p = np.cumsum(v) / n
    return tau, p / p[-1], a.max() / (p[-1])                          # 位置、加速度の山の係数（θ/T² に対する倍率）


TAU, TRAP, TRAP_PEAK = _trap_table()


def trap(u):
    return np.interp(np.clip(u, 0, 1), TAU, TRAP)


def robot(sea=None, flywheel=False):
    r = f11.robot(0.8)
    if sea:
        f, z = sea
        k = I_ARM * (2 * np.pi * f)**2; c = 2 * z * np.sqrt(k * I_ARM)
        r["arms"][0]["sea"] = {"yaw": {"stiffness": round(k, 1), "damping": round(c, 1), "rotor_inertia": 0.5}}
    if flywheel:
        r["flywheel"] = {"inertia": 0.31, "mass": 10.0, "radius": 0.25, "pos": [-0.15, 0, 0.45], "torque_limit": 2000}
    return r


def trial(case):
    name, T, prof, sea, fw_gain, hold = case
    r = robot(sea, fw_gain > 0); xml, joints = build(r, STAGE)
    shape = minjerk if prof == "minjerk" else trap

    def pf(t, st):
        return {"arm_yaw": R(180) * shape((t - T0) / T), "arm_pitch": 0.0, "arm_elbow1": 0.0}
    ids = {}

    def ctrl(m, d, t):
        if not ids:
            ids["arm"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm_yaw")
            ids["fw"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "flywheel")
        if ids["fw"] >= 0:
            d.ctrl[ids["fw"]] = -fw_gain * d.actuator_force[ids["arm"]]       # 肩のモーターと逆向きのトルクで、車体への反作用を打ち消す

    def log(m, d):
        a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm_yaw"); f = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "flywheel")
        tau_arm = d.actuator_force[a]; tau_fw = d.actuator_force[f] if f >= 0 else 0.0
        yaw = d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "arm_yaw")]]
        sj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "arm_yaw_spring")
        if sj >= 0:
            yaw += d.qpos[m.jnt_qposadr[sj]]
        fj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "flywheel")
        wf = d.qvel[m.jnt_dofadr[fj]] if fj >= 0 else 0.0
        return [-(tau_arm + tau_fw), yaw, wf]
    sf = None
    if hold:
        def sf(t, st):
            w = 10.0 * (0.0 - st["yaw"]) - 1.0 * st["yaw_rate"]
            dv = float(np.clip(w * B / 2 / 0.8, -0.3, 0.3)); return (-dv, dv)
    rec, s = run_drive(xml, joints, ARM0, 0.0, T0 + T + 3.0, RS, pose_fun=pf, speed_fun=sf, ctrl_fun=ctrl, log_fun=log, record_every=20)
    lg = np.array(rec["log"]); t = rec["t"]; yaw_b = np.degrees(rec["yaw"] - rec["yaw"][0])
    arm = np.degrees(lg[:, 1]); err = arm - 180.0
    after = t > T0 + T
    settle = next((float(t[i] - (T0 + T)) for i in range(len(t)) if after[i] and np.all(np.abs(err[i:]) < 2.0)), np.nan)
    out = dict(name=name, T=T, base_yaw=float(yaw_b[-1]), base_yaw_max=float(np.abs(yaw_b).max()), max_roll=s["max_roll_deg"],
               base_torque_peak=float(np.abs(np.convolve(lg[:, 0], np.ones(5) / 5, mode="same")).max()),   # 0.05 s の移動平均（一瞬の衝撃の山を除く）
               base_torque_spike=float(np.abs(lg[:, 0]).max()),
               arm_lag_max=float(np.abs(np.degrees(R(180) * np.array([minjerk((tt - T0) / T) if prof == "minjerk" else trap((tt - T0) / T) for tt in t])) - arm).max()),
               arm_overshoot=float(max(0.0, err[after].max())) if after.any() else np.nan,
               settle_s=settle, flywheel_rpm_peak=float(np.abs(lg[:, 2]).max() * 60 / (2 * np.pi)), diverged=s["diverged"])
    return case, out, {"t": t.tolist(), "yaw": yaw_b.tolist(), "torque": lg[:, 0].tolist(), "arm": arm.tolist()}


if __name__ == "__main__":
    t0 = time.time()
    S_SOFT = (1.5, 0.7)
    cases = [("13-0 硬い肩・最小ジャーク", T, "minjerk", None, 0.0, False) for T in (1.0, 0.8, 0.6)]
    cases += [("13-1 軌道の形（加速度を平らに）", T, "trap", None, 0.0, False) for T in (1.0, 0.8)]
    cases += [(f"13-2 しなり {f} Hz・減衰比 {z}", 1.0, "minjerk", (f, z), 0.0, False) for f in (1.5, 3.0) for z in (0.3, 0.7)]
    cases += [(f"13-3 逆回転するおもり（打ち消す割合 {g}）", T, "minjerk", None, g, False) for g in (0.5, 1.0) for T in (1.0, 0.6)]
    cases += [("13-4 しなり＋おもり 0.5＋履帯で向きを保つ", T, "minjerk", S_SOFT, 0.5, True) for T in (1.0, 0.6)]
    cases += [("13-4 しなり＋軌道の形", 0.8, "trap", S_SOFT, 0.0, False)]
    with Pool(4) as pool:
        out = pool.map(trial, cases)
    print(f"（{len(cases)} 試行、{time.time() - t0:.0f} s）  参考: 車体を回し始めるトルク M_s ≈ {M_S:.0f} N·m、腕の慣性 {I_ARM} kg·m²")
    print(f"軌道の形の加速度の山: 最小ジャーク 5.77 θ/T²、加速度を平らにした軌道 {TRAP_PEAK:.2f} θ/T²")
    print("反作用の最大は 0.05 s の移動平均の最大値（括弧内は一瞬の衝撃を含む最大値）。腕の遅れは、計画した軌道からの最大の遅れ")
    print(f"{'仕組み':38s} {'時間':>5s} {'車体の向き':>9s} {'反作用の最大':>18s} {'横の傾き':>7s} {'腕の遅れ':>7s} {'腕の行き過ぎ':>9s} {'止まるまで':>8s} {'円盤の回転':>9s}")
    res = []
    for case, o, _ in out:
        res.append(o)
        st = "-" if np.isnan(o["settle_s"]) else f"{o['settle_s']:.2f} s"
        print(f"{o['name']:38s} {o['T']:4.1f}s {o['base_yaw']:+8.2f}° {o['base_torque_peak']:6.0f}（{o['base_torque_spike']:5.0f}）N·m {o['max_roll']:6.1f}° {o['arm_lag_max']:6.1f}° {o['arm_overshoot']:8.1f}° {st:>8s} "
              f"{o['flywheel_rpm_peak']:7.0f} rpm" + ("  発散" if o["diverged"] else ""))
    json.dump(res, open(DATA / "field13_reaction.json", "w"), indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    pick = {"13-0 硬い肩・最小ジャーク": "rigid, min-jerk", "13-1 軌道の形（加速度を平らに）": "rigid, flat-accel profile",
            "13-2 しなり 1.5 Hz・減衰比 0.7": "compliant 1.5 Hz, zeta 0.7", "13-3 逆回転するおもり（打ち消す割合 1.0）": "counter-rotating flywheel x1.0"}
    for case, o, rec in out:
        if o["name"] in pick and o["T"] == 1.0:
            lab = pick[o["name"]]
            ax[0].plot(rec["t"], rec["torque"], label=lab); ax[1].plot(rec["t"], rec["yaw"], label=lab); ax[2].plot(rec["t"], rec["arm"], label=lab)
    ax[0].axhspan(-M_S, M_S, color="g", alpha=0.08); ax[0].set_title("reaction torque on hull (green: below breakaway M_s)"); ax[0].set_ylabel("N m")
    ax[1].set_title("hull heading change"); ax[1].set_ylabel("deg"); ax[2].set_title("arm yaw angle (overshoot = side effect)"); ax[2].set_ylabel("deg")
    for a in ax:
        a.set_xlabel("t [s]"); a.legend(fontsize=7)
    fig.suptitle("Field13: arm swing 0 -> 180 deg in 1 s with 10 kg payload — mechanisms added step by step", y=1.0)
    fig.tight_layout(); fig.savefig(FIGF / "field13_reaction.png", dpi=130)
    print("図 → figures/field/field13_reaction.png")
