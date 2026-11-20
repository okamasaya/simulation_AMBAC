"""Field14: 滑らかな止め方 — トルク制御の肩と、流体ブレーキ（MR・ダイラタント・擬塑性・チキソトロピー）（段階 14）.

段階 13 で、車体が回るのは腕を「止めるとき」で、位置サーボの遅れが急ブレーキを生むことが分かった。
ここでは、肩の旋回を「トルクを直接制御するモーター」に変え、トルクの上限を、車体が滑り出す境目 M_s の 0.4 倍（104 N·m）にする
（最初に 0.8 倍で試すと、トルクの山が境目より低くても、上限のトルクが長く続いて車体が 22° 回った。段階 12 の実測でも 0.6 倍付近から滑り始めた）。
モーターは「この位置から上限のトルクで止めれば目標で止まれる」減速の曲線に沿って、腕を 0 → 180° に動かす。
これに、流体ブレーキ（回転ダンパー）を並べて付ける。ブレーキは車体に取り付けるので、その反作用も車体に返る。
流体の性質（ω: 腕の角速度、トルクは動きと逆向き）:
  粘性（ニュートン流体）   τ = c ω
  MR 流体（ビンガム流体）   τ = τ_y sign(ω) + c₀ ω。τ_y を電流（磁場）で変えられる（応答の遅れ 10 ms）
  ダイラタント（速いほど硬い） τ = K ω|ω|
  擬塑性（速いほど柔らかい） τ = K |ω|^0.5 sign(ω)
  チキソトロピー           τ = (η∞ + η₀ λ) ω、構造 λ（0〜1）: dλ/dt = (1 − λ)/t_r − k λ |ω|（動くと壊れ、休むと戻る）
強さは、腕の速さ 5 rad/s で 50 N·m 程度になるようにそろえた。
MR 流体は制御する：モーターは前に押すだけ（逆向きのトルクを出さない）で、止めるのは MR のブレーキに任せる。止まった後は τ_y = 60 N·m で保持する。
到着の後に、腕を外から 80 N·m で 0.3 s 押し、ずれを測る（止まっている間の保持の強さ）。
さらに、モーターの電源を切った状態（トルク 0）で同じように押し、流体だけでどれだけ腕を保持できるかを比べる。
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import json
import re
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from common import DATA, FIG
from ambac.field.assembly import build, load
from ambac.field.drive import run_drive
import field11_turning as f11
import field13_reaction as f13

FIGF = FIG / "field"
R = np.radians
STAGE = load("stages", "practice"); RS = 0.10
I_ARM, M_S = 12.1, 260.0
TAU_CAP = 0.4 * M_S                 # 段階 12 の実測では 0.6 M_s 付近から滑り始めたため、その下に設定
T0, TH = 1.0, R(180)
T_PUSH, PUSH = 3.5, 80.0           # 外から押す時刻（T0 からの経過）と強さ
ARM0 = {"arm_yaw": 0.0, "arm_pitch": 0.0, "arm_elbow1": 0.0}
DEVICES = {
    "none": "ブレーキなし",
    "viscous": "粘性（ニュートン流体）",
    "mr": "MR 流体（制御、モーターは逆回転しない）",
    "dilatant": "ダイラタント（速いほど硬い）",
    "thinning": "擬塑性（速いほど柔らかい）",
    "thixo": "チキソトロピー（動くと柔らかく、休むと固まる）",
}


def device_torque(kind, w, st, dt, cmd_brake=0.0):
    """流体ブレーキが腕に与えるトルク（動きと逆向き）。st は内部状態（λ、MR の τ_y）"""
    if kind == "viscous":
        return -10.0 * w
    if kind == "dilatant":
        return -2.0 * w * abs(w)
    if kind == "thinning":
        return -22.0 * np.sqrt(abs(w)) * np.sign(w)
    if kind == "thixo":
        lam = st.get("lam", 1.0)
        lam += dt * ((1 - lam) / 2.0 - 3.0 * lam * abs(w)); lam = min(max(lam, 0.0), 1.0); st["lam"] = lam
        return -(2.0 + 40.0 * lam) * w
    if kind == "mr":
        ty = st.get("ty", 0.0); ty += dt / 0.01 * (cmd_brake - ty); st["ty"] = ty   # 磁場の応答の遅れ 10 ms
        # ビンガム流体: 動いていれば τ_y、止まっていれば（ほぼ）静止摩擦として最大 τ_y まで保持
        return -(ty * np.tanh(w / 0.05) + 1.0 * w)
    return 0.0


def trial(args):
    kind, power_off = (args, False) if isinstance(args, str) else args
    r = f11.robot(0.8); xml, joints = build(r, STAGE)
    xml = re.sub(r'<position name="arm_yaw" joint="arm_yaw"[^/]*/>', '<motor name="arm_yaw" joint="arm_yaw" gear="1"/>', xml)
    st = {}; ids = {}; acc = dict(heat=0.0, motor_neg=0.0, motor_work=0.0)
    m_ = mujoco.MjModel.from_xml_string(xml); dt = m_.opt.timestep

    def ctrl(m, d, t):
        if not ids:
            ids["a"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm_yaw")
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "arm_yaw"); ids["q"] = m.jnt_qposadr[j]; ids["v"] = m.jnt_dofadr[j]
        th = d.qpos[ids["q"]]; w = d.qvel[ids["v"]]
        if t < T0:
            tau_des = I_ARM * 20.0 * (0.0 - th) * 5 - I_ARM * 10 * w           # 開始前は 0° に保持
        else:
            e = TH - th; a_cap = TAU_CAP / I_ARM
            w_d = np.sign(e) * min(7.0, np.sqrt(2 * a_cap * 0.7 * abs(e)))          # 減速の曲線（上限のトルクの 7 割で止まれる速さ）
            tau_des = I_ARM * 20.0 * (w_d - w) + (I_ARM * 30.0 * e if abs(e) < R(3) else 0.0)
        tau_des = float(np.clip(tau_des, -TAU_CAP, TAU_CAP))
        cmd_brake = 0.0
        if kind == "mr":
            if t >= T0 and tau_des * np.sign(w + 1e-9) < 0 and abs(w) > 0.05:   # 減速したい → モーターは止め、MR で止める
                cmd_brake = abs(tau_des); tau_m = 0.0
            else:
                tau_m = max(tau_des, 0.0) if t >= T0 and abs(TH - th) > R(2) else tau_des
                if t >= T0 and abs(TH - th) <= R(2):
                    cmd_brake = 60.0                                                # 到着後は MR で保持
        else:
            tau_m = tau_des
        if power_off and t > T0 + T_PUSH - 1.0:          # 電源を切る（モーターのトルク 0）。MR は保持の磁場だけ残す（永久磁石と電磁石の組み合わせを想定）
            tau_m = 0.0
            cmd_brake = 60.0 if kind == "mr" else cmd_brake
        tau_dev = device_torque(kind, w, st, dt, cmd_brake)
        push = PUSH if T0 + T_PUSH <= t < T0 + T_PUSH + 0.3 else 0.0
        d.ctrl[ids["a"]] = tau_m
        d.qfrc_applied[ids["v"]] = tau_dev + push          # ブレーキと外からの押し（関節の内力として、反作用は車体へ）
        acc["heat"] += -tau_dev * w * dt
        p = tau_m * w * dt; acc["motor_work"] += max(p, 0.0); acc["motor_neg"] += max(-p, 0.0)
        st["last"] = (tau_m, tau_dev)

    def log(m, d):
        tm, td = st.get("last", (0.0, 0.0))
        return [-(tm + td), d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "arm_yaw")]], tm, td, st.get("lam", np.nan)]
    rec, s = run_drive(xml, joints, ARM0, 0.0, T0 + T_PUSH + 1.5, RS, pose_fun=lambda t, st_: {"arm_pitch": 0.0, "arm_elbow1": 0.0},
                       ctrl_fun=ctrl, log_fun=log, record_every=20)
    lg = np.array(rec["log"]); t = rec["t"]; arm = np.degrees(lg[:, 1]); yaw = np.degrees(rec["yaw"] - rec["yaw"][0])
    err = arm - 180.0
    arrive = next((float(t[i] - T0) for i in range(len(t)) if t[i] > T0 and np.all(np.abs(err[i:np.searchsorted(t, T0 + T_PUSH)]) < 2.0)), np.nan)
    before = (t > T0 + T_PUSH - 0.2) & (t < T0 + T_PUSH); during = (t >= T0 + T_PUSH) & (t < T0 + T_PUSH + 1.0)
    out = dict(kind=kind, name=DEVICES[kind], arrive_s=arrive, overshoot=float(max(0.0, err[(t > T0) & (t < T0 + T_PUSH)].max())),
               base_yaw=float(yaw[np.searchsorted(t, T0 + T_PUSH) - 1]),
               base_torque_peak=float(np.abs(np.convolve(lg[:, 0], np.ones(5) / 5, mode="same")[t < T0 + T_PUSH]).max()),
               motor_peak=float(np.abs(lg[:, 2][t < T0 + T_PUSH]).max()), device_peak=float(np.abs(lg[:, 3][t < T0 + T_PUSH]).max()),
               heat_J=acc["heat"], motor_work_J=acc["motor_work"], motor_regen_J=acc["motor_neg"],
               push_deflection=float(np.abs(err[during] - err[before].mean()).max()), max_roll=s["max_roll_deg"], diverged=s["diverged"])
    out["power_off"] = power_off
    if power_off:
        out["drift_after_push"] = float(abs(err[-1] - err[before].mean()))
    return (kind, power_off), out, {"t": t.tolist(), "torque": lg[:, 0].tolist(), "arm": arm.tolist(), "yaw": yaw.tolist(), "motor": lg[:, 2].tolist(), "dev": lg[:, 3].tolist()}


if __name__ == "__main__":
    t0 = time.time()
    with Pool(4) as pool:
        out = pool.map(trial, [(k, False) for k in DEVICES]); out_off = pool.map(trial, [(k, True) for k in DEVICES])
    base = [r for r in json.load(open(DATA / "field13_reaction.json")) if r["name"].startswith("13-0") and r["T"] == 1.0][0]
    print(f"（{len(out) + len(out_off)} 試行、{time.time() - t0:.0f} s）  モーターのトルクの上限 {TAU_CAP:.0f} N·m（境目 M_s = {M_S:.0f} N·m の {TAU_CAP / M_S:.1f} 倍）")
    print(f"比較の基準（段階 13、位置サーボで 1 s）: 車体の向き {base['base_yaw']:+.2f}°、反作用の最大 {base['base_torque_peak']:.0f} N·m")
    print(f"{'ブレーキ':30s} {'到着':>6s} {'行き過ぎ':>6s} {'車体の向き':>8s} {'反作用の最大':>9s} {'モーター':>7s} {'ブレーキ':>7s} {'熱':>6s} {'モーターの逆向きの仕事':>10s} {'押されたずれ':>8s}")
    res = []
    for _, o, _ in out:
        res.append(o)
        print(f"{o['name']:30s} {o['arrive_s']:5.2f}s {o['overshoot']:5.1f}° {o['base_yaw']:+7.2f}° {o['base_torque_peak']:7.0f} N·m {o['motor_peak']:5.0f} N·m {o['device_peak']:5.0f} N·m "
              f"{o['heat_J']:5.0f} J {o['motor_regen_J']:8.0f} J {o['push_deflection']:9.1f}°" + ("  発散" if o["diverged"] else ""))
    print(f"=== 電源を切った保持：到着後にモーターを止め、腕を外から {PUSH:.0f} N·m で 0.3 s 押す ===")
    for _, o, _ in out_off:
        res.append(o)
        print(f"  {o['name']:30s} 押し始めから 1 s の最大のずれ {o['push_deflection']:6.1f}°、押し終わって 1.2 s 後のずれ {o['drift_after_push']:6.1f}°")
    json.dump(res, open(DATA / "field14_braking.json", "w"), indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(1, 4, figsize=(20, 4.4))
    lab = {"none": "torque motor only", "viscous": "viscous", "mr": "MR (controlled)", "dilatant": "dilatant (shear thickening)",
           "thinning": "shear thinning", "thixo": "thixotropic"}
    for (kind, _), o, rec in out:
        ax[0].plot(rec["t"], rec["torque"], label=lab[kind]); ax[1].plot(rec["t"], rec["arm"], label=lab[kind]); ax[2].plot(rec["t"], rec["yaw"], label=lab[kind])
    ax[0].axhspan(-M_S, M_S, color="g", alpha=0.08); ax[0].set_title("reaction torque on hull (green: below breakaway)"); ax[0].set_ylabel("N m")
    ax[1].set_title(f"arm yaw (push {PUSH:.0f} N m at t = {T0 + T_PUSH:.1f} s)"); ax[1].set_ylabel("deg")
    ax[2].set_title("hull heading"); ax[2].set_ylabel("deg")
    for (kind, _), o, rec in out_off:
        t_ = np.array(rec["t"]); k = t_ > T0 + T_PUSH - 1.2
        ax[3].plot(t_[k], np.array(rec["arm"])[k] - 180.0, label=lab[kind])
    ax[3].axvspan(T0 + T_PUSH, T0 + T_PUSH + 0.3, color="0.85"); ax[3].set_title(f"power off, push {PUSH:.0f} N m (grey): arm deflection"); ax[3].set_ylabel("deg")
    for a in ax:
        a.set_xlabel("t [s]"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGF / "field14_braking.png", dpi=130)
    print("図 → figures/field/field14_braking.png")
