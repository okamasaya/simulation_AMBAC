"""Field15: 機能を分けた組み合わせ — 「速く動かす」「止める」「保持する」を別の部品に受け持たせ、上の制御で使い分ける（段階 15）.

段階 14 で、流体ブレーキは種類によって得意な場面が逆になることが分かった（ダイラタントは速い動きを止める、擬塑性・MR は保持）。
1 つの物質ですべてをこなすのではなく、機能ごとに部品を分けて組み合わせる。
  ① 基準：トルク制御のモーターだけ（段階 14）
  ② モーター＋MR（制御を改良）：加速はモーター、減速は MR（目標でちょうど止まるブレーキの強さを毎ステップ計算）、最後の微調整はモーター、保持は MR
  ③ モーター＋ダイラタント＋MR：②に、速さの上限と衝撃への備えとしてダイラタント（受け身）を加える
  ④ モーター＋ダイラタント＋擬塑性：受け身の部品だけ（制御する部品はモーターのみ）
試験 A: 荷物 10 kg を持った腕を 0 → 180° 振り、到着後にモーターの電源を切って、腕を外から 80 N·m で 0.3 s 押す（保持）
試験 C: 振っている途中（開始 0.5 s 後）に、腕に衝撃 150 N·m を 0.1 s 与える。逆向き（何かにぶつかった）と、同じ向き（後ろから押された）の 2 通り
流体の式と強さは段階 14 と同じ。モーターのトルクの上限 104 N·m（M_s の 0.4 倍）。
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
from field14_braking import ARM0, I_ARM, M_S, PUSH, R, RS, STAGE, T0, T_PUSH, TAU_CAP, TH, device_torque

FIGF = FIG / "field"
COMBOS = {"motor": ("① モーターだけ", [], False), "mr": ("② モーター＋MR（改良）", [], True),
          "dil_mr": ("③ モーター＋ダイラタント＋MR", ["dilatant"], True), "dil_thin": ("④ モーター＋ダイラタント＋擬塑性（受け身だけ）", ["dilatant", "thinning"], False)}
IMPACT_T, IMPACT = 0.5, -150.0


def trial(args):
    combo, impact = args
    name, passive, use_mr = COMBOS[combo]
    r = f11.robot(0.8); xml, joints = build(r, STAGE)
    xml = re.sub(r'<position name="arm_yaw" joint="arm_yaw"[^/]*/>', '<motor name="arm_yaw" joint="arm_yaw" gear="1"/>', xml)
    dt = mujoco.MjModel.from_xml_string(xml).opt.timestep
    st = {p: {} for p in passive}; mr = {"ty": 0.0, "mode": "accel", "calm": 0.0}; ids = {}
    acc = dict(heat=0.0, regen=0.0, work=0.0)

    def ctrl(m, d, t):
        if not ids:
            ids["a"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm_yaw")
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "arm_yaw"); ids["q"] = m.jnt_qposadr[j]; ids["v"] = m.jnt_dofadr[j]
        th = d.qpos[ids["q"]]; w = d.qvel[ids["v"]]; e = TH - th
        if t < T0:
            tau_des = I_ARM * 100.0 * (0.0 - th) - I_ARM * 10 * w
        else:
            w_d = np.sign(e) * min(7.0, np.sqrt(2 * (TAU_CAP / I_ARM) * 0.7 * abs(e)))
            tau_des = I_ARM * 20.0 * (w_d - w) + (I_ARM * 30.0 * e if abs(e) < R(3) else 0.0)
        tau_des = float(np.clip(tau_des, -TAU_CAP, TAU_CAP)); tau_m = tau_des; ty_cmd = 0.0
        if use_mr and t >= T0:
            md = mr["mode"]
            if md == "accel" and tau_des * np.sign(w + 1e-9) < 0 and abs(w) > 0.3:
                md = "brake"
            if md == "brake":
                # 目標でちょうど止まるブレーキの強さ: I ω² / (2 × 残りの距離)。行き過ぎたら上限で止める
                ty_cmd = TAU_CAP if e * np.sign(w) <= R(0.3) else min(TAU_CAP, I_ARM * w * w / (2 * abs(e)))
                tau_m = 0.0
                if abs(w) < 0.3:
                    md = "fine"
            if md == "fine":
                tau_m = tau_des; ty_cmd = 0.0
                mr["calm"] = mr["calm"] + dt if (abs(e) < R(0.5) and abs(w) < 0.05) else 0.0
                if mr["calm"] > 0.2:
                    md = "hold"
            if md == "hold":
                ty_cmd = 60.0; tau_m = 0.0                       # 保持は MR に任せ、モーターは休む
                if abs(e) > R(1.5):
                    md = "fine"; mr["calm"] = 0.0
            mr["mode"] = md
        if t > T0 + T_PUSH - 1.0:                               # 電源を切る（MR の保持の磁場は残す）
            tau_m = 0.0
            ty_cmd = 60.0 if use_mr else 0.0
        tau_dev = sum(device_torque(p, w, st[p], dt) for p in passive)
        if use_mr:
            mr["ty"] += dt / 0.01 * (ty_cmd - mr["ty"]); tau_dev += -(mr["ty"] * np.tanh(w / 0.05) + 1.0 * w)
        ext = 0.0
        if impact and T0 + IMPACT_T <= t < T0 + IMPACT_T + 0.1:
            ext += impact * abs(IMPACT)
        if T0 + T_PUSH <= t < T0 + T_PUSH + 0.3:
            ext += PUSH
        d.ctrl[ids["a"]] = tau_m; d.qfrc_applied[ids["v"]] = tau_dev + ext
        acc["heat"] += -tau_dev * w * dt; p = tau_m * w * dt; acc["work"] += max(p, 0.0); acc["regen"] += max(-p, 0.0)
        st["_last"] = (tau_m, tau_dev)

    def log(m, d):
        tm, td = st.get("_last", (0.0, 0.0))
        return [-(tm + td), d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "arm_yaw")]], tm, td]
    rec, s = run_drive(xml, joints, ARM0, 0.0, T0 + T_PUSH + 1.5, RS, pose_fun=lambda t, s_: {"arm_pitch": 0.0, "arm_elbow1": 0.0},
                       ctrl_fun=ctrl, log_fun=log, record_every=20)
    lg = np.array(rec["log"]); t = rec["t"]; arm = np.degrees(lg[:, 1]); err = arm - 180.0; yaw = np.degrees(rec["yaw"] - rec["yaw"][0])
    i_push = np.searchsorted(t, T0 + T_PUSH)
    arrive = next((float(t[i] - T0) for i in range(len(t)) if t[i] > T0 and np.all(np.abs(err[i:i_push - 100]) < 2.0)), np.nan)
    before = (t > T0 + T_PUSH - 0.2) & (t < T0 + T_PUSH)
    out = dict(combo=combo, name=name, impact=impact, arrive_s=arrive, overshoot=float(max(0.0, err[(t > T0) & (t < T0 + T_PUSH)].max())),
               final_err=float(abs(err[before].mean())), base_yaw=float(yaw[i_push - 1]),
               base_torque_peak=float(np.abs(np.convolve(lg[:, 0], np.ones(5) / 5, mode="same")[t < T0 + T_PUSH]).max()),
               heat_J=acc["heat"], motor_work_J=acc["work"], motor_regen_J=acc["regen"],
               push_drift=float(abs(err[-1] - err[before].mean())), max_roll=s["max_roll_deg"], diverged=s["diverged"])
    if impact:
        k = (t > T0 + IMPACT_T) & (t < T0 + IMPACT_T + 0.6)
        sp = np.gradient(arm, t)[k]
        out["impact_speed_min"] = float(sp.min()); out["impact_speed_max"] = float(sp.max())      # 衝撃の後の腕の速さの最小・最大（°/s）
    return args, out, {"t": t.tolist(), "arm": arm.tolist(), "torque": lg[:, 0].tolist(), "yaw": yaw.tolist()}


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(c, 0) for c in COMBOS] + [(c, -1) for c in COMBOS] + [(c, +1) for c in COMBOS]
    with Pool(4) as pool:
        out = pool.map(trial, jobs)
    print(f"（{len(jobs)} 試行、{time.time() - t0:.0f} s）  モーターのトルクの上限 {TAU_CAP:.0f} N·m")
    print("=== 試験 A：振る（到着後に電源を切り、80 N·m で 0.3 s 押す）===")
    print(f"{'組み合わせ':32s} {'到着':>6s} {'行き過ぎ':>6s} {'止まった位置の誤差':>8s} {'車体の向き':>8s} {'反作用':>7s} {'熱':>6s} {'モーターの逆向き':>8s} {'押された後のずれ':>8s}")
    res = []
    for (c, imp), o, _ in out:
        res.append(o)
        if not imp:
            print(f"{o['name']:32s} {o['arrive_s']:5.2f}s {o['overshoot']:5.1f}° {o['final_err']:9.2f}° {o['base_yaw']:+7.2f}° {o['base_torque_peak']:5.0f} N·m "
                  f"{o['heat_J']:5.0f} J {o['motor_regen_J']:8.0f} J {o['push_drift']:10.1f}°")
    print(f"=== 試験 C：振っている途中（開始 {IMPACT_T} s 後）に、衝撃 {abs(IMPACT):.0f} N·m を 0.1 s（参考：衝撃なしの腕の最大の速さは約 300 °/s）===")
    for (c, imp), o, _ in out:
        if imp:
            d_ = "逆向き（ぶつかった）" if imp < 0 else "同じ向き（押された）"
            print(f"  {d_} {o['name']:32s} 到着 {o['arrive_s']:5.2f} s、行き過ぎ {o['overshoot']:4.1f}°、車体の向き {o['base_yaw']:+.2f}°、"
                  f"衝撃の後の腕の速さ {o['impact_speed_min']:5.0f}〜{o['impact_speed_max']:5.0f} °/s")
    json.dump(res, open(DATA / "field15_combined.json", "w"), indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    lab = {"motor": "motor only", "mr": "motor + MR (improved)", "dil_mr": "motor + dilatant + MR", "dil_thin": "motor + dilatant + shear thinning (passive)"}
    for (c, imp), o, rec in out:
        ls = "--" if imp else "-"
        if not imp:
            ax[0].plot(rec["t"], np.array(rec["arm"]) - 180, label=lab[c])
            ax[2].plot(rec["t"], np.array(rec["arm"]) - 180, label=lab[c])
        else:
            if imp > 0:
                ax[1].plot(rec["t"], rec["arm"], ls, label=lab[c])
    ax[0].set_xlim(T0 + 0.8, T0 + 2.6); ax[0].set_ylim(-6, 6); ax[0].set_title("A: arrival near 180 deg (error)"); ax[0].set_ylabel("deg")
    ax[1].axvspan(T0 + IMPACT_T, T0 + IMPACT_T + 0.1, color="0.85"); ax[1].set_xlim(T0 - 0.2, T0 + 2.5)
    ax[1].set_title(f"C: push along +{abs(IMPACT):.0f} N m mid-swing (grey)"); ax[1].set_ylabel("arm yaw [deg]")
    ax[2].axvspan(T0 + T_PUSH, T0 + T_PUSH + 0.3, color="0.85"); ax[2].set_xlim(T0 + T_PUSH - 1.2, T0 + T_PUSH + 1.5)
    ax[2].set_title(f"B: power off, push {PUSH:.0f} N m (grey)"); ax[2].set_ylabel("arm error [deg]")
    for a in ax:
        a.set_xlabel("t [s]"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGF / "field15_combined.png", dpi=130)
    print("図 → figures/field/field15_combined.png")
