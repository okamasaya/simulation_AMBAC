"""Field17: 3 軸の肩（旋回・俯仰・ひねり）＋肘の腕全体の動作に、機能を分けた部品（動かす＝モーター、止める・保持する＝MR）を当てはめる（段階 17）.

機体: 左右 2 本の履帯（段階 11）に、3 軸の肩＋肘の腕（荷物 10 kg）。4 関節を同時に動かす。重力に逆らう保持が入るのが、これまでとの違い。
動作: 前の低い位置 → (区間 1) 左横の高い位置 → 1 s 止まる → (区間 2) 胸の前に腕をたたむ → 1 s 止まる → 電源を切って 5 s
（MR は、電源が切れたら、その位置で全関節をつかむ）
方式:
  ① 位置サーボ・速い（1 区間 1 s、最小ジャーク）   ② 位置サーボ・遅い（1 区間 2 s）
  ③ トルク制御のモーター（減速の曲線、重力の分は打ち消す、トルクの上限は関節ごと）
  ④ モーター＋MR（関節ごと）：加速はモーター、減速は MR（目標で止まる強さを計算）、最後の微調整はモーター、保持は MR（電力なし）
関節ごとのトルクの上限（モーター）: 旋回 104（車体が滑らない範囲、段階 14）、俯仰 300、ひねり 80、肘 150 N·m。MR の最大: 150 / 350 / 100 / 200 N·m
モーターの銅損 0.04 τ² W（段階 16 と同じ。位置サーボも同じ式で、出したトルクから計算する）。
電源を切った後は、荷物の高さがどれだけ下がるかで、保持の強さを比べる。
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

FIGF = FIG / "field"
R = np.radians
STAGE = load("stages", "practice"); RS = 0.10
J = ["arm_yaw", "arm_pitch", "arm_roll", "arm_elbow1"]
P0 = np.array(R([0, -30, 0, -60]))          # 前の低い位置
P1 = np.array(R([120, 40, 60, -90]))        # 左横の高い位置
P2 = np.array(R([-60, 60, -30, -120]))      # 胸の前にたたむ
CAP = np.array([104.0, 300.0, 80.0, 150.0]); MRMAX = np.array([150.0, 350.0, 100.0, 200.0])
ALPHA = 0.04
T_START, DWELL, T_OFF = 1.0, 1.0, 5.0


def robot():
    r = f11.robot(0.8); r["arms"][0]["shoulder"] = "ball3"
    return r


def setup(scheme):
    """方式ごとのモデル（XML）と制御（毎ステップ呼ぶ関数）を作る。GIF でも使う"""
    xml, joints = build(robot(), STAGE)
    torque_mode = scheme in ("motor", "mr")
    if torque_mode:
        for j in J:
            xml = re.sub(rf'<position name="{j}" joint="{j}"[^/]*/>', f'<motor name="{j}" joint="{j}" gear="1"/>', xml)
    m0 = mujoco.MjModel.from_xml_string(xml); dt = m0.opt.timestep
    Tseg = {"servo_fast": 1.0, "servo_slow": 2.0}.get(scheme, None)
    st = dict(seg=0, t_arrive=[np.nan, np.nan], t_seg0=T_START, mode=["accel"] * 4, calm=np.zeros(4), ty=np.zeros(4), hold=np.zeros(4),
              cu=0.0, heat=0.0, power_off_t=None)
    ids = {}

    def target(seg):
        return P1 if seg == 0 else P2

    def ctrl(m, d, t):
        if not ids:
            ids["a"] = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in J]
            jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in J]
            ids["q"] = [m.jnt_qposadr[i] for i in jid]; ids["v"] = [m.jnt_dofadr[i] for i in jid]
            ids["n"] = 0
        if ids["n"] % 50 == 0:                                        # 姿勢で変わる関節の慣性を、こまめに計算し直す
            M = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, d, M); ids["I"] = np.array([M[v, v] for v in ids["v"]])
        ids["n"] += 1
        q = d.qpos[ids["q"]]; w = d.qvel[ids["v"]]
        seg = st["seg"]; tgt = target(seg); start = P0 if seg == 0 else P1
        # 区間の切り替え: 到着（全関節が 1° 以内）してから DWELL 秒後に次の区間へ
        err = tgt - q
        if np.isnan(st["t_arrive"][seg]) and t > st["t_seg0"] + 0.3 and np.all(np.abs(err) < R(1.0)) and np.all(np.abs(w) < 0.05):
            st["t_arrive"][seg] = t
        if Tseg is not None and np.isnan(st["t_arrive"][seg]) and t > st["t_seg0"] + Tseg + 1.5:
            st["t_arrive"][seg] = t                                   # 位置サーボはおおむね着いたとみなす
        if seg == 0 and not np.isnan(st["t_arrive"][0]) and t > st["t_arrive"][0] + DWELL:
            st["seg"] = 1; st["t_seg0"] = t; st["mode"] = ["accel"] * 4; st["calm"][:] = 0
            seg = 1; tgt = target(1); start = P1; err = tgt - q
        if seg == 1 and not np.isnan(st["t_arrive"][1]) and st["power_off_t"] is None and t > st["t_arrive"][1] + DWELL:
            st["power_off_t"] = t
        off = st["power_off_t"] is not None
        if not torque_mode:
            if t < T_START:
                d.ctrl[ids["a"]] = P0
            elif not off:
                u = np.clip((t - st["t_seg0"]) / Tseg, 0, 1); s = 10 * u**3 - 15 * u**4 + 6 * u**5
                d.ctrl[ids["a"]] = start + (tgt - start) * s
                st["cu"] += ALPHA * np.sum(d.actuator_force[ids["a"]]**2) * dt
            else:
                # 電源を切る: 位置サーボの力を 0 にする（ゲインを 0 に相当）。現在の角度を指令にし、関節は自由に動く
                d.ctrl[ids["a"]] = q
                d.qfrc_applied[ids["v"]] = -(d.actuator_force[ids["a"]])
            return
        g = d.qfrc_bias[ids["v"]]
        if t < T_START:
            tgt = P0; err = tgt - q
        a_cap = CAP / ids["I"]
        w_d = np.sign(err) * np.minimum(4.0, np.sqrt(2 * a_cap * 0.6 * np.abs(err)))
        tau = ids["I"] * 20.0 * (w_d - w) + np.where(np.abs(err) < R(3), ids["I"] * 40.0 * err, 0.0) + g
        tau = np.clip(tau, -CAP, CAP)
        tau_m = tau.copy(); ty_cmd = np.zeros(4); hold = np.zeros(4, bool)
        if scheme == "mr" and t >= T_START:
            for k in range(4):
                md = st["mode"][k]
                if md == "accel" and tau[k] * np.sign(w[k] + 1e-9) < 0 and abs(w[k]) > 0.2:
                    md = "brake"
                if md == "brake":
                    need = abs(tau[k])
                    ty_cmd[k] = min(need, MRMAX[k]); tau_m[k] = 0.0 if need <= MRMAX[k] else tau[k] + np.sign(w[k]) * MRMAX[k]
                    if abs(w[k]) < 0.2:
                        md = "fine"
                if md == "fine":
                    st["calm"][k] = st["calm"][k] + dt if (abs(err[k]) < R(0.5) and abs(w[k]) < 0.05) else 0.0
                    if st["calm"][k] > 0.2:
                        md = "hold"; st["hold"][k] = q[k]
                if md == "hold":
                    hold[k] = True; tau_m[k] = 0.0
                    if abs(q[k] - tgt[k]) > R(1.5):
                        md = "fine"
                st["mode"][k] = md
        if off:
            tau_m[:] = 0.0                                            # 電源を切る（MR の保持の磁場は残す）
            if scheme == "mr":
                for k in range(4):                                    # 電源が切れたら、その位置で全関節を MR でつかむ
                    if st["mode"][k] != "hold":
                        st["mode"][k] = "hold"; st["hold"][k] = q[k]
                    hold[k] = True
        # MR のトルク: 減速は粘塑性（ビンガム）、保持は静止摩擦（ずれに比例して最大 MRMAX まで）
        tau_mr = np.zeros(4)
        if scheme == "mr":
            st["ty"] += dt / 0.01 * (ty_cmd - st["ty"])
            for k in range(4):
                if hold[k]:
                    tau_mr[k] = np.clip(-50000.0 * (q[k] - st["hold"][k]) - 1500.0 * w[k], -MRMAX[k], MRMAX[k])
                else:
                    tau_mr[k] = -(st["ty"][k] * np.tanh(w[k] / 0.05) + 1.0 * w[k])
        d.ctrl[ids["a"]] = tau_m; d.qfrc_applied[ids["v"]] = tau_mr
        st["cu"] += ALPHA * np.sum(tau_m**2) * dt; st["heat"] += -np.sum(tau_mr * w) * dt

    T_total = T_START + 2 * (4.0 if Tseg is None else Tseg + 1.6) + 2 * DWELL + T_OFF + 1.0
    return xml, joints, ctrl, st, T_total


def hold_power():
    """目標の姿勢（P1・P2）で、重力に逆らって止まっているのに必要なモーターの銅損 [W]"""
    xml, _ = build(robot(), STAGE); m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    out = {}
    for nm, P in (("P1", P1), ("P2", P2)):
        mujoco.mj_resetData(m, d)
        for j, v in zip(J, P):
            d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] = v
        mujoco.mj_forward(m, d)
        g = np.array([d.qfrc_bias[m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] for j in J])
        out[nm] = dict(torque=g.tolist(), copper_W=float(ALPHA * np.sum(g**2)))
    return out


def trial(scheme):
    xml, joints, ctrl, st, T_total = setup(scheme)

    def log(m, d):
        p = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "arm_payload")]
        return [*d.qpos[[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in J]], *p]
    rec, s = run_drive(xml, joints, dict(zip(J, P0)), 0.0, T_total, RS, ctrl_fun=ctrl, log_fun=log, record_every=20)
    lg = np.array(rec["log"]); t = rec["t"]
    # 目標の姿勢での荷物の位置（同じモデルで順運動学）
    mm = mujoco.MjModel.from_xml_string(xml); dd = mujoco.MjData(mm); mujoco.mj_forward(mm, dd)
    for j, v in zip(J, P2):
        dd.qpos[mm.jnt_qposadr[mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_JOINT, j)]] = v
    mujoco.mj_forward(mm, dd)
    base0 = dd.xpos[1].copy(); p_goal = dd.xpos[mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, "arm_payload")] - base0
    yaw = np.degrees(rec["yaw"] - rec["yaw"][0])
    t_off = st["power_off_t"] or t[-1]
    i_off = np.searchsorted(t, t_off - 0.05) - 1
    q_off = lg[i_off, :4]; i_end = len(t) - 1
    z_off = lg[i_off, 6]; drop = float((z_off - lg[i_off:, 6].min()) * 1000)          # 電源を切った後の荷物の高さの下がり [mm]
    before_off = t < t_off
    t_done = st["t_arrive"][1] - T_START - DWELL if not np.isnan(st["t_arrive"][1]) else np.nan
    out = dict(scheme=scheme, t_motion=float(t_done), err_final_deg=float(np.degrees(np.abs(P2 - q_off)).max()),
               base_yaw=float(yaw[i_off]), max_roll=float(np.degrees(np.abs(np.array(rec["roll"])[before_off]).max())),
               max_roll_after_off=float(np.degrees(np.abs(np.array(rec["roll"])[~before_off]).max())) if (~before_off).any() else np.nan,
               motor_cu_J=st["cu"], mr_heat_J=st["heat"], power_off_drop_mm=drop, diverged=s["diverged"])
    return scheme, out, {"t": t.tolist(), "q": lg[:, :4].tolist(), "yaw": yaw.tolist(), "off": st["power_off_t"]}


if __name__ == "__main__":
    t0 = time.time()
    with Pool(4) as pool:
        out = pool.map(trial, ["servo_fast", "servo_slow", "motor", "mr"])
    names = {"servo_fast": "① 位置サーボ・速い（1 区間 1 s）", "servo_slow": "② 位置サーボ・遅い（1 区間 2 s）",
             "motor": "③ トルク制御のモーター", "mr": "④ モーター＋MR（関節ごと）"}
    print(f"（{len(out)} 試行、{time.time() - t0:.0f} s）")
    print(f"{'方式':28s} {'2 区間の動作時間':>10s} {'関節の誤差':>8s} {'車体の向き':>8s} {'横の傾き':>7s} {'モーターの銅損':>10s} {'MR の熱':>7s} {'電源を切ると荷物が':>10s} {'そのときの横の傾き':>9s}")
    res = []
    for sch, o, _ in out:
        o["name"] = names[sch]; res.append(o)
        print(f"{o['name']:28s} {o['t_motion']:9.2f} s {o['err_final_deg']:7.2f}° {o['base_yaw']:+7.2f}° {o['max_roll']:6.1f}° {o['motor_cu_J']:9.0f} J {o['mr_heat_J']:6.0f} J {o['power_off_drop_mm']:7.0f} mm 下がる {o['max_roll_after_off']:8.1f}°"
              + ("  発散" if o["diverged"] else ""))
    hp = hold_power()
    for nm, v in hp.items():
        print(f"重力に逆らって {nm} の姿勢で止まっているのに必要な関節のトルク {np.round(v['torque'], 0)} N·m → モーターなら銅損 {v['copper_W']:.0f} W"
              f"（1 分間で {v['copper_W'] * 60 / 1000:.0f} kJ）。MR の保持なら 0 W（保持に電力が要らない方式）")
    res.append(dict(hold_power=hp))
    json.dump(res, open(DATA / "field17_arm3d.json", "w"), indent=2, ensure_ascii=False, default=float)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    lab = {"servo_fast": "servo, 1 s / segment", "servo_slow": "servo, 2 s / segment", "motor": "torque motor", "mr": "motor + MR"}
    for sch, o, rec in out:
        t = np.array(rec["t"]); q = np.degrees(np.array(rec["q"]))
        ax[0].plot(t, q[:, 1], label=lab[sch]); ax[1].plot(t, rec["yaw"], label=lab[sch])
        if rec["off"]:
            ax[0].axvline(rec["off"], color=ax[0].lines[-1].get_color(), ls=":", lw=0.8)
    ax[0].set_title("shoulder pitch (dotted: power off)"); ax[0].set_ylabel("deg")
    ax[1].set_title("hull heading"); ax[1].set_ylabel("deg")
    xs = np.arange(4); w_ = 0.35
    ax[2].bar(xs - w_ / 2, [o["motor_cu_J"] for _, o, _ in out], w_, label="motor copper loss")
    ax[2].bar(xs + w_ / 2, [o["mr_heat_J"] for _, o, _ in out], w_, label="MR heat")
    ax[2].set_xticks(xs); ax[2].set_xticklabels([lab[s] for s, _, _ in out], rotation=15, fontsize=8); ax[2].set_ylabel("J"); ax[2].set_title("heat over the whole task"); ax[2].legend()
    for a in ax[:2]:
        a.set_xlabel("t [s]"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGF / "field17_arm3d.png", dpi=130)
    print("図 → figures/field/field17_arm3d.png")
