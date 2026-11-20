"""Field16: 繰り返し動かしたときの発熱と放熱 — MR 流体の温度による性能低下と劣化（段階 16）.

段階 15 の「モーター＋MR（動かす＝モーター、止める・保持する＝MR）」を、何百回も繰り返したときに何が起きるかを調べる。
腕の動き（約 1 s）と温度の変化（数分〜数時間）は時間の桁が違うため、2 つを組み合わせる。
  (1) MuJoCo：MR の最大のブレーキの強さを、公称値の 100〜20% に下げたとき、1 回の振りがどう変わるか
      （足りない分はモーターが逆向きのトルクで補う）。1 回あたりのブレーキの熱、モーターの銅損、行き過ぎを表にする
  (2) 熱のモデル（熱容量 C と、周囲への熱抵抗 R の 1 次の式）：C dT/dt = P − (T − T_周囲)/R
      MR の温度が上がると効きが下がり（係数 β %/℃、仮定）、モーターの負担と発熱が増える連鎖を含める
仮定した値（実在の製品の値ではない。範囲として扱う）:
  MR ブレーキ: 公称の最大トルク 150 N·m、熱容量 700 J/K、熱抵抗 1.5 K/W（自然放熱）/ 0.5 K/W（ファン）、流体 20 cm³、使用の上限 120 ℃
              効きの温度係数 β = 0.1 / 0.3 / 0.5 %/℃（具体的な値は確認できなかった）
              保持に電力が要らない方式（永久磁石と電磁石の組み合わせ）と、電磁石で保持する方式（コイル 15 W を常時）
  モーター:   銅損 P = α τ²（α = 0.04 W/(N·m)²：50 N·m を出し続けると 100 W）、巻線の抵抗は 0.39 %/℃ で増える、
              熱容量 600 J/K、熱抵抗 0.8 K/W、巻線の上限 120 ℃。回生の効率 60 %
  寿命:       MR 流体の一生に熱に変えられるエネルギー（LDE）10⁷ J/cm³（Carlson）
動作: 0 → 180° → 0 → … と往復し、1 回振るごとに 1.0 s 止まる（保持）。周囲 25 ℃。
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
from ambac.field.assembly import build
from ambac.field.drive import run_drive
import field11_turning as f11
from field14_braking import ARM0, I_ARM, R, RS, STAGE, T0, TAU_CAP, TH

FIGF = FIG / "field"
MR_NOM = 150.0
ALPHA = 0.04
DWELL = 1.0
T_AMB = 25.0


def swing(args):
    """1 回の振り（0 → 180°）。mode: "motor"（モーターだけ）または "mr"（MR の最大のトルク = cap）"""
    mode, cap = args
    r = f11.robot(0.8); xml, joints = build(r, STAGE)
    xml = re.sub(r'<position name="arm_yaw" joint="arm_yaw"[^/]*/>', '<motor name="arm_yaw" joint="arm_yaw" gear="1"/>', xml)
    dt = mujoco.MjModel.from_xml_string(xml).opt.timestep
    mr = {"ty": 0.0, "mode": "accel", "calm": 0.0}; ids = {}
    acc = dict(heat=0.0, cu=0.0, regen=0.0, work=0.0, t_done=np.nan)

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
        if mode == "mr" and t >= T0:
            md = mr["mode"]
            if md == "accel" and tau_des * np.sign(w + 1e-9) < 0 and abs(w) > 0.3:
                md = "brake"
            if md == "brake":
                need = TAU_CAP if e * np.sign(w) <= R(0.3) else min(TAU_CAP, I_ARM * w * w / (2 * abs(e)))
                ty_cmd = min(need, cap)
                tau_m = -np.sign(w) * max(0.0, need - cap)          # MR で足りない分はモーターが逆向きに補う
                if abs(w) < 0.3:
                    md = "fine"
            if md == "fine":
                tau_m = tau_des; ty_cmd = 0.0
                mr["calm"] = mr["calm"] + dt if (abs(e) < R(0.5) and abs(w) < 0.05) else 0.0
                if mr["calm"] > 0.2:
                    md = "hold"
            if md == "hold":
                ty_cmd = min(60.0, cap); tau_m = 0.0
            mr["mode"] = md
        tau_dev = 0.0
        if mode == "mr":
            mr["ty"] += dt / 0.01 * (ty_cmd - mr["ty"]); tau_dev = -(mr["ty"] * np.tanh(w / 0.05) + 1.0 * w)
        d.ctrl[ids["a"]] = tau_m; d.qfrc_applied[ids["v"]] = tau_dev
        if t >= T0 and np.isnan(acc["t_done"]):
            acc["heat"] += -tau_dev * w * dt; acc["cu"] += ALPHA * tau_m**2 * dt
            p = tau_m * w * dt; acc["work"] += max(p, 0.0); acc["regen"] += max(-p, 0.0)
            if abs(e) < R(0.5) and abs(w) < 0.05 and t > T0 + 0.5:
                acc["t_done"] = t - T0
    rec, s = run_drive(xml, joints, ARM0, 0.0, T0 + 3.0, RS, pose_fun=lambda t, s_: {"arm_pitch": 0.0, "arm_elbow1": 0.0},
                       ctrl_fun=ctrl, log_fun=lambda m, d: [d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "arm_yaw")]]],
                       record_every=20)
    arm = np.degrees(np.array(rec["log"])[:, 0]); t = rec["t"]; yaw = np.degrees(rec["yaw"] - rec["yaw"][0])
    return dict(mode=mode, cap=cap, overshoot=float(max(0.0, (arm[t > T0] - 180).max())), t_move=acc["t_done"],
                brake_heat=acc["heat"], motor_cu=acc["cu"], motor_work=acc["work"], motor_regen=acc["regen"], base_yaw=float(yaw[-1]))


def sustainable_dwell(table, **kw):
    """温度が上限（120 ℃）を超えずに、4 時間動き続けられる最短の休み時間（二分法）"""
    lo, hi = 0.0, 60.0
    if thermal(table, dwell=hi, hours=4.0, **kw)["limit"]:
        return np.inf
    for _ in range(14):
        mid = (lo + hi) / 2
        if thermal(table, dwell=mid, hours=4.0, **kw)["limit"]:
            lo = mid
        else:
            hi = mid
    return hi


def thermal(table, motor_only, beta=0.003, R_mr=1.5, coil_hold=0.0, hours=2.0, regen_eff=0.6, dwell=DWELL, stop_at_limit=True):
    """繰り返しの熱の計算（1 回の振りごとに温度を更新）。table: MR の効きの割合 → 1 回の振りの量"""
    if not motor_only:
        caps = np.array(sorted(table)); key = lambda k: np.array([table[c][k] for c in caps])
    T_mr = T_AMB; T_m = T_AMB; t = 0.0; n = 0; lde = 0.0; batt = 0.0
    hist = []; reason = None
    while t < hours * 3600:
        if motor_only:
            q = table["motor"]; frac = None
        else:
            frac = max(0.0, 1.0 - beta * (T_mr - T_AMB))            # MR の効きの割合（公称に対して）
            q = {k: float(np.interp(frac, caps, key(k))) for k in ("t_move", "brake_heat", "motor_cu", "motor_work", "motor_regen", "overshoot")}
        dt_cycle = q["t_move"] + dwell
        rcu = 1 + 0.0039 * (T_m - T_AMB)                               # 巻線の抵抗の増加
        P_m = q["motor_cu"] * rcu / dt_cycle
        T_m += dt_cycle * (P_m - (T_m - T_AMB) / 0.8) / 600.0
        if not motor_only:
            P_mr = q["brake_heat"] / dt_cycle + coil_hold
            T_mr += dt_cycle * (P_mr - (T_mr - T_AMB) / R_mr) / 700.0
            lde += q["brake_heat"]
        batt += q["motor_work"] + q["motor_cu"] * rcu - regen_eff * q["motor_regen"]
        t += dt_cycle; n += 1
        if n % 20 == 0:
            hist.append((t, T_m, T_mr if not motor_only else np.nan, frac if frac is not None else np.nan, q["overshoot"]))
        if T_m > 120 and reason is None:
            reason = ("motor", t, n)
        if not motor_only and T_mr > 120 and reason is None:
            reason = ("mr", t, n)
        if reason and stop_at_limit:
            break
    return dict(hist=np.array(hist), limit=reason, n=n, T_m=T_m, T_mr=T_mr, lde_J=lde, battery_J_per_cycle=batt / n,
                frac_end=frac if not motor_only else None, overshoot_end=q["overshoot"])


if __name__ == "__main__":
    t0 = time.time()
    CAPS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    with Pool(4) as pool:
        out = pool.map(swing, [("motor", 0.0)] + [("mr", MR_NOM * c) for c in CAPS])
    table = {"motor": out[0]}
    for c, o in zip(CAPS, out[1:]):
        table[c] = o
    print(f"（MuJoCo {len(out)} 試行、{time.time() - t0:.0f} s）")
    print("=== (1) 1 回の振り（0 → 180°）：MR の効きを下げたとき ===")
    print(f"{'構成':26s} {'動作時間':>7s} {'行き過ぎ':>6s} {'ブレーキの熱':>8s} {'モーターの銅損':>9s} {'モーターの逆向きの仕事':>10s} {'車体の向き':>8s}")
    print(f"{'モーターだけ':26s} {table['motor']['t_move']:6.2f}s {table['motor']['overshoot']:5.1f}° {0:7.0f} J {table['motor']['motor_cu']:8.0f} J {table['motor']['motor_regen']:10.0f} J {table['motor']['base_yaw']:+7.2f}°")
    for c in CAPS:
        o = table[c]
        print(f"{'モーター＋MR（効き ' + str(int(c * 100)) + '%）':26s} {o['t_move']:6.2f}s {o['overshoot']:5.1f}° {o['brake_heat']:7.0f} J {o['motor_cu']:8.0f} J {o['motor_regen']:10.0f} J {o['base_yaw']:+7.2f}°")
    mr_table = {c: table[c] for c in CAPS}
    cases = [("モーターだけ（回生 60%）", dict(motor_only=True)),
             ("モーター＋MR（保持に電力不要、自然放熱）", dict(motor_only=False)),
             ("モーター＋MR（電磁石で保持 15 W、自然放熱）", dict(motor_only=False, coil_hold=15.0)),
             ("モーター＋MR（保持に電力不要、ファン）", dict(motor_only=False, R_mr=0.5))]
    print(f"=== (2) 繰り返し（往復し、1 回ごとに {DWELL} s 保持。どれかが 120 ℃ に達したら停止。MR の効きの温度係数 β = 0.3 %/℃）===")
    res = dict(swing=table, thermal={})
    curves = {}
    for nm, kw in cases:
        th = thermal(mr_table if not kw.get("motor_only") else table, **kw)
        curves[nm] = th["hist"]
        lim = th["limit"]
        lim_s = f"{'モーター' if lim[0] == 'motor' else 'MR'}が 120 ℃ に到達して停止：{lim[1] / 60:.1f} 分（{lim[2]} 回）" if lim else "2 時間で 120 ℃ に達しない"
        mr_s = "" if kw.get("motor_only") else f"、MR {th['T_mr']:.0f} ℃（効き {th['frac_end'] * 100:.0f}%）"
        dw = sustainable_dwell(mr_table if not kw.get("motor_only") else table, **kw)
        q = table["motor"] if kw.get("motor_only") else table[1.0]
        rate = 60.0 / (q["t_move"] + dw)
        print(f"  {nm:34s}: {lim_s}。停止時 モーター {th['T_m']:.0f} ℃{mr_s}。"
              f"続けられる動作：1 回ごとに {dw:.1f} s 休む → 毎分 {rate:.1f} 回、1 回あたりの電池のエネルギー {th['battery_J_per_cycle']:.0f} J")
        res["thermal"][nm + "_sustainable"] = dict(dwell=dw, per_min=rate)
        res["thermal"][nm] = dict(limit=lim, T_m=th["T_m"], T_mr=th["T_mr"], battery_J_per_cycle=th["battery_J_per_cycle"], frac_end=th["frac_end"])
    print("=== (3) MR の効きの温度係数 β による違い（保持に電力不要、自然放熱、続けられる最大の頻度で 4 時間）===")
    for beta in (0.001, 0.003, 0.005):
        dw = sustainable_dwell(mr_table, motor_only=False, beta=beta)
        th = thermal(mr_table, False, beta=beta, dwell=dw, hours=4.0)
        print(f"  β = {beta * 100:.1f} %/℃: 続けられる休み時間 {dw:.1f} s（毎分 {60 / (table[1.0]['t_move'] + dw):.1f} 回）。そのときの 4 時間後 MR {th['T_mr']:.0f} ℃、"
              f"効き {th['frac_end'] * 100:.0f}%、モーター {th['T_m']:.0f} ℃、行き過ぎ {th['overshoot_end']:.1f}°")
        res["thermal"][f"beta_{beta}"] = dict(limit=lim, T_mr=th["T_mr"], frac_end=th["frac_end"], T_m=th["T_m"])
    vol = 20.0; lde = 1e7 * vol; per = table[1.0]["brake_heat"]
    n_life = lde / per
    print(f"=== (4) 寿命の見積もり：LDE 10⁷ J/cm³ × 流体 {vol:.0f} cm³ = {lde:.1e} J、1 回の振りで {per:.0f} J → 約 {n_life:.1e} 回"
          f"（{DWELL} s ごとに往復し続けると約 {n_life * (table[1.0]['t_move'] + DWELL) / 3600:.0f} 時間）===")
    res["life_cycles"] = n_life
    json.dump(res, open(DATA / "field16_thermal.json", "w"), indent=2, ensure_ascii=False, default=float)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    caps = np.array(CAPS)
    ax[0].plot(caps * 100, [table[c]["brake_heat"] for c in CAPS], "o-", label="MR brake heat")
    ax[0].plot(caps * 100, [table[c]["motor_regen"] for c in CAPS], "s-", label="motor reverse work (regen)")
    ax[0].plot(caps * 100, [table[c]["motor_cu"] for c in CAPS], "^-", label="motor copper loss")
    ax[0].set_xlabel("MR brake capacity [% of nominal]"); ax[0].set_ylabel("J per swing"); ax[0].set_title("one swing: who absorbs the braking energy"); ax[0].legend(fontsize=8)
    labs = {"モーターだけ（回生 60%）": "motor only", "モーター＋MR（保持に電力不要、自然放熱）": "motor + MR (no hold power)",
            "モーター＋MR（電磁石で保持 15 W、自然放熱）": "motor + MR (coil hold 15 W)", "モーター＋MR（保持に電力不要、ファン）": "motor + MR + fan"}
    for nm, h in curves.items():
        if len(h):
            ax[1].plot(h[:, 0] / 60, h[:, 2], label=labs[nm]); ax[2].plot(h[:, 0] / 60, h[:, 1], label=labs[nm])
    for a, ttl in ((ax[1], "MR brake temperature"), (ax[2], "motor winding temperature")):
        a.axhline(120, color="r", ls=":", lw=1); a.set_xlabel("time [min]"); a.set_ylabel("deg C"); a.set_title(ttl); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGF / "field16_thermal.png", dpi=130)
    print("図 → figures/field/field16_thermal.png")
