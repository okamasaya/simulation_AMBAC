"""Field19: 二本脚の足裏の摩擦と足首の強さ — 滑り・転倒の境目を測る（下半身の段 2）.

段階 9 の二本脚（biped_waist：脚 24 kg・足裏 0.25 × 0.3 m、腰の上の上体 56 kg、総質量 97 kg）は、
脚と足を 1 つの箱にまとめ、足首は「十分に強い」と仮定していた。ここでは
  足（2 kg、0.25 × 0.3 × 0.05 m、床と接触）＋ 足首（高さ 0.08 m、pitch と roll の 2 軸、位置サーボ）＋ 脚（22 kg）
に分け、段階 9 の荷物運び（静的なつり合いの腰／動的な補正の腰、最短時間 T）で次の境目を測る。
  (1) 滑り   : 床の摩擦係数 μ を下げていき、足が 5 mm 以上ずれる境目 μ*
  (2) 足首   : 足首のトルクの上限 τ_max を下げていき、足首が 1° 以上曲がる（脚ごと倒れ始める）境目 τ*（pitch・roll 別）
  (3) 転倒   : 足裏の長さ（前後）・幅（左右）を縮めていき、足が 1° 以上浮く（かかと・つま先が上がる）境目 L*
照合の層
  1 層目（予測）: 摩擦・足首が十分に強い（μ = 2、足首 kp 10 倍）ときの床反力と足首のトルクから
                  必要な摩擦 μ_req(t) = |F_t|/F_n、必要な足首のトルク τ_req(t)、足圧中心（CoP）の位置を求め、
                  境目を μ* ≈ max μ_req、τ* ≈ max |τ_req|、L* ≈ 2·max |CoP| と予測する。
                  床反力は重心の加速度（ニュートンの式、M(a − g)）、足首のトルクは脚より上の角運動量の変化（オイラーの式）
                  からも独立に計算し、MuJoCo の接触力・アクチュエータの力と照合する。
  2 層目（実測）: MuJoCo で μ・τ_max・足裏の大きさを二分法で動かし、成功・失敗の境目を直接求める。
誤差の評価
  計算誤差: 時間刻み、積分法、摩擦円錐（楕円／多角形）、impratio
  系統誤差: 足裏の接触の形（箱／4 隅の点）、接触の柔らかさ、足首のサーボの硬さ、足首の軸（pitch のみ）、足の質量、滑りの判定のしきい値
  入力の不確かさ: 部品表の質量（上体・脚・荷物・腕 ±10%、足 ±15%）のモンテカルロ（1 層目の予測の幅）
"""
import os
os.environ.setdefault("MUJOCO_GL", "disable")
import copy
import json
import time

import matplotlib.pyplot as plt
import mujoco
import numpy as np

import field09_biped as f9
from common import DATA, FIG
from ambac.field.assembly import build, load
from ambac.field.evaluate import _min_jerk, _targets
from ambac.field.ground import pitch_yaw, roll

D = np.degrees
R = np.radians
OUT = FIG / "field"; OUT.mkdir(exist_ok=True)
T_START = time.time()
BIPED = load("robots", "biped_waist")
ANKLE = {"foot_size": [0.25, 0.3, 0.05], "foot_mass": 2.0, "height": 0.08, "axes": "pitch_roll", "armature": 0.1}
DT = 2.5e-4               # 時間刻み（計算誤差の表を参照。0.5 ms では滑りの量が +12%）
STRONG_KP = 10.0          # 予測（1 層目）の足首は、基準の 10 倍の硬さ
SLIDE_MM, TILT_DEG, ANKLE_DEG = 5.0, 1.0, 1.0
_gx, _gy = np.meshgrid(np.linspace(-0.125, 0.125, 401), np.linspace(-0.15, 0.15, 481))
RHO_UNIFORM = float(np.hypot(_gx, _gy).mean())       # 一様な圧力の足裏（0.25 × 0.3 m）がねじりを受け持つ有効半径
RHO_CORNER = float(np.hypot(0.125, 0.15))             # 4 隅の点だけで受け持つとき（MuJoCo の箱と床の接触）
S9 = json.load(open(DATA / "field09_biped.json"))


def robot_with(ankle=None, actuator=None, masses=None):
    r = copy.deepcopy(BIPED)
    r["ankle"] = {**ANKLE, **(ankle or {})}
    if actuator:
        r["actuator"].update(actuator)
    if masses:   # 部品表の質量を倍率で変える（モンテカルロ）
        r["base"]["mass"] *= masses.get("leg", 1.0)
        r["ankle"]["foot_mass"] *= masses.get("foot", 1.0)
        r["waist"]["torso_mass"] *= masses.get("torso", 1.0)
        r["arms"][0]["payload"] *= masses.get("payload", 1.0)
        for ln in r["arms"][0]["links"]:
            ln["mass"] *= masses.get("arm", 1.0)
    return r


def stage_with(name, mu=None):
    s = load("stages", name)
    if mu is not None:
        s["friction"] = mu
    return s


def patch_xml(xml, opt=None, solref=None):
    """MJCF の <option> と接触の柔らかさを差し替える（計算誤差・系統誤差の試験）"""
    if opt:
        for k, v in opt.items():
            import re
            if re.search(rf'{k}="[^"]*"', xml.split("<option", 1)[1].split("/>", 1)[0]):
                head, rest = xml.split("<option", 1); o, tail = rest.split("/>", 1)
                o = re.sub(rf'{k}="[^"]*"', f'{k}="{v}"', o); xml = head + "<option" + o + "/>" + tail
            else:
                xml = xml.replace("<option ", f'<option {k}="{v}" ', 1)
    if solref:
        xml = xml.replace('solref="0.004 1"', f'solref="{solref}"')
    return xml


class Motion:
    """段階 9 の動作（腰の静的なつり合い＋動的な補正、最短時間 T）。腰の角度の曲線は足首なしの機体で計算した値を使う"""

    def __init__(self, stage_name, strategy, T_scale=1.0):
        self.stage_name, self.strategy = stage_name, strategy
        self.prof = np.radians(S9[stage_name]["balance_profile_deg"])
        r = S9[stage_name][strategy]
        self.T = r["T"] * T_scale; self.coef = np.array(r["coef"])
        self.off = f9.make_offsets(self.prof)(strategy, self.coef)
        self.label = f"{stage_name}/{strategy}" + ("" if T_scale == 1 else f"/T×{T_scale:g}")


def run(robot, stage, mo, opt=None, solref=None, settle=0.5, hold=1.0, rec_every=2, slide_thr=SLIDE_MM):
    """動作を 1 回実行して、足・足首・床反力の時系列と、成功・失敗の判定を返す"""
    xml, joints = build(robot, stage)
    xml = patch_xml(xml, {"timestep": DT, **(opt or {})}, solref)
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    a, b = _targets(joints, "arm")
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in joints]
    act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in joints]
    ji = {j: i for i, j in enumerate(joints)}
    A = np.array([a.get(j, 0.0) for j in joints]); B = np.array([b.get(j, 0.0) for j in joints])

    def target(tau):
        q = A + (B - A) * _min_jerk(tau)
        for j, v in mo.off(tau).items():
            q[ji[j]] += v
        return q
    d.qpos[adr] = target(0.0); d.ctrl[act] = target(0.0); mujoco.mj_forward(m, d)
    foot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base"); leg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "leg")
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    ank = [ji[j] for j in joints if j.startswith("ankle")]
    Mtot = float(m.body_mass.sum()); g = np.array(m.opt.gravity)
    keys = ("t", "q", "Mo", "P", "foot_xyz", "foot_tilt", "leg_tilt", "F", "cop", "tau_ank", "com", "L_leg", "ncon", "ank_err", "yaw")
    rec = {k: [] for k in keys}
    n = int(round((settle + mo.T + hold) / m.opt.timestep)); c6 = np.zeros(6)
    for k in range(n + 1):
        t = d.time - settle
        d.ctrl[act] = target(np.clip(t / mo.T, 0, 1))
        if k % rec_every == 0:
            mujoco.mj_subtreeVel(m, d)
            F = np.zeros(3); Mo = np.zeros(3); nc = 0
            for i in range(d.ncon):
                c = d.contact[i]
                if floor not in (c.geom1, c.geom2):
                    continue
                mujoco.mj_contactForce(m, d, i, c6)
                fw = c.frame.reshape(3, 3).T @ c6[:3]          # 接触座標 → ワールド（geom1 が受ける力の反対向き）
                if c.geom1 != floor:      # MuJoCo の接触力は geom1 が geom2 に及ぼす力。床 → 足の向きにそろえる
                    fw = -fw
                F += fw; Mo += np.cross(c.pos, fw); nc += 1
            p, y = pitch_yaw(d.qpos); rl = roll(d.qpos)
            Rl = d.xmat[leg].reshape(3, 3); zl = Rl[:, 2]
            cop = np.array([-Mo[1] / F[2], Mo[0] / F[2]]) if F[2] > 1e-6 else np.array([np.nan, np.nan])   # 床の面（z=0）での足圧中心
            rec["t"].append(t); rec["Mo"].append(Mo); rec["P"].append(m.body_subtreemass[1] * d.subtree_linvel[1]); rec["q"].append(d.qpos[adr].copy()); rec["foot_xyz"].append(d.qpos[0:3].copy())
            rec["foot_tilt"].append(max(abs(p), abs(rl))); rec["leg_tilt"].append(np.arccos(np.clip(zl[2], -1, 1)))
            rec["F"].append(F); rec["cop"].append(cop - d.qpos[0:2]); rec["tau_ank"].append(d.actuator_force[[act[i] for i in ank]].copy())
            rec["com"].append(d.subtree_com[1].copy()); rec["ncon"].append(nc); rec["yaw"].append(y)
            Lc = d.subtree_angmom[leg].copy(); cl = d.subtree_com[leg].copy(); ml = m.body_subtreemass[leg]
            pa = d.xanchor[m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ankle_pitch")]].copy()
            vl = d.subtree_linvel[leg].copy()
            rec["L_leg"].append(np.concatenate([Lc + ml * np.cross(cl - pa, vl), cl - pa, [ml], Rl[:, 1], Rl[:, 0]]))
            rec["ank_err"].append(np.abs(d.qpos[[adr[i] for i in ank]] - target(np.clip(t / mo.T, 0, 1))[ank]).max())
        mujoco.mj_step(m, d)
        if d.qpos[2] < -0.5 or not np.isfinite(d.qpos).all():
            break
    rec = {k: np.array(v) for k, v in rec.items()}
    if rec["t"][-1] < 0:      # 動作の前に発散した
        return rec, dict(slide_mm=np.inf, slide_end_mm=np.inf, foot_tilt_deg=np.inf, leg_tilt_deg=np.inf, ankle_err_deg=np.inf,
                         arm_err_deg=np.inf, finished=False, yaw_deg=np.nan, Mtot=Mtot, g=g, slip=True, tip=True, ankle_yield=True, ok=False)
    mv = rec["t"] >= 0; i0 = np.argmin(abs(rec["t"]))
    slide = np.linalg.norm(rec["foot_xyz"][mv][:, :2] - rec["foot_xyz"][i0, :2], axis=1)
    arm = [ji[j] for j in joints if j.startswith("arm")]
    arm_err = D(np.abs(rec["q"][-1, arm] - target(1.0)[arm]).max())
    s = dict(slide_mm=float(slide.max() * 1000), slide_end_mm=float(slide[-1] * 1000),
             foot_tilt_deg=float(D(rec["foot_tilt"][mv].max())), leg_tilt_deg=float(D(rec["leg_tilt"][mv].max())),
             ankle_err_deg=float(D(rec["ank_err"][mv].max())), arm_err_deg=float(arm_err),
             finished=bool(rec["t"][-1] >= mo.T + hold - 0.01), yaw_deg=float(D(rec["yaw"][-1] - rec["yaw"][i0])),
             Mtot=Mtot, g=g)
    s["slip"] = s["slide_mm"] >= slide_thr
    s["tip"] = s["foot_tilt_deg"] >= TILT_DEG
    s["ankle_yield"] = s["ankle_err_deg"] >= ANKLE_DEG
    s["ok"] = s["finished"] and not (s["slip"] or s["tip"] or s["ankle_yield"]) and arm_err < 1.0
    return rec, s


def required(rec, s, mo):
    """1 層目: 必要な摩擦・足首のトルク・足圧中心（動作中 t ∈ [0, T + 0.5]）"""
    w = (rec["t"] >= 0) & (rec["t"] <= mo.T + 0.5)
    F = rec["F"][w]; Ft = np.linalg.norm(F[:, :2], axis=1); Fn = F[:, 2]
    mu = Ft / np.maximum(Fn, 1e-6)
    tau = np.abs(rec["tau_ank"][w])
    cop = rec["cop"][w]
    # ねじり（鉛直軸まわり）: 足圧中心まわりの床反力のモーメント Mz と、足裏の面で一様な圧力のときの有効半径 ρ
    copw = cop + rec["foot_xyz"][w][:, :2]
    Mz = rec["Mo"][w][:, 2] - (copw[:, 0] * F[:, 1] - copw[:, 1] * F[:, 0])
    mu_yaw = np.abs(Mz) / np.maximum(Fn, 1e-6) / RHO_UNIFORM
    mu_yaw_c = np.abs(Mz) / np.maximum(Fn, 1e-6) / RHO_CORNER
    # 並進とねじりを合わせた必要な摩擦（一様な圧力の足裏で、並進とねじりの両方を受け持つときの近似：二乗和）
    mu_comb = np.sqrt(mu**2 + mu_yaw**2)
    return dict(mu_req=float(mu.max()), t_mu=float(rec["t"][w][mu.argmax()]),
                mu_yaw_req=float(mu_yaw.max()), mu_yaw_req_corner=float(mu_yaw_c.max()), mu_comb_req=float(mu_comb.max()),
                Mz_max=float(np.abs(Mz).max()),
                tau_pitch=float(tau[:, 0].max()), tau_roll=float(tau[:, 1].max()) if tau.shape[1] > 1 else 0.0,
                cop_x=float(np.abs(cop[:, 0]).max()), cop_y=float(np.abs(cop[:, 1]).max()),
                cop_x_fwd=float(cop[:, 0].max()), cop_x_back=float(cop[:, 0].min()),
                Fn_min=float(Fn.min()), Fn_max=float(Fn.max()))


def newton_euler(rec, s, mo):
    """1 層目の内部の照合: 床反力の力積と全体の運動量の変化（ΔP = ∫(F + M g) dt）、
    足首のトルクの角力積と脚より上の角運動量の変化（足首まわり、ΔL = ∫(τ + r × m g) dt）を比べる。
    （加速度を数値微分すると接触の衝撃で雑音が大きいので、積分した形で比べる）"""
    t = rec["t"]; w = (t >= 0) & (t <= mo.T + 0.5); t = t[w]
    imp = np.concatenate([[np.zeros(3)], np.cumsum(0.5 * (rec["F"][w][1:] + rec["F"][w][:-1]) * np.diff(t)[:, None], 0)])
    dP = rec["P"][w] - rec["P"][w][0] - s["Mtot"] * s["g"] * (t - t[0])[:, None]
    eP = np.abs(imp - dP).max(0); sP = np.abs(dP - dP.mean(0)).max(0) + 1e-9
    L = rec["L_leg"][w]; Lp = L[:, 0:3]; r = L[:, 3:6]; ml = L[:, 6:7]; ey = L[:, 7:10]
    grav = np.cross(r, ml * s["g"])
    tau_y = rec["tau_ank"][w, 0][:, None] * ey                  # pitch のアクチュエータ（脚の y 軸まわり）
    tau_x = (rec["tau_ank"][w, 1][:, None] * L[:, 10:13]) if rec["tau_ank"].shape[1] > 1 else 0 * tau_y
    rhs = tau_y + tau_x + grav
    ang = np.concatenate([[np.zeros(3)], np.cumsum(0.5 * (rhs[1:] + rhs[:-1]) * np.diff(t)[:, None], 0)])
    dL = Lp - Lp[0]
    eL = np.abs(ang - dL).max(0); sL = np.abs(dL).max(0) + np.abs(np.cumsum(np.abs(rhs) * np.gradient(t)[:, None], 0)).max(0) * 0
    return dict(dP_rel_x=float(eP[0] / sP[0]), dP_rel_z=float(eP[2] / sP[2]),
                dL_rel_y=float(eL[1] / (np.abs(dL[:, 1]).max() + 1e-9)), dL_rel_x=float(eL[0] / (np.abs(dL[:, 0]).max() + 1e-9)))


def bisect(fun_ok, lo, hi, iters=9, log=False):
    """fun_ok(x) が lo で失敗・hi で成功になる境目（log=True なら対数で二分）"""
    for _ in range(iters):
        mid = np.sqrt(lo * hi) if log else (lo + hi) / 2
        if fun_ok(mid):
            hi = mid
        else:
            lo = mid
    return hi, lo


STRONG = {"kp": 3e4 * STRONG_KP, "kv": 1500 * np.sqrt(STRONG_KP)}
MOTIONS = [("practice", "static+dyn", 1.0), ("practice", "static+dyn", 1.25), ("practice", "static+dyn", 1.5),
           ("practice", "static+dyn", 2.0), ("practice", "static", 1.0), ("moon", "static+dyn", 1.0), ("moon", "static+dyn", 1.5)]


def inverse(mo, ankle_extra=None, masses=None, opt=None, win=0.05):
    """1 層目（逆動力学）: 足を床に固定した（溶接した）機体に動作をさせ、脚より上の運動量・角運動量の変化から
    床が足に及ぼすべき力・モーメントを求める。足が揺れたり浮いたりしないので、動作そのものが要求する量になる。
      F = dP/dt − M g、 M_o = dH_o/dt − Σ m c × g（ワールドの原点まわり）
      足圧中心（仮想の ZMP、Vukobratović）: x = −M_o,y / F_z、y = M_o,x / F_z
      ねじり: 足圧中心まわりの鉛直のモーメント M_z
      足首のトルク: アクチュエータの力（脚の pitch・roll 軸）"""
    if isinstance(mo, tuple):
        mo = Motion(*mo)
    robot = robot_with({**STRONG, **(ankle_extra or {})}, masses=masses)
    xml, joints = build(robot, stage_with(mo.stage_name, 2.0))
    xml = patch_xml(xml, {"timestep": DT, **(opt or {})}).replace('<freejoint name="base"/>', "")
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    a, b = _targets(joints, "arm")
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in joints]
    act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in joints]
    ji = {j: i for i, j in enumerate(joints)}
    A = np.array([a.get(j, 0.0) for j in joints]); B = np.array([b.get(j, 0.0) for j in joints])

    def target(tau):
        q = A + (B - A) * _min_jerk(tau)
        for j, v in mo.off(tau).items():
            q[ji[j]] += v
        return q
    d.qpos[adr] = target(0.0); d.ctrl[act] = target(0.0); mujoco.mj_forward(m, d)
    foot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base"); leg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "leg")
    ank = [act[ji[j]] for j in joints if j.startswith("ankle")]
    g = np.array(m.opt.gravity); mf = m.body_mass[foot]; cf = d.xipos[foot].copy(); ml = m.body_subtreemass[leg]
    t_, P_, H_, G_, tau_ = [], [], [], [], []
    n = int(round((0.3 + mo.T + 0.5) / m.opt.timestep))
    for k in range(n + 1):
        t = d.time - 0.3
        d.ctrl[act] = target(np.clip(t / mo.T, 0, 1))
        mujoco.mj_subtreeVel(m, d)
        cl = d.subtree_com[leg].copy(); vl = d.subtree_linvel[leg].copy()
        t_.append(t); P_.append(ml * vl); H_.append(d.subtree_angmom[leg] + ml * np.cross(cl, vl))
        G_.append(np.cross(ml * cl + mf * cf, g)); tau_.append(d.actuator_force[ank].copy())
        mujoco.mj_step(m, d)
    t = np.array(t_); P = np.array(P_); H = np.array(H_); G = np.array(G_); tau = np.abs(np.array(tau_))
    F = np.gradient(P, t, axis=0) - (ml + mf) * g
    Mo = np.gradient(H, t, axis=0) - G
    w = (t >= 0) & (t <= mo.T + 0.3)
    F, Mo, tau, t = F[w], Mo[w], tau[w], t[w]
    zx, zy = -Mo[:, 1] / F[:, 2], Mo[:, 0] / F[:, 2]
    Mz = Mo[:, 2] - (zx * F[:, 1] - zy * F[:, 0])
    Fn = F[:, 2]; mu = np.linalg.norm(F[:, :2], axis=1) / Fn
    k = max(1, int(round(win / m.opt.timestep)))
    smooth = lambda x: np.convolve(x, np.ones(k) / k, mode="same")
    return dict(mu_req=float(mu.max()), mu_yaw_req=float((np.abs(Mz) / Fn / RHO_UNIFORM).max()),
                mu_yaw_req_corner=float((np.abs(Mz) / Fn / RHO_CORNER).max()),
                mu_comb_req=float(np.sqrt(mu**2 + (Mz / Fn / RHO_UNIFORM)**2).max()), Mz_max=float(np.abs(Mz).max()),
                tau_pitch=float(tau[:, 0].max()), tau_roll=float(tau[:, 1].max()) if tau.shape[1] > 1 else 0.0,
                tau_pitch_avg=float(smooth(tau[:, 0]).max()), tau_roll_avg=float(smooth(tau[:, 1]).max()) if tau.shape[1] > 1 else 0.0,
                zmp_fwd=float(zx.max()), zmp_back=float(zx.min()), zmp_y=float(np.abs(zy).max()),
                L_pred=float(2 * max(zx.max(), -zx.min())), W_pred=float(2 * np.abs(zy).max()),
                Fn_min=float(Fn.min()), Fn_max=float(Fn.max()), series=dict(t=t, mu=mu, mu_yaw=np.abs(Mz) / Fn / RHO_UNIFORM, zx=zx, zy=zy, tau=tau))


def predict(mo, ankle_extra=None, masses=None, opt=None):
    """1 層目の予測（逆動力学、足を固定）と、摩擦・足首が十分に強い機体での床反力の照合（運動量・角運動量）
    mo: Motion か、(ステージ, 戦略, T の倍率) の組（並列計算で渡すため）"""
    if isinstance(mo, tuple):
        mo = Motion(*mo)
    out = inverse(mo, ankle_extra, masses, opt); out.pop("series")
    rec, s = run(robot_with({**STRONG, **(ankle_extra or {})}, masses=masses), stage_with(mo.stage_name, 2.0), mo, opt=opt)
    r = required(rec, s, mo)
    out.update({"ok_strong": s["ok"], "slide_strong_mm": s["slide_mm"], "yaw_strong_deg": s["yaw_deg"],
                "mu_req_free": r["mu_req"], "mu_yaw_free": r["mu_yaw_req"], "tau_pitch_free": r["tau_pitch"], "tau_roll_free": r["tau_roll"]})
    out.update(newton_euler(rec, s, mo))
    return out


def t_min(args):
    """足首つきの機体で、μ = 0.6 のまま成功する最短の T の倍率（二分法）"""
    st_name, strat, ank = args

    def ok(ts):
        return run(robot_with(ank), stage_with(st_name), Motion(st_name, strat, ts))[1]["ok"]
    return bisect(ok, 1.0, 1.25, iters=7) if ok(1.25) else (np.nan, np.nan)


def boundaries(args):
    """2 層目の実測：μ、足首のトルクの上限（pitch・roll）、足裏の長さ・幅の境目を二分法で求める"""
    st_name, strat, ts, opt, solref, ank, slide_thr = args
    mo = Motion(st_name, strat, ts)
    base = {**(ank or {})}

    def ok(stage_mu=0.6, extra=None, act=None):
        rb = robot_with({**base, **(extra or {})}, actuator=act)
        return run(rb, stage_with(st_name, stage_mu), mo, opt=opt, solref=solref, slide_thr=slide_thr)[1]["ok"]
    res = {"label": mo.label, "T": mo.T}
    res["ok_base"] = ok()
    strong = {**STRONG, **base}
    res["ok_strong_mu2"] = ok(2.0, strong)
    res["mu_star"] = bisect(lambda mu: ok(mu, strong), 0.02, 2.0, iters=9, log=True) if res["ok_strong_mu2"] else (np.nan, np.nan)
    for ax in ["pitch", "roll"]:
        f = lambda tl, ax=ax: ok(2.0, strong, act={"torque_limit": {f"ankle_{ax}": tl}})
        res[f"tau_{ax}_star"] = bisect(f, 2.0, 3000.0, iters=10, log=True) if res["ok_strong_mu2"] else (np.nan, np.nan)
    fs = ANKLE["foot_size"]
    fL = lambda L: ok(2.0, {**strong, "foot_size": [L, fs[1], fs[2]]})
    fW = lambda W: ok(2.0, {**strong, "foot_size": [fs[0], W, fs[2]]})
    res["L_star"] = bisect(fL, 0.04, 1.0, iters=9, log=True) if fL(1.0) else (np.nan, np.nan)
    res["W_star"] = bisect(fW, 0.04, 1.0, iters=9, log=True) if fW(1.0) else (np.nan, np.nan)
    return res


def main():
    from multiprocessing import Pool
    out = {"ankle": ANKLE, "dt": DT, "rho_uniform": RHO_UNIFORM, "rho_corner": RHO_CORNER, "motions": {}}
    print(f"足首つきの二本脚：足 {ANKLE['foot_mass']} kg、足裏 {ANKLE['foot_size'][:2]} m、足首の高さ {ANKLE['height']} m、"
          f"回転子の慣性 {ANKLE['armature']} kg·m²、時間刻み {DT * 1e3} ms", flush=True)
    print(f"ねじりの有効半径: 一様な圧力 {RHO_UNIFORM:.4f} m、4 隅の点 {RHO_CORNER:.4f} m（比 {RHO_CORNER / RHO_UNIFORM:.2f}）", flush=True)
    # ---- 1 層目の予測と 2 層目の実測 ----
    with Pool(4) as pool:
        meas = pool.map(boundaries, [(a, b, c, None, None, None, SLIDE_MM) for a, b, c in MOTIONS])
        preds = pool.starmap(predict, [((a, b, c),) for a, b, c in MOTIONS])
        tmins = pool.map(t_min, [("practice", "static+dyn", None), ("practice", "static+dyn", STRONG),
                                 ("moon", "static+dyn", None), ("practice", "static", None)])
    out["t_min"] = {}
    for (st_, sg_, an_), tm in zip([("practice", "static+dyn", "kp 3e4"), ("practice", "static+dyn", "kp 3e5"),
                                    ("moon", "static+dyn", "kp 3e4"), ("practice", "static", "kp 3e4")], tmins):
        T9 = S9[st_][sg_]["T"]
        out["t_min"][f"{st_}/{sg_}/{an_}"] = [T9 * tm[0], T9 * tm[1]]
        print(f"最短時間（足首つき、μ 0.6、足首 {an_}）{st_}/{sg_}: {T9 * tm[0]:.3f} s（段階 9 の足首なし {T9:.3f} s の {tm[0]:.3f} 倍）", flush=True)
    for (a, b, c), pr, me in zip(MOTIONS, preds, meas):
        out["motions"][me["label"]] = {"pred": pr, "meas": me}
        mg = 97.0 * load("stages", a)["gravity"]
        print(f"=== {me['label']}  T = {me['T']:.3f} s  基準（μ 0.6、足首 kp 3e4）{'成功' if me['ok_base'] else '失敗'}"
              f"  強い足首・μ 2 で {'成功' if me['ok_strong_mu2'] else '失敗'} ===", flush=True)
        print(f"  滑り   : 予測 並進 {pr['mu_req']:.3f}、ねじり（一様）{pr['mu_yaw_req']:.3f}（4 隅 {pr['mu_yaw_req_corner']:.3f}）、"
              f"合成 {pr['mu_comb_req']:.3f}  |  実測 μ* = {me['mu_star'][0]:.3f}", flush=True)
        print(f"  足首   : 予測 pitch {pr['tau_pitch']:.1f} / roll {pr['tau_roll']:.1f} N·m（50 ms 平均の最大 {pr['tau_pitch_avg']:.1f} / {pr['tau_roll_avg']:.1f}）"
              f"  |  実測 τ* pitch {me['tau_pitch_star'][0]:.1f} / roll {me['tau_roll_star'][0]:.1f} N·m"
              f"（Mg × 足裏の半分 = {mg * 0.125:.0f} / {mg * 0.15:.0f}）", flush=True)
        print(f"  転倒   : 予測 足圧中心 前 {pr['zmp_fwd']:+.3f} 後 {pr['zmp_back']:+.3f} 横 ±{pr['zmp_y']:.3f} m → L ≥ {pr['L_pred']:.3f}、W ≥ {pr['W_pred']:.3f}"
              f"  |  実測 L* = {me['L_star'][0]:.3f}、W* = {me['W_star'][0]:.3f} m", flush=True)
        print(f"  照合   : 運動量 x {pr['dP_rel_x']:.1e}、z {pr['dP_rel_z']:.1e}、足首の角運動量 y {pr['dL_rel_y']:.1e}、x {pr['dL_rel_x']:.1e}"
              f"  [{time.time() - T_START:.0f} s]", flush=True)
    json.dump(out, open(DATA / "field19_ankle.json", "w"), indent=1, ensure_ascii=False, default=float)



def figure():
    """図: 必要な摩擦・足首のトルク・足圧中心と、境目の T 依存（data/field19_ankle.json）"""
    res = json.load(open(DATA / "field19_ankle.json"))["motions"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.4))
    for ts, col in [(1.0, "tab:red"), (1.5, "tab:blue")]:
        mo = Motion("practice", "static+dyn", ts)
        rec, s = run(robot_with(STRONG), stage_with("practice", 2.0), mo)
        t = rec["t"] / mo.T; F = rec["F"]; Fn = np.maximum(F[:, 2], 1e-6)
        mu = np.linalg.norm(F[:, :2], axis=1) / Fn
        copw = rec["cop"] + rec["foot_xyz"][:, :2]
        Mz = rec["Mo"][:, 2] - (copw[:, 0] * F[:, 1] - copw[:, 1] * F[:, 0])
        ax[0, 0].plot(t, mu, color=col, label=f"T×{ts:g}: translation |Ft|/Fn")
        ax[0, 0].plot(t, np.abs(Mz) / Fn / RHO_UNIFORM, color=col, ls="--", label=f"T×{ts:g}: twist |Mz|/(Fn ρ)")
        ax[0, 1].plot(t, rec["tau_ank"][:, 0], color=col, label=f"T×{ts:g}: pitch")
        ax[0, 1].plot(t, rec["tau_ank"][:, 1], color=col, ls="--", label=f"T×{ts:g}: roll")
        sr = inverse(mo)["series"]
        ax[1, 0].plot(sr["zx"], sr["zy"], color=col, lw=1, label=f"T×{ts:g}: fictitious ZMP (foot fixed)")
    ax[0, 0].axhline(0.6, color="k", lw=0.8, ls=":"); ax[0, 0].text(1.6, 0.62, "μ = 0.6 (practice)", fontsize=8)
    ax[0, 0].set_ylim(0, 2.2); ax[0, 0].set_title("required friction (strong ankle, μ = 2)"); ax[0, 0].legend(fontsize=7)
    ax[0, 1].set_title("ankle torque [N·m]"); ax[0, 1].legend(fontsize=7)
    for a_ in ax[0]:
        a_.set_xlabel("t / T"); a_.axvspan(0, 1, color="0.93", zorder=0); a_.set_xlim(-0.2, 2.0)
    ax[1, 0].add_patch(plt.Rectangle((-0.125, -0.15), 0.25, 0.3, fill=False, color="k", lw=1.5))
    ax[1, 0].text(-0.12, 0.16, "foot 0.25 × 0.3 m", fontsize=8)
    ax[1, 0].set_aspect("equal"); ax[1, 0].set_xlabel("x (forward) [m]"); ax[1, 0].set_ylabel("y (left) [m]")
    ax[1, 0].set_title("ZMP / centre of pressure relative to ankle"); ax[1, 0].legend(fontsize=7)
    sc = [1.0, 1.25, 1.5, 2.0]
    lab = lambda x: "practice/static+dyn" + ("" if x == 1 else f"/T×{x:g}")
    g = lambda k, x, part: res[lab(x)][part][k]
    ax[1, 1].semilogy(sc, [g("mu_req", x, "pred") for x in sc], "o--", color="tab:gray", label="pred: translation")
    ax[1, 1].semilogy(sc, [g("mu_yaw_req", x, "pred") for x in sc], "s--", color="tab:orange", label="pred: twist (uniform pressure)")
    ax[1, 1].semilogy(sc, [g("mu_yaw_req_corner", x, "pred") for x in sc], "^--", color="tab:green", label="pred: twist (4 corners)")
    ax[1, 1].semilogy(sc, [g("mu_star", x, "meas")[0] for x in sc], "k*-", ms=10, label="MuJoCo: μ* (slip < 5 mm)")
    ax[1, 1].axhline(0.6, color="k", lw=0.8, ls=":")
    ax[1, 1].set_xlabel("T / T_stage9"); ax[1, 1].set_title("friction boundary vs motion time (practice)"); ax[1, 1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "field19_ankle.png", dpi=130)
    print("図 → figures/field/field19_ankle.png")


# ---- 誤差の評価（python field19_ankle.py errors）----------------------------------------------
REF = ("practice", "static+dyn", 1.25)
VARIANTS = {   # 名前: (opt, solref, 足首の設定, 滑りのしきい値 mm)
    "基準": (None, None, None, SLIDE_MM),
    "計算: 時間刻み 0.5 ms": ({"timestep": 5e-4}, None, None, SLIDE_MM),
    "計算: 時間刻み 0.1 ms": ({"timestep": 1e-4}, None, None, SLIDE_MM),
    "計算: 多角形の摩擦円錐": ({"cone": "pyramidal"}, None, None, SLIDE_MM),
    "計算: impratio 1": ({"impratio": 1}, None, None, SLIDE_MM),
    "系統: 足裏を 4 隅の点に": (None, None, {"contact": "corners"}, SLIDE_MM),
    "系統: 接触を柔らかく（solref 0.02）": (None, "0.02 1", None, SLIDE_MM),
    "系統: 足首 pitch のみ（roll は固定）": (None, None, {"axes": "pitch"}, SLIDE_MM),
    "系統: 足首のサーボ kp 3e4（基準の 1/10）": (None, None, {"kp": 3e4, "kv": 1500}, SLIDE_MM),
    "系統: 足首のサーボ kp 3e3": (None, None, {"kp": 3e3, "kv": 475}, SLIDE_MM),
    "系統: 回転子の慣性 0.01 kg·m²": (None, None, {"armature": 0.01}, SLIDE_MM),
    "系統: 回転子の慣性 1 kg·m²": (None, None, {"armature": 1.0}, SLIDE_MM),
    "系統: 足の質量 4 kg": (None, None, {"foot_mass": 4.0}, SLIDE_MM),
    "系統: 足首の高さ 0.12 m": (None, None, {"height": 0.12}, SLIDE_MM),
    "系統: 滑りのしきい値 1 mm": (None, None, None, 1.0),
}


def mc_one(seed):
    rng = np.random.default_rng(seed)
    ms = {"torso": 1 + rng.uniform(-0.1, 0.1), "leg": 1 + rng.uniform(-0.1, 0.1), "payload": 1 + rng.uniform(-0.1, 0.1),
          "arm": 1 + rng.uniform(-0.1, 0.1), "foot": 1 + rng.uniform(-0.15, 0.15)}
    mo = Motion(*REF)
    pr = predict(mo, masses=ms)
    ok = lambda mu: run(robot_with(STRONG, masses=ms), stage_with(REF[0], mu), mo)[1]["ok"]
    okt = lambda tl: run(robot_with(STRONG, actuator={"torque_limit": {"ankle_pitch": tl}}, masses=ms), stage_with(REF[0], 2.0), mo)[1]["ok"]
    pr["mu_star"] = bisect(ok, 0.02, 2.0, iters=8, log=True)[0]
    pr["tau_pitch_star"] = bisect(okt, 2.0, 3000.0, iters=9, log=True)[0]
    return ms, pr


def errors_main():
    from multiprocessing import Pool
    res = {}
    names = list(VARIANTS)
    with Pool(4) as pool:
        meas = pool.map(boundaries, [(*REF, *VARIANTS[k]) for k in names])
        preds = pool.starmap(predict, [(REF, VARIANTS[k][2], None, VARIANTS[k][0]) for k in names])
        mc = pool.map(mc_one, range(40))
    print(f"=== 誤差の評価：{Motion(*REF).label}（T = {Motion(*REF).T:.3f} s）===")
    print("| 条件 | 予測 μ（並進 / ねじり一様 / 4 隅） | 実測 μ* | 予測 τ pitch / roll | 実測 τ* pitch / roll | 予測 L / W | 実測 L* / W* |")
    for k, pr, me in zip(names, preds, meas):
        res[k] = {"pred": pr, "meas": me}
        tr = "—" if VARIANTS[k][2] and VARIANTS[k][2].get("axes") == "pitch" else f"{me['tau_roll_star'][0]:.1f}"
        print(f"| {k} | {pr['mu_req']:.3f} / {pr['mu_yaw_req']:.3f} / {pr['mu_yaw_req_corner']:.3f} | {me['mu_star'][0]:.3f} | "
              f"{pr['tau_pitch']:.1f} / {pr['tau_roll']:.1f} | {me['tau_pitch_star'][0]:.1f} / {tr} | "
              f"{pr['L_pred']:.3f} / {pr['W_pred']:.3f} | {me['L_star'][0]:.3f} / {me['W_star'][0]:.3f} |", flush=True)
    keys = ["mu_req", "mu_yaw_req", "mu_star", "tau_pitch", "tau_roll", "tau_pitch_star", "L_pred", "W_pred"]
    arr = {k: np.array([p[k] for _, p in mc]) for k in keys}
    print("=== 入力（部品表の質量）の不確かさ：40 回 ===")
    for k in keys:
        print(f"  {k:15s}: {arr[k].mean():.4f} ± {arr[k].std(ddof=1):.4f}（{100 * arr[k].std(ddof=1) / arr[k].mean():.1f}%）")
    res["mc"] = {k: [float(x) for x in v] for k, v in arr.items()}
    res["mc_masses"] = [m for m, _ in mc]
    json.dump(res, open(DATA / "field19_errors.json", "w"), indent=1, ensure_ascii=False, default=float)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "errors":
        errors_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "figure":
        figure()
    else:
        main()
