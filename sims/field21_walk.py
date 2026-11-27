"""Field21: 二本脚を MuJoCo で歩かせる（下半身の段 4）.

段階 20 では、重心を支持する足の上に置くだけの「準静的な計画」は 1 相 2 s より速いと倒れた。
ここでは、歩行の標準的な方法である
  線形倒立振子（LIPM）＋ ZMP のプレビュー制御（Kajita ら 2003）
で重心の動きを作り、3 次元の二本脚（左右の倒れも自由）を歩かせる。

機体（段階 20 と同じ寸法・質量、97 kg）: 上体 56 kg、大腿 16.5 kg・0.45 m、下腿 3 kg・0.45 m、足 1 kg（かかと 0.10、つま先 0.15、幅 0.10 m）。
  脚の関節: 股関節 roll・pitch、膝、足首 pitch・roll（片脚 5 自由度、ひねりなし＝まっすぐ歩く）。位置サーボ、トルクの上限 250 s³ N·m。
  骨盤（上体）は自由関節（6 自由度、拘束なし）。
計画
  (A) 準静的: 重心を、片脚支持の間は支持する足の中央の真上に置き、両脚支持の間に移す（段階 20 の考え方を歩行に）
  (B) ZMP: 足の置き場所から ZMP の目標を作り、LIPM のプレビュー制御で重心の動きを求める
  どちらも、重心の目標に合わせて骨盤の位置を逆運動学で決め、関節角を位置サーボの目標にする（開ループ）。
  重力の補償: 準静的なトルク（支持する足の床反力を最小ノルムで解く）を目標角のずれとして足す（段階 20 と同じ）。
照合の層
  1 層目: LIPM の ZMP（計画）、2 層目: MuJoCo の床反力から求めた足圧中心（実測）
評価: 1 歩の時間 T_step と歩幅 L を変えて、8 歩を転ばずに歩けるか（上体の傾き 15° 未満、最後の位置の誤差 10 cm 未満、足の滑り）
"""
import os
os.environ.setdefault("MUJOCO_GL", "disable")
import json
import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

from common import DATA, FIG, smooth_step

D = np.degrees
R = np.radians
G = 9.80665
OUT = FIG / "field"; OUT.mkdir(exist_ok=True)
T_START = time.time()

L_TH, L_SH = 0.45, 0.45
M_TRUNK, M_TH, M_SH, M_FOOT = 56.0, 16.5, 3.0, 1.0
TRUNK = (0.25, 0.4, 0.6); TRUNK_COM = 0.25
HEEL, TOE, FOOT_W, FOOT_H, ANK_H = 0.10, 0.15, 0.10, 0.04, 0.06
HIP_Y = 0.10
TAU = 250.0
ARM_M, ARM_L, ARM_Y, ARM_Z, ARM_C = 4.0, 0.60, 0.22, 0.50, 0.45   # 段階 22 の腕: 片腕の質量・長さ、肩の位置（骨盤から）、重心の位置（肩から長さの割合）
RANGES = {"hr": (-30, 30), "hp": (-30, 120), "kn": (-1, 140), "ap": (-50, 50), "ar": (-30, 30)}
SEG = ("hr", "hp", "kn", "ap", "ar")
JOINTS = [f"{j}{sd}" for sd in "LR" for j in SEG]


@dataclass
class Cfg:
    s: float = 1.0
    planner: str = "zmp"              # "zmp" / "static"
    T_step: float = 0.8               # 1 歩の時間（s = 1。実際は × √s）
    ds: float = 0.2                   # 両脚支持の割合
    L: float = 0.25                   # 歩幅（s = 1）
    n_steps: int = 8
    z_hip: float = 0.85               # 骨盤（股関節）の高さ（足首から、s = 1。脚の長さ 0.9）
    lift: float = 0.05                # 振り出す足の高さ
    mu: float = 0.8
    kp: float = 3e4
    tau: float = TAU
    dt: float = 5e-4
    integrator: str = "implicitfast"
    armature: float = 0.05
    solref: str = "0.004 1"
    preview: float = 1.6              # プレビューの時間（s）
    ff: bool = True                   # 重力の補償
    extra: dict = field(default_factory=dict)


def build_xml(c: Cfg):
    s = c.s; m3 = s**3; tau = c.tau * m3
    ms = c.extra.get("mass", {})
    mt, mth, msh, mf = M_TRUNK * ms.get("trunk", 1), M_TH * ms.get("thigh", 1), M_SH * ms.get("shank", 1), M_FOOT * ms.get("foot", 1)
    ib = lambda m, a, b, cc: f"{m * (b*b + cc*cc) / 12:.6g} {m * (a*a + cc*cc) / 12:.6g} {m * (a*a + b*b) / 12:.6g}"
    rod = lambda m, L: f"{m * L * L / 12:.6g} {m * L * L / 12:.6g} {m * (0.06*s)**2 / 2:.6g}"
    tx, ty, tz = [v * s for v in TRUNK]
    lth, lsh = L_TH * s, L_SH * s
    fx0, fx1 = -HEEL * s, TOE * s; fl = fx1 - fx0; fh = FOOT_H * s; ah = ANK_H * s; fw = FOOT_W * s
    kp = c.kp * m3; kv = kp * 0.05 * np.sqrt(s); arm = c.armature * s**5
    jr = lambda k: f'range="{RANGES[k][0]} {RANGES[k][1]}" limited="true" armature="{arm}"'
    mid = '<inertial pos="0 0 0" mass="1e-3" diaginertia="1e-6 1e-6 1e-6"/>'
    legs = ""
    ar_ = c.extra.get("arms")          # 段階 22: 腕（肩の前後の振り）。上体 56 kg の中から腕の分を引く
    acts_arm = ""
    if ar_:
        ma, La = ar_.get("m", ARM_M) * m3, ar_.get("L", ARM_L) * s
        mt = mt - 2 * ar_.get("m", ARM_M)
        kpa = ar_.get("kp", 3e3) * m3; kva = kpa * 0.05 * np.sqrt(s)
        for sd, y in [("L", ARM_Y * s), ("R", -ARM_Y * s)]:
            legs += f"""
      <body name="arm{sd}" pos="0 {y} {ARM_Z * s}">
        <joint name="sh{sd}" axis="0 -1 0" range="-90 90" limited="true" armature="{arm}"/>
        <inertial pos="0 0 {-ARM_C * La}" mass="{ma}" diaginertia="{ma * La * La / 12 + ma * (ARM_C - 0.5)**2 * La * La:.6g} {ma * La * La / 12 + ma * (ARM_C - 0.5)**2 * La * La:.6g} {ma * (0.05 * s)**2 / 2:.6g}"/>
        <geom type="capsule" fromto="0 0 0 0 0 {-La}" size="{0.04 * s}" contype="0" conaffinity="0" rgba="0.9 0.6 0.3 1"/>
      </body>"""
            acts_arm += f'<position name="sh{sd}" joint="sh{sd}" kp="{kpa}" kv="{kva}" forcelimited="true" forcerange="{-tau} {tau}"/>'
    for sd, y in [("L", HIP_Y * s), ("R", -HIP_Y * s)]:
        legs += f"""
      <body name="hipr{sd}" pos="0 {y} 0">
        <joint name="hr{sd}" axis="1 0 0" {jr('hr')}/>{mid}
        <body name="thigh{sd}">
          <joint name="hp{sd}" axis="0 -1 0" {jr('hp')}/>
          <inertial pos="0 0 {-lth/2}" mass="{mth*m3}" diaginertia="{rod(mth*m3, lth)}"/>
          <geom type="capsule" fromto="0 0 0 0 0 {-lth}" size="{0.06*s}" contype="0" conaffinity="0" rgba="0.3 0.5 0.9 1"/>
          <body name="shank{sd}" pos="0 0 {-lth}">
            <joint name="kn{sd}" axis="0 1 0" {jr('kn')}/>
            <inertial pos="0 0 {-lsh/2}" mass="{msh*m3}" diaginertia="{rod(msh*m3, lsh)}"/>
            <geom type="capsule" fromto="0 0 0 0 0 {-lsh}" size="{0.045*s}" contype="0" conaffinity="0" rgba="0.3 0.5 0.9 1"/>
            <body name="ankp{sd}" pos="0 0 {-lsh}">
              <joint name="ap{sd}" axis="0 -1 0" {jr('ap')}/>{mid}
              <body name="foot{sd}">
                <joint name="ar{sd}" axis="1 0 0" {jr('ar')}/>
                <site name="ank{sd}" pos="0 0 0" size="0.01"/>
                <inertial pos="{(fx0+fx1)/2} 0 {-ah + fh/2}" mass="{mf*m3}" diaginertia="{ib(mf*m3, fl, fw, fh)}"/>
                <geom name="sole{sd}" type="box" pos="{(fx0+fx1)/2} 0 {-ah + fh/2}" size="{fl/2} {fw/2} {fh/2}" friction="{c.mu} 0.005 0.0001" rgba="0.4 0.4 0.45 1"/>
              </body>
            </body>
          </body>
        </body>
      </body>"""
    acts = "".join(f'<position name="{j}" joint="{j}" kp="{kp}" kv="{kv}" forcelimited="true" forcerange="{-tau} {tau}"/>' for j in JOINTS) + acts_arm
    z0 = ah + c.z_hip * s
    return f"""<mujoco model="biped_walk">
  <option gravity="0 0 {-G}" timestep="{c.dt}" integrator="{c.integrator}" cone="elliptic" impratio="10"/>
  <default><geom solref="{c.solref}" solimp="0.95 0.99 0.001"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="{20*s} {20*s} 0.1" friction="{c.mu} 0.005 0.0001" rgba="0.85 0.85 0.8 1"/>
    <body name="trunk" pos="0 0 {z0}">
      <freejoint name="root"/>
      <inertial pos="0 0 {TRUNK_COM*s}" mass="{mt*m3}" diaginertia="{ib(mt*m3, tx, ty, tz)}"/>
      <geom type="box" pos="0 0 {tz/2}" size="{tx/2} {ty/2} {tz/2}" contype="0" conaffinity="0" rgba="0.65 0.65 0.7 1"/>
      {legs}
    </body>
  </worldbody>
  <actuator>{acts}</actuator>
</mujoco>"""


class Robot:
    def __init__(self, c: Cfg):
        self.c = c
        self.m = mujoco.MjModel.from_xml_string(build_xml(c)); self.d = mujoco.MjData(self.m)
        m = self.m
        jid = lambda j: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        self.qa = {j: m.jnt_qposadr[jid(j)] for j in JOINTS}
        self.va = {j: m.jnt_dofadr[jid(j)] for j in JOINTS}
        self.site = {sd: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"ank{sd}") for sd in "LR"}
        self.foot = {sd: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"foot{sd}") for sd in "LR"}
        self.lim = {f"{k}{sd}": np.radians(RANGES[k]) for sd in "LR" for k in SEG}
        self.arm = {sd: m.jnt_qposadr[jid(f"sh{sd}")] for sd in "LR"} if c.extra.get("arms") else {}
        self.z0 = m.body_pos[1][2]
        q = self.d.qpos
        for sd in "LR":
            q[self.qa[f"hp{sd}"]], q[self.qa[f"kn{sd}"]], q[self.qa[f"ap{sd}"]] = R(15), R(30), R(15)

    def ik(self, root_xyz, feet, q0, iters=30):
        """骨盤の位置（姿勢は水平・正面）と、足首の位置・足の向き（水平）から関節角を求める"""
        m, d = self.m, self.d
        q = q0.copy(); q[0:3] = root_xyz; q[3:7] = [1, 0, 0, 0]
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
        err = 1.0
        for _ in range(iters):
            d.qpos[:] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            err = 0.0
            for sd in "LR":
                p = d.site_xpos[self.site[sd]]; Rm = d.xmat[self.foot[sd]].reshape(3, 3)
                e = np.concatenate([np.asarray(feet[sd]) - p, 0.5 * np.cross(Rm[:, 0], [1, 0, 0]) + 0.5 * np.cross(Rm[:, 1], [0, 1, 0]) + 0.5 * np.cross(Rm[:, 2], [0, 0, 1])])
                err = max(err, np.abs(e[:3]).max())
                mujoco.mj_jacSite(m, d, jacp, jacr, self.site[sd])
                cols = [self.va[f"{k}{sd}"] for k in SEG]
                J = np.vstack([jacp[:, cols], jacr[:2, cols]])
                ee = np.concatenate([e[:3], e[3:5]])
                dq = np.linalg.solve(J.T @ J + 1e-8 * np.eye(5), J.T @ ee)
                for i, k in enumerate(SEG):
                    a = self.qa[f"{k}{sd}"]
                    q[a] = np.clip(q[a] + np.clip(dq[i], -0.3, 0.3), *self.lim[f"{k}{sd}"])
            if err < 1e-8:
                break
        ok = err < 1e-4 * self.c.s
        return q, ok

    def com(self, q):
        self.d.qpos[:] = q; mujoco.mj_kinematics(self.m, self.d); mujoco.mj_comPos(self.m, self.d)
        return self.d.subtree_com[1].copy()

    def pelvis_for_com(self, com_xy, z, feet, q0, xy0):
        """重心の水平位置が com_xy になる骨盤の水平位置（固定点反復）"""
        xy = np.array(xy0, float); q = q0
        for _ in range(8):
            q, ok = self.ik((xy[0], xy[1], z), feet, q)
            e = com_xy - self.com(q)[:2]
            xy = xy + e
            if np.abs(e).max() < 1e-6 * self.c.s:
                break
        q, ok = self.ik((xy[0], xy[1], z), feet, q)
        return q, ok

    def static_torques(self, q, support):
        """準静的なトルク: 支持する足の床反力（各 6 成分）を未知数に、骨盤の 6 自由度の力が 0 になる最小ノルム解"""
        m, d = self.m, self.d
        d.qpos[:] = q; d.qvel[:] = 0; mujoco.mj_forward(m, d)
        gq = d.qfrc_bias.copy()
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
        Js = []
        for sd in support:
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site[sd]); Js.append(np.vstack([jacp, jacr]))
        Jc = np.vstack(Js)
        f = np.linalg.lstsq(Jc[:, :6].T, gq[:6], rcond=None)[0]
        tau = gq - Jc.T @ f
        return np.array([tau[self.va[j]] for j in JOINTS])


# ---- 足の置き場所と ZMP の目標 -------------------------------------------------------------
def footsteps(c: Cfg):
    """足の置き場所の列（時刻、支持の状態）。最初に 1 s 両脚で立ち、右足から踏み出す。最後の 1 歩で両足をそろえる"""
    s = c.s; Ts = c.T_step * np.sqrt(s); L = c.L * s; y = HIP_Y * s; ah = ANK_H * s
    steps = []          # (t0, t1, swing_side, from_xy, to_xy)
    pos = {"L": np.array([0.0, y]), "R": np.array([0.0, -y])}
    t = 1.0 * np.sqrt(s)
    for k in range(c.n_steps):
        sw = "R" if k % 2 == 0 else "L"
        last = k == c.n_steps - 1
        other = pos["L" if sw == "R" else "R"]
        tgt_x = other[0] + (L if (k > 0 and not last) else (L / 2 if k == 0 else 0.0))
        if last:
            tgt_x = other[0]
        new = np.array([tgt_x, pos[sw][1]])
        so = c.extra.get("step_off")       # 段階 24: k 歩目の足を置く場所を d だけずらす（以後の足の前後の位置も続けてずれる）
        if so and k in so:
            new = new + np.asarray(so[k]) * s
        steps.append((t, t + Ts, sw, pos[sw].copy(), new))
        pos[sw] = new; t += Ts
    return steps, t + 1.5 * np.sqrt(s), ah


def feet_at(t, c, steps, ah):
    """時刻 t の両足首の位置と、支持の状態"""
    s = c.s; Tds = c.ds * c.T_step * np.sqrt(s)
    pos = {"L": np.array([0.0, HIP_Y * s, ah]), "R": np.array([0.0, -HIP_Y * s, ah])}
    support = "LR"
    for (t0, t1, sw, a, b) in steps:
        if t >= t1:
            pos[sw] = np.array([b[0], b[1], ah])
        elif t > t0:
            # 両脚支持（前半 ds/2）→ 片脚支持（振り出し）→ 両脚支持（後半 ds/2）
            ta, tb = t0 + Tds / 2, t1 - Tds / 2
            if ta < t < tb:
                u = (t - ta) / (tb - ta); v = smooth_step(u)[0]
                xy = a + (b - a) * v
                z = ah + c.lift * s * np.sin(np.pi * u)
                pos[sw] = np.array([xy[0], xy[1], z])
                support = "L" if sw == "R" else "R"
            else:
                pos[sw] = np.array([a[0], a[1], ah]) if t <= ta else np.array([b[0], b[1], ah])
            break
    return pos, support


def arm_angles(t, c, steps, ah):
    """段階 22: 腕を脚と逆に振る。右腕の前への角 = A × (左足 − 右足の前後の差) / 歩幅、左腕はその逆"""
    ar_ = c.extra.get("arms")
    if not ar_:
        return {}
    pos, _ = feet_at(t, c, steps, ah)
    u = (pos["L"][0] - pos["R"][0]) / (c.L * c.s)
    A = np.radians(ar_.get("A", 0.0))
    return {"R": A * u, "L": -A * u}


def zmp_ref(t, c, steps):
    """ZMP の目標: 片脚支持では支持する足の中央（足首の +0.025 s 前）、両脚支持の間に次の足へ直線で移す"""
    s = c.s; Tds = c.ds * c.T_step * np.sqrt(s); xc = (TOE - HEEL) / 2 * s
    cur = np.array([xc, 0.0])                       # 最初は両足の中央
    for k, (t0, t1, sw, a, b) in enumerate(steps):
        stance = steps[k - 1][4] if k > 0 else np.array([0.0, (HIP_Y if sw == "R" else -HIP_Y) * s])
        stance = np.array([stance[0] + xc, stance[1]])
        if t < t0:
            return cur
        if t < t0 + Tds / 2 + 1e-9:
            u = (t - t0) / (Tds / 2) if Tds > 0 else 1.0
            return cur + (stance - cur) * smooth_step(min(u, 1))[0]
        if t < t1 - Tds / 2:
            return stance
        if t < t1:
            nxt = np.array([b[0] + xc, b[1]])
            u = (t - (t1 - Tds / 2)) / (Tds / 2)
            if k == len(steps) - 1:
                nxt = np.array([b[0] + xc, 0.0])
            return stance + (nxt - stance) * smooth_step(min(u, 1))[0] * (1.0 if k < len(steps) - 1 else 1.0)
        cur = np.array([b[0] + xc, b[1]]) if k < len(steps) - 1 else np.array([b[0] + xc, 0.0])
    return cur


def preview_com(zref, dt, zc, preview, i0=0, x0=None):
    """LIPM の ZMP プレビュー制御（Kajita 2003）: zref (N×2) の ZMP を実現する重心の軌道"""
    A = np.array([[1, dt, dt * dt / 2], [0, 1, dt], [0, 0, 1]]); B = np.array([[dt**3 / 6], [dt * dt / 2], [dt]])
    C = np.array([[1, 0, -zc / G]])
    # 誤差の積分を入れた拡大系
    Ai = np.block([[np.eye(1), C @ A], [np.zeros((3, 1)), A]]); Bi = np.vstack([C @ B, B])
    Q = np.diag([1.0, 0, 0, 0]); Rr = np.array([[1e-6]])
    P = solve_discrete_are(Ai, Bi, Q, Rr)
    K = np.linalg.solve(Rr + Bi.T @ P @ Bi, Bi.T @ P @ Ai)
    Ke, Kx = K[0, 0], K[0, 1:]
    Np = int(round(preview / dt))
    Ac = Ai - Bi @ K; Xi = -Ac.T @ P @ np.array([[1], [0], [0], [0]])
    f = np.zeros(Np)
    for i in range(Np):
        f[i] = (np.linalg.solve(Rr + Bi.T @ P @ Bi, Bi.T @ Xi))[0, 0]
        Xi = Ac.T @ Xi
    N = len(zref); zpad = np.vstack([zref, np.repeat(zref[-1:], Np, 0)])
    out = np.zeros((N, 2)); zmp = np.zeros((N, 2))
    for ax in range(2):
        x = np.array([zref[0, ax], 0, 0]); e = 0.0
        for k in range(N):
            if x0 is not None and k == i0:        # 段階 24: 測った重心の位置・速度から計画し直す
                x = np.array([x0[0][ax], x0[1][ax], 0.0])
            out[k, ax] = x[0]; zmp[k, ax] = (C @ x)[0]
            e += zmp[k, ax] - zref[k, ax]
            u = -Ke * e - Kx @ x - f @ zpad[k + 1:k + 1 + Np, ax]
            x = A @ x + B[:, 0] * u
    return out, zmp


def plan(c: Cfg, dt_plan=0.01):
    rb = Robot(c); s = c.s
    steps, T_end, ah = footsteps(c)
    t = np.arange(0, T_end, dt_plan)
    zr = np.array([zmp_ref(tt, c, steps) for tt in t])
    feet0, _ = feet_at(0.0, c, steps, ah)
    z_hip = ah + c.z_hip * s
    q00 = rb.d.qpos.copy()
    for sd, a_ in arm_angles(0.0, c, steps, ah).items():
        q00[rb.arm[sd]] = a_
    q0, _ = rb.ik((0, 0, z_hip), feet0, q00)
    zc = rb.com(q0)[2]
    if c.planner == "zmp":
        rf = c.extra.get("replan_from")
        com_xy, zmp_lipm = preview_com(zr, dt_plan, zc, c.preview * np.sqrt(s),
                                       *((rf["i"], (np.asarray(rf["x"]), np.asarray(rf["v"]))) if rf else (0, None)))
    else:   # 準静的: 重心 = ZMP の目標（重心をそのまま支持する足の上へ）
        com_xy, zmp_lipm = zr.copy(), zr.copy()
    Q = []; ok_all = True; sup = []; tauff = []; Sens = []
    q = q0.copy(); xy = np.zeros(2)
    Lr = 0.98 * (L_TH + L_SH) * s
    for k, tt in enumerate(t):
        feet, support = feet_at(tt, c, steps, ah)
        # 骨盤の高さ: 両足に届く高さ（股関節〜足首が脚の長さの 98% 以内）に下げる（段階 20 と同じ）
        z = z_hip
        for sd, a_ in arm_angles(tt, c, steps, ah).items():
            q[rb.arm[sd]] = a_
        for _ in range(2):
            px = xy if k > 0 else com_xy[k]
            for sd, hy in (("L", HIP_Y * s), ("R", -HIP_Y * s)):
                dx, dy = feet[sd][0] - px[0], feet[sd][1] - (px[1] + hy)
                z = min(z, feet[sd][2] + np.sqrt(max(Lr**2 - dx * dx - dy * dy, 0.0)))
            q, ok = rb.pelvis_for_com(com_xy[k], z, feet, q, xy)
            xy = q[0:2].copy()
        xy = q[0:2].copy()
        ok_all &= ok
        Q.append(q.copy()); sup.append(support)
        if c.extra.get("stab"):          # 段階 23: 骨盤を水平に δ ずらしたときの脚の関節角の変化（足は床に固定）
            dl = 0.005 * s; Sk = np.zeros((len(JOINTS), 2))
            adr_ = [rb.qa[j] for j in JOINTS]
            for ax_ in range(2):
                rx = q[0:3].copy(); rx[ax_] += dl
                q2, _ = rb.ik(rx, feet, q.copy())
                Sk[:, ax_] = (q2[adr_] - q[adr_]) / dl
            Sens.append(Sk)
        tauff.append(rb.static_torques(q, support) if c.ff else np.zeros(len(JOINTS)))
    return dict(rb=rb, t=t, Q=np.array(Q), ok=bool(ok_all), support=sup, zref=zr, zmp_lipm=zmp_lipm, com=com_xy, zc=zc,
                tauff=np.array(tauff), steps=steps, T_end=T_end, S=np.array(Sens) if Sens else None)


def execute(c: Cfg, P=None, record=False):
    if P is None:
        P = plan(c)
    rb = P["rb"]; m, d = rb.m, rb.d; s = c.s
    act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in JOINTS]
    adr = [rb.qa[j] for j in JOINTS]
    if rb.arm:
        act += [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"sh{sd}") for sd in "LR"]
        adr += [rb.arm[sd] for sd in "LR"]
    kp = c.kp * s**3
    Qj = P["Q"][:, adr].copy()
    if c.ff:
        Qj[:, :len(JOINTS)] += P["tauff"] / kp
    mujoco.mj_resetData(m, d); d.qpos[:] = P["Q"][0]; d.qpos[2] += 0.0005 * s; mujoco.mj_forward(m, d)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    sole = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"sole{sd}"): sd for sd in "LR"}
    n = int(round(P["T_end"] / m.opt.timestep)); c6 = np.zeros(6)
    rec = {k: [] for k in ("t", "root", "cop", "com", "tilt", "tau", "mz")}
    tau_lim = c.tau * s**3; sat = 0.0; tilt_max = 0.0; fell = False
    # 段階 23: 閉ループの安定化（DCM のずれ → 骨盤を水平にずらす）と、外から押す力
    st = c.extra.get("stab"); push = c.extra.get("push")
    if st:
        dtp = P["t"][1] - P["t"][0]; om = np.sqrt(G / P["zc"])
        xi_ref = P["com"] + np.gradient(P["com"], dtp, axis=0) / om
        nc = max(1, int(round(st.get("dt_c", 0.01) / m.opt.timestep)))
        ndl = int(round(st.get("delay", 0.0) / (nc * m.opt.timestep)))
        rng_ = np.random.default_rng(st.get("seed", 0)); queue = []; corr = np.zeros(2)
        Kx, dmax = st.get("K", 1.0), st.get("max", 0.08) * s
        rec_st = []
    old = None; t_sw = -1.0; step_info = None; t_next = 0.0; ks_fix = None; offs = {}; n_re = 0
    for k in range(n):
        tt = d.time
        i = min(int(tt / (P["t"][1] - P["t"][0])), len(P["t"]) - 2); w = (tt - P["t"][i]) / (P["t"][1] - P["t"][0])
        u_ = Qj[i] * (1 - w) + Qj[i + 1] * w
        if st:
            if k % nc == 0:
                mujoco.mj_subtreeVel(m, d)
                cm = d.subtree_com[1][:2] + rng_.normal(0, st.get("noise", 0.0), 2)
                vm = d.subtree_linvel[1][:2] + rng_.normal(0, st.get("noise_v", 0.0), 2)
                e = (cm + vm / om) - (xi_ref[i] * (1 - w) + xi_ref[i + 1] * w)
                if st.get("lpf", 0.0) > 0:      # 1 次の低域通過フィルタ（時定数 lpf 秒）
                    ef = e if not rec_st else ef + min(1.0, nc * m.opt.timestep / st["lpf"]) * (e - ef)
                    e_use = ef
                else:
                    e_use = e
                queue.append(np.clip(-Kx * e_use, -dmax, dmax))
                corr = queue.pop(0) if len(queue) > ndl else np.zeros(2)
                rec_st.append(e)
                # 段階 24: 足の置き場所の戦略。DCM のずれが e_th を超えたら一度だけ、次に着地する足を capture point の方へずらして計画し直す
                # 段階 24: 足の置き場所の戦略。DCM のずれが e_th を超えたら、次に着地する足を capture point の方へずらして計画し直す。
                # 着地の t_min 前まで、50 ms ごとに残りのずれで足し直す（押されている間はずれがまだ育つため）
                if st.get("step") and tt >= t_next and np.linalg.norm(e) > st.get("e_th", 0.03) * s:
                    stp = P["steps"]; Tds_ = c.ds * c.T_step * np.sqrt(s)
                    if ks_fix is not None and tt > stp[ks_fix][1] - Tds_ / 2:     # 前の踏み出しが着地した → 次の足で続けて直す（複数歩）
                        ks_fix = None
                    if ks_fix is None:
                        ks_fix = next((kk for kk, st_ in enumerate(stp) if st_[1] - Tds_ / 2 - tt > st.get("t_min", 0.15) * np.sqrt(s)), None)
                    ks = ks_fix
                    if ks is not None and ks < len(stp) - 1 and stp[ks][1] - Tds_ / 2 - tt > st.get("t_min", 0.15) * np.sqrt(s):
                        t_land = stp[ks][1] - Tds_ / 2
                        dl = offs.get(ks, np.zeros(2)) + e * np.exp(om * (t_land - tt)) * st.get("g_step", 1.0)
                        side = stp[ks][2]
                        if side == "L":
                            dl[1] = max(dl[1], -0.02 * s)     # 左足は右へ寄せない（脚が交差する）
                        else:
                            dl[1] = min(dl[1], 0.02 * s)
                        nrm = np.linalg.norm(dl); dmx = st.get("step_max", 0.30) * s
                        dl = dl * min(1.0, dmx / max(nrm, 1e-9))
                        offs[ks] = dl
                        ex2 = {**c.extra, "step_off": {kk: (v / s).tolist() for kk, v in offs.items()}}
                        if st.get("from_state"):       # 計画を、いまの重心の位置・速度から作り直す（目標の跳びをなくす）
                            # 位置は計画のまま、DCM のずれ e を速度に入れる（重心の計画と MuJoCo の 2〜3 cm の差で目標が跳ばないように）
                            vp = (P["com"][min(i + 1, len(P["com"]) - 1)] - P["com"][max(i - 1, 0)]) / (2 * dtp)
                            ex2["replan_from"] = {"i": int(i), "x": P["com"][i].tolist(), "v": (vp + om * e).tolist()}
                        c2 = Cfg(**{**c.__dict__, "extra": ex2})
                        P2 = plan(c2)
                        Qj2 = P2["Q"][:, adr].copy()
                        if c.ff:
                            Qj2[:, :len(JOINTS)] += P2["tauff"] / kp
                        old = (Qj, P["S"]); Qj = Qj2; P = {**P2, "rb": rb}
                        xi_ref = P["com"] + np.gradient(P["com"], dtp, axis=0) / om
                        t_sw = tt; n_re += 1
                        step_info = dict(offs={int(kk): np.round(v, 3).tolist() for kk, v in offs.items()}, t=tt, n=n_re)
                        t_next = tt + st.get("re_dt", 0.05)
            Sk = P["S"][i] * (1 - w) + P["S"][i + 1] * w
            u_ = u_.copy(); u_[:len(JOINTS)] += Sk @ corr
            if old is not None and tt - t_sw < 0.05:      # 切り替えを 50 ms でなめらかに
                bl = (tt - t_sw) / 0.05; uo = old[0][i] * (1 - w) + old[0][i + 1] * w
                uo = uo.copy(); uo[:len(JOINTS)] += (old[1][i] * (1 - w) + old[1][i + 1] * w) @ corr
                u_ = (1 - bl) * uo + bl * u_
        d.ctrl[act] = u_
        if push:
            d.xfrc_applied[1, :3] = np.r_[push["F"], 0.0] * s**3 if push["t"] <= tt < push["t"] + push.get("dur", 0.1) * np.sqrt(s) else 0.0
        mujoco.mj_step(m, d)
        Rm = d.xmat[1].reshape(3, 3); tilt = D(np.arccos(np.clip(Rm[2, 2], -1, 1)))
        tilt_max = max(tilt_max, tilt)
        sat += m.opt.timestep * np.any(np.abs(d.actuator_force[act]) >= 0.999 * tau_lim)
        if record and k % 10 == 0:
            F = np.zeros(3); Mo = np.zeros(3)
            for ci in range(d.ncon):
                cc = d.contact[ci]
                if floor not in (cc.geom1, cc.geom2):
                    continue
                mujoco.mj_contactForce(m, d, ci, c6)
                fw = cc.frame.reshape(3, 3).T @ c6[:3]
                if cc.geom1 != floor:
                    fw = -fw
                F += fw; Mo += np.cross(cc.pos, fw)
            cop = np.array([-Mo[1] / F[2], Mo[0] / F[2]]) if F[2] > 1e-3 else np.array([np.nan, np.nan])
            rec["mz"].append(Mo[2] - (cop[0] * F[1] - cop[1] * F[0]) if F[2] > 1e-3 else np.nan)   # 足圧中心まわりの鉛直軸のモーメント（ヨー）
            rec["t"].append(tt); rec["root"].append(d.qpos[0:3].copy()); rec["cop"].append(cop)
            rec["com"].append(d.subtree_com[1].copy()); rec["tilt"].append(tilt); rec["tau"].append(d.actuator_force[act].copy())
        if d.qpos[2] < 0.5 * rb.z0 or not np.isfinite(d.qpos).all():
            fell = True; break
    plan_end = P["Q"][-1][0:2]
    pos_err = float(np.linalg.norm(d.qpos[0:2] - plan_end)) if not fell else np.inf
    yaw = float(D(2 * np.arctan2(d.qpos[6], d.qpos[3])))   # 上体の向き（鉛直軸まわり）
    out = dict(fell=fell, tilt_max=float(tilt_max), pos_err=pos_err, sat_s=float(sat), t_end=float(d.time),
               dist=float(d.qpos[0]), plan_dist=float(plan_end[0]), yaw=yaw)
    out["ok"] = bool(not fell and tilt_max < 15 and pos_err < 0.10 * s)
    if st and step_info is not None:
        out["step"] = step_info
    if st:
        out["dcm_rms"] = float(np.sqrt(np.mean(np.sum(np.array(rec_st)**2, 1)))) if rec_st else np.nan
    if record:
        out["rec"] = {k: np.array(v) for k, v in rec.items()}
    return out


def run1(args):
    """並列計算用: (Cfg の上書き) → 結果の要約"""
    over = args
    c = Cfg(**over)
    try:
        P = plan(c); ex = execute(c, P)
        ex["ik_ok"] = P["ok"]
    except Exception as e:   # noqa: BLE001
        ex = dict(ok=False, fell=True, err=str(e))
    ex.update(over=over)
    return ex


def t_min(args):
    """歩幅 L で、8 歩を歩ける最短の 1 歩の時間（二分法、[0.3, 1.6] s）"""
    over, lo, hi, it = args
    ok = lambda T: run1({**over, "T_step": T})["ok"]
    if not ok(hi):
        return (np.nan, hi)
    for _ in range(it):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return (hi, lo)


def peak_torque(args):
    """トルクの上限をなくしたときの最大の関節トルク / 上限（必要なトルクの比）"""
    over = args
    c = Cfg(**{**over, "tau": 1e6})
    P = plan(c); ex = execute(c, P, record=True)
    tau = np.abs(ex["rec"]["tau"]).max(0) if len(ex["rec"]["tau"]) else np.full(len(JOINTS), np.nan)
    lim = TAU * c.s**3
    return dict(over=over, ratio=float(tau.max() / lim), per_joint={j: float(v / lim) for j, v in zip(JOINTS, tau)}, ok=ex["ok"])


def layer_check(c: Cfg):
    """1 層目（LIPM の ZMP）と 2 層目（MuJoCo の足圧中心）を比べる（片脚支持の区間）"""
    P = plan(c); ex = execute(c, P, record=True)
    r = ex["rec"]; tz = r["t"]
    zl = np.column_stack([np.interp(tz, P["t"], P["zmp_lipm"][:, a]) for a in range(2)])
    zr = np.column_stack([np.interp(tz, P["t"], P["zref"][:, a]) for a in range(2)])
    single = np.array([P["support"][min(int(t / (P["t"][1] - P["t"][0])), len(P["t"]) - 1)] != "LR" for t in tz])
    ok = single & np.isfinite(r["cop"][:, 0])
    rms = lambda a, b: np.sqrt(np.nanmean((a[ok] - b[ok])**2, 0))
    comp = np.column_stack([np.interp(tz, P["t"], P["com"][:, a]) for a in range(2)])
    return dict(rms_lipm_vs_ref=rms(zl, zr).tolist(), rms_cop_vs_lipm=rms(r["cop"], zl).tolist(),
                rms_com_vs_plan=np.sqrt(np.mean((r["com"][:, :2] - comp)**2, 0)).tolist(), ex={k: v for k, v in ex.items() if k != "rec"},
                series=dict(t=tz, cop=r["cop"], zl=zl, zr=zr, com=r["com"][:, :2], comp=comp, single=single))


def main():
    from multiprocessing import Pool
    out = {}
    TS = [1.2, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4]; LS = [0.15, 0.25, 0.35, 0.45]
    with Pool(4) as pool:
        grid = pool.map(run1, [{"T_step": T, "L": L} for L in LS for T in TS])
        stat = pool.map(run1, [{"planner": "static", "T_step": T, "ds": ds} for ds in (0.2, 0.5) for T in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)])
        tm = pool.map(t_min, [({"L": L}, 0.3, 1.6, 7) for L in LS] + [({"planner": "static", "ds": 0.5}, 0.8, 8.0, 7)])
        scale = pool.map(peak_torque, [{"s": s_, "z_hip": z} for s_ in (0.5, 1.0, 2.0) for z in (0.80, 0.85, 0.88)])
        scale_ok = pool.map(run1, [{"s": s_, "z_hip": z} for s_ in (0.5, 1.0, 2.0) for z in (0.80, 0.85, 0.88)])
    print("=== ZMP の計画（開ループ）: 8 歩を歩けたか（○ / × 転倒 / △ 位置の誤差 10 cm 以上）。速さ = L / T ===")
    print("| L \\ T_step | " + " | ".join(f"{T}" for T in TS) + " |")
    for a, L in enumerate(LS):
        row = []
        for b, T in enumerate(TS):
            g = grid[a * len(TS) + b]
            row.append("○" if g["ok"] else ("×" if g.get("fell") else "△"))
        print(f"| {L} | " + " | ".join(row) + " |")
    out["grid"] = [{k: v for k, v in g.items()} for g in grid]
    print("=== 最短の 1 歩の時間（二分法） ===")
    for L, t_ in zip(LS + ["静的 L=0.25, ds 0.5"], tm):
        print(f"  L = {L}: T_min = {t_[0]:.3f} s（{t_[1]:.3f}〜{t_[0]:.3f}）" + (f"、速さ {float(L) / t_[0]:.2f} m/s" if isinstance(L, float) and np.isfinite(t_[0]) else ""))
    out["t_min"] = {str(L): t_ for L, t_ in zip(LS + ["static_ds0.5"], tm)}
    print("=== 準静的の計画（重心を支持する足の真上へ） ===")
    for g in stat:
        print(f"  ds {g['over']['ds']}, T_step {g['over']['T_step']}: {'歩けた' if g['ok'] else ('転倒' if g.get('fell') else '位置の誤差')}"
              f"（上体の傾き {g.get('tilt_max', np.nan):.1f}°、進んだ距離 {g.get('dist', np.nan):.2f} / {g.get('plan_dist', np.nan):.2f} m）")
    out["static"] = stat
    print("=== 倍率 s と、骨盤の高さ（膝の曲げ）: 必要なトルク / 上限（250 s³ N·m）、歩けたか（T_step 0.8√s、L 0.25 s） ===")
    for pt, ok in zip(scale, scale_ok):
        j = max(pt["per_joint"], key=pt["per_joint"].get)
        print(f"  s = {pt['over']['s']}, 股関節の高さ {pt['over']['z_hip']}: 必要 {pt['ratio']:.2f}（{j}）、上限ありで {'歩けた' if ok['ok'] else '歩けない'}")
    out["scale"] = scale; out["scale_ok"] = scale_ok
    lc = layer_check(Cfg())
    print(f"=== 照合（T 0.8 s、L 0.25 m、片脚支持の区間）: LIPM の ZMP と目標 {np.round(lc['rms_lipm_vs_ref'], 4)} m、"
          f"MuJoCo の足圧中心と LIPM {np.round(lc['rms_cop_vs_lipm'], 4)} m、重心と計画 {np.round(lc['rms_com_vs_plan'], 4)} m（x, y の RMS） ===")
    out["layer"] = {k: v for k, v in lc.items() if k != "series"}
    json.dump(out, open(DATA / "field21_walk.json", "w"), indent=1, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"[{time.time() - T_START:.0f} s]")


VARIANTS = {
    "基準": {},
    "計算: 時間刻み 0.25 ms": {"dt": 2.5e-4},
    "計算: 時間刻み 1 ms": {"dt": 1e-3},
    "系統: サーボ kp 1e4": {"kp": 1e4},
    "系統: サーボ kp 1e5": {"kp": 1e5},
    "系統: 重力の補償なし": {"ff": False},
    "系統: 接触を柔らかく（solref 0.02）": {"solref": "0.02 1"},
    "系統: 摩擦 μ 0.5": {"mu": 0.5},
    "系統: 摩擦 μ 1.5（ねじれの滑りの確認）": {"mu": 1.5},
    "系統: 回転子の慣性 0.005 / 0.5": None,
    "系統: プレビュー 0.8 s": {"preview": 0.8},
    "系統: 両脚支持 10% / 30%": None,
    "系統: 骨盤の高さ 0.80 / 0.88": None,
}


TS_ERR = np.round(np.arange(0.32, 0.801, 0.04), 3)


def scan_summary(rs):
    """T の走査の結果 → 成功の割合、遅い側の連続した成功の下端 T_slow、最も速い成功、最大の向きのずれ"""
    oks = [r["ok"] for r in rs]
    k = len(oks)
    while k > 0 and oks[k - 1]:
        k -= 1
    t_slow = float(TS_ERR[k]) if k < len(oks) else np.nan
    fast = [float(T) for T, o in zip(TS_ERR, oks) if o]
    return dict(frac=float(np.mean(oks)), t_slow=t_slow, t_fast=min(fast) if fast else np.nan,
                gap=[float(T) for T, o in zip(TS_ERR, oks) if not o and fast and T > min(fast) and T < t_slow],
                yaw_max=float(np.nanmax([abs(r.get("yaw", np.nan)) if not r.get("fell") else np.nan for r in rs] + [0])),
                oks=oks)


def errors_main():
    """成功の地図が T について単調でない（F37）ので、二分法ではなく T の走査で比べる"""
    from multiprocessing import Pool
    names = []
    pairs = {"系統: 回転子の慣性 0.005 / 0.5": [{"armature": 0.005}, {"armature": 0.5}],
             "系統: 両脚支持 10% / 30%": [{"ds": 0.1}, {"ds": 0.3}],
             "系統: 骨盤の高さ 0.80 / 0.88": [{"z_hip": 0.80}, {"z_hip": 0.88}]}
    for k, v in VARIANTS.items():
        for vv in (pairs[k] if v is None else [v]):
            names.append((k, vv))
    rng = np.random.default_rng(1)
    mc = [{"trunk": 1 + rng.uniform(-.1, .1), "thigh": 1 + rng.uniform(-.1, .1), "shank": 1 + rng.uniform(-.1, .1),
           "foot": 1 + rng.uniform(-.15, .15)} for _ in range(16)]
    sets = [vv for _, vv in names] + [{"extra": {"mass": m_}} for m_ in mc]
    jobs = [{"L": 0.25, **vv, "T_step": float(T)} for vv in sets for T in TS_ERR]
    with Pool(4) as pool:
        res = pool.map(run1, jobs, chunksize=2)
    n = len(TS_ERR); summ = [scan_summary(res[i * n:(i + 1) * n]) for i in range(len(sets))]
    print(f"=== 誤差の評価：歩幅 0.25 m、T_step = {TS_ERR[0]}〜{TS_ERR[-1]} s（0.04 s 刻み）で 8 歩を歩けたか ===")
    print("| 条件 | 成功の割合 | 遅い側の下端 T_slow | 最も速い成功 | 間の失敗 | 向きのずれの最大 |")
    out = {"T": TS_ERR.tolist(), "variants": []}
    for (k, vv), sm in zip(names, summ):
        print(f"| {k} {vv if vv else ''} | {sm['frac']:.2f} | {sm['t_slow']:.2f} | {sm['t_fast']:.2f} | {sm['gap']} | {sm['yaw_max']:.1f}° | "
              + "".join("○" if o else "×" for o in sm["oks"]), flush=True)
        out["variants"].append(dict(name=k, over=vv, **sm))
    ms = summ[len(names):]
    ts = np.array([m_["t_slow"] for m_ in ms]); fr = np.array([m_["frac"] for m_ in ms])
    print(f"入力（部品表の質量、16 回）: T_slow = {np.nanmean(ts):.3f} ± {np.nanstd(ts, ddof=1):.3f} s、成功の割合 {fr.mean():.2f} ± {fr.std(ddof=1):.2f}")
    for m_ in ms:
        print("  " + "".join("○" if o else "×" for o in m_["oks"]))
    out["mc"] = {"masses": mc, "summary": ms}
    json.dump(out, open(DATA / "field21_errors.json", "w"), indent=1, ensure_ascii=False, default=float)


def progress(ex):
    """連続な評価: 転んだら進んだ距離の割合、歩けたら 1 − 位置の誤差の割合"""
    if ex.get("fell", True):
        return max(0.0, ex.get("dist", 0.0) / max(ex.get("plan_dist", 1.0), 1e-9))
    return max(0.0, 1 - ex["pos_err"] / ex["plan_dist"]) - ex["tilt_max"] / 90


def landscape_main():
    """評価の地形: (1) 1 歩の時間の細かい走査、(2) 2 種類の評価関数でネルダー・ミードを同じ初期値から"""
    from multiprocessing import Pool
    from scipy.optimize import minimize
    TS = np.round(np.arange(0.40, 0.801, 0.02), 3)
    with Pool(4) as pool:
        scan = pool.map(run1, [{"T_step": float(T), "L": 0.25} for T in TS])
    out = {"scan": [dict(T=float(T), ok=g["ok"], fell=g.get("fell"), progress=progress(g), tilt=g.get("tilt_max")) for T, g in zip(TS, scan)]}
    print("=== 1 歩の時間の走査（L 0.25 m）: T, 成功, 進み具合, 上体の最大の傾き ===")
    for r in out["scan"]:
        print(f"  {r['T']:.2f}  {'○' if r['ok'] else '×'}  {r['progress']:.3f}  {r['tilt']:.1f}°")
    # 最適化: 変数 (T_step, 骨盤の高さ, 両脚支持の割合)。速く歩きたい（T を小さく）
    x0 = np.array([0.8, 0.85, 0.2]); simplex = np.array([x0, x0 + [-0.15, 0, 0], x0 + [0, -0.04, 0], x0 + [0, 0, 0.1]])
    clip = lambda x: dict(T_step=float(np.clip(x[0], 0.25, 1.5)), L=0.25, z_hip=float(np.clip(x[1], 0.75, 0.89)), ds=float(np.clip(x[2], 0.05, 0.6)))
    hist = {}
    for name, J in [("二値（成功なら T、失敗なら T + 10）", lambda ex, o: o["T_step"] + (0 if ex["ok"] else 10)),
                    ("連続（T + 10 × (1 − 進み具合)）", lambda ex, o: o["T_step"] + 10 * (1 - progress(ex)))]:
        h = []

        def f(x, J=J, h=h):
            o = clip(x); ex = run1(o); v = J(ex, o); h.append((o["T_step"], o["z_hip"], o["ds"], v, ex["ok"])); return v
        r = minimize(f, x0, method="Nelder-Mead", options=dict(maxfev=45, initial_simplex=simplex, xatol=1e-3, fatol=1e-4))
        best = min((x for x in h if x[4]), key=lambda x: x[0], default=None)
        print(f"=== 最適化 {name}: 評価 {len(h)} 回、最良の成功 T = {best[0]:.3f} s（骨盤 {best[1]:.3f}、両脚支持 {best[2]:.2f}）" if best else f"=== {name}: 成功なし")
        hist[name] = h
    out["opt"] = hist
    json.dump(out, open(DATA / "field21_landscape.json", "w"), indent=1, ensure_ascii=False, default=float)


def static_pred():
    """F36: 準静的の計画の ZMP のずれ（LIPM: zc/g × 重心の加速度）が足の幅の半分（5 cm）になる T_step を二分法で見積もる"""
    def dev(T):
        P = plan(Cfg(planner="static", T_step=T, ds=0.5)); dt = P["t"][1] - P["t"][0]
        acc = np.gradient(np.gradient(P["com"][:, 1], dt), dt)
        return np.abs(P["com"][:, 1] - P["zc"] / G * acc - P["zref"][:, 1]).max()
    for T in (2.0, 4.0, 6.0, 8.0):
        print(f"  T_step {T}: 左右の ZMP のずれの最大 {dev(T):.3f} m（足の幅の半分 {FOOT_W / 2} m）")
    lo, hi = 4.0, 8.0
    for _ in range(8):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if dev(mid) > FOOT_W / 2 else (lo, mid)
    print(f"見積もり T_step = {hi:.3f} s（MuJoCo の二分法 6.144 s）")


def figure():
    res = json.load(open(DATA / "field21_walk.json"))
    lc = layer_check(Cfg()); sr = lc["series"]
    fig, axs = plt.subplots(2, 2, figsize=(13, 9)); ax = axs.ravel()
    ax[0].plot(sr["t"], sr["zr"][:, 1], "k:", lw=1, label="ZMP reference (y)")
    ax[0].plot(sr["t"], sr["zl"][:, 1], "k--", lw=1, label="LIPM ZMP (plan)")
    cop = np.where(sr["single"][:, None], sr["cop"], np.nan)
    ax[0].plot(sr["t"], cop[:, 1], "tab:red", lw=1, label="MuJoCo CoP (single support)")
    ax[0].plot(sr["t"], sr["com"][:, 1], "tab:blue", lw=1.5, label="MuJoCo CoM (y)")
    ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("lateral y [m]"); ax[0].legend(fontsize=7); ax[0].set_title("walking T=0.8 s, L=0.25 m: lateral ZMP/CoM", fontsize=9)
    TS = [1.2, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4]; LS = [0.15, 0.25, 0.35, 0.45]
    M = np.zeros((len(LS), len(TS)))
    for a in range(len(LS)):
        for b in range(len(TS)):
            g = res["grid"][a * len(TS) + b]; M[a, b] = 2 if g["ok"] else (0 if g.get("fell") else 1)
    ax[1].imshow(M, cmap="RdYlGn", vmin=0, vmax=2, origin="lower", aspect="auto")
    ax[1].set_xticks(range(len(TS))); ax[1].set_xticklabels(TS); ax[1].set_yticks(range(len(LS))); ax[1].set_yticklabels(LS)
    for a in range(len(LS)):
        for b in range(len(TS)):
            ax[1].text(b, a, f"{LS[a] / TS[b]:.2f}", ha="center", va="center", fontsize=7)
    ax[1].set_xlabel("step time T_step [s]"); ax[1].set_ylabel("step length L [m]")
    ax[1].set_title("ZMP plan, open loop: green = 8 steps OK, yellow = drift > 10 cm, red = fall\n(numbers: speed m/s)", fontsize=9)
    sc = res["scale"]; S = [0.5, 1.0, 2.0]; Z = [0.80, 0.85, 0.88]
    for z, mk in zip(Z, "os^"):
        ax[2].plot(S, [next(p["ratio"] for p in sc if p["over"]["s"] == s_ and p["over"]["z_hip"] == z) for s_ in S], mk + "-", label=f"hip height {z} m" + (" (knee bent)" if z < 0.82 else " (knee nearly straight)" if z > 0.87 else ""))
    ax[2].axhline(1, color="k", lw=0.8, ls=":"); ax[2].text(0.52, 1.03, "torque limit 250 s³", fontsize=7)
    ax[2].set_xscale("log"); ax[2].minorticks_off(); ax[2].set_xticks(S); ax[2].set_xticklabels(["0.5", "1", "2"])
    ax[2].set_xlabel("scale s (T_step 0.8√s, L 0.25 s)"); ax[2].set_ylabel("peak joint torque / limit"); ax[2].legend(fontsize=7)
    ax[2].set_title("required torque vs scale (∝ s for fixed shape)", fontsize=9)
    try:
        ld = json.load(open(DATA / "field21_landscape.json"))
        Tq = [r["T"] for r in ld["scan"]]; pr = [r["progress"] for r in ld["scan"]]; okq = [r["ok"] for r in ld["scan"]]
        ax[3].plot(Tq, pr, "k-", lw=0.8)
        ax[3].scatter(Tq, pr, c=["tab:green" if o else "tab:red" for o in okq], zorder=3)
        ax[3].set_xlabel("step time T_step [s] (L 0.25 m)"); ax[3].set_ylabel("continuous score (progress − tilt/90°)")
        ax[3].set_title("success is NOT monotonic in T: failure window 0.48–0.60 s\n(green OK, red fail; fails = yaw slip of the stance foot)", fontsize=9)
    except FileNotFoundError:
        ax[3].axis("off")
    fig.tight_layout(); fig.savefig(OUT / "field21_walk.png", dpi=130)
    print("図 → figures/field/field21_walk.png")


if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else "main"
    {"main": main, "errors": errors_main, "landscape": landscape_main, "figure": figure, "static_pred": static_pred}[arg]()
