"""Field20: 二本脚の段差を MuJoCo で上る（下半身の段 3）.

Sim10 Part B（準静的な式）の結果
  ・上体を立てたまま 0.18 m、上体を前に傾けて重心を前へ移すと 0.60 m（膝・股関節のトルクの上限 各 250 N·m）
  ・可動域の限界 0.63 m（股関節の屈曲 ≤ 120°、膝 ≤ 140°）
  ・長さ s 倍・質量 s³ 倍・トルクの上限 s³ 倍で、越えられる段差 / s は s = 0.5 → 0.63、1 → 0.60、2 → 0.18（s⁴ の逆転）
を、膝と足首を持つ二本脚の MuJoCo の機体で、実際に足を段に乗せて上って確かめる。

機体（Sim10 Part B と同じ寸法・質量）: 上体 56 kg（腕を含む）、大腿 16.5 kg・0.45 m、下腿 3 kg・0.45 m、足 1 kg（かかと 0.10 m、つま先 0.15 m）。
  矢状面（前後・上下・ピッチ）に拘束する（平面の二本脚の実験装置 RABBIT・MABEL のブームと同じ）。左右の倒れは段 4 で扱う。
  関節は位置サーボ。股関節・膝・足首のトルクの上限 250 s³ N·m。
動作（準静的、各相 2 s × √s）
  P0 両足で立つ → P1 右足を振り上げて段の上に置く（重心は左足の上）→ P2 重心を右足の上へ移す（上体を前傾）
  → P3 右足で体を持ち上げ、左足を引き上げる → P4 左足を段に置いて立つ
  重心の位置は、各時刻で骨盤の前後位置を逆運動学で調整して決める（重心の目標は支持している足の上）。
照合の層
  1 層目 (a) Sim10 の式、(b) 計画した姿勢の列で、支持している足の床反力を未知数とする準静的なつり合い（τ = g(q) − J_cᵀ f）から関節トルクを求める
  2 層目 MuJoCo で実際に上る（トルクの上限つきのサーボ）。段の高さを二分法で動かし、上れる最大の高さを求める
"""
import os
os.environ.setdefault("MUJOCO_GL", "disable")
import json
import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from common import DATA, FIG, smooth_step

D = np.degrees
R = np.radians
G = 9.80665
OUT = FIG / "field"; OUT.mkdir(exist_ok=True)
T_START = time.time()

# ---- 機体（Sim10 Part B と同じ寸法・質量。s = 1） --------------------------------------------
L_TH, L_SH = 0.45, 0.45
M_TRUNK, M_TH, M_SH, M_FOOT = 56.0, 16.5, 3.0, 1.0       # 合計 97 kg
TRUNK = (0.25, 0.4, 0.6); TRUNK_COM = 0.25               # 股関節から上体の重心までの高さ
HEEL, TOE, FOOT_W, FOOT_H, ANK_H = 0.10, 0.15, 0.10, 0.04, 0.06
HIP_Y = 0.10
TAU = 250.0                                              # 股関節・膝・足首のトルクの上限（s = 1）
HIP_RANGE, KNEE_RANGE, ANK_RANGE = (-30, 120), (-1, 140), (-50, 50)
STEP_FWD, CLEAR = 0.30, 0.05                             # 段の上に置く足首の前方位置、足を上げるすき間
STAND = 0.895                                            # 立っているときの股関節〜足首の距離（脚の長さ 0.9 の 99.4%、膝 12°）
REACH = 0.98                                             # 計画で使う股関節〜足首の距離の上限（脚の長さに対する比）


@dataclass
class Cfg:
    s: float = 1.0
    h: float = 0.2                    # 段の高さ（実寸）
    mode: str = "lean"                # "upright"（上体を立てたまま）/ "lean"（上体を前傾）
    lean_deg: float = 40.0            # 持ち上げ始めの上体の前傾
    xc: float = 0.10                  # 持ち上げ始めの重心の前方位置（支持する足首から、s = 1 の長さ）
    step_fwd: float = STEP_FWD        # 段の上に置く足首の前方位置（s = 1）。段の前面はかかとの 3 cm 手前
    T_phase: float = 2.0              # 1 相の時間（s = 1）。実時間は × √s
    mu: float = 0.8
    kp: float = 3e4                   # サーボの硬さ（s = 1、実際は × s³）
    tau: float = TAU                  # トルクの上限（s = 1、実際は × s³）
    dt: float = 2.5e-4
    integrator: str = "implicitfast"
    armature: float = 0.05            # 関節の回転子の見かけの慣性（s = 1、× s⁵）
    solref: str = "0.004 1"
    extra: dict = field(default_factory=dict)


def build_xml(c: Cfg):
    s = c.s; m3 = s**3; tau = c.tau * m3
    ms = c.extra.get("mass", {})
    M_TRUNK, M_TH, M_SH, M_FOOT = (M0 * ms.get(k, 1.0) for M0, k in
                                   [(globals()["M_TRUNK"], "trunk"), (globals()["M_TH"], "thigh"), (globals()["M_SH"], "shank"), (globals()["M_FOOT"], "foot")])
    tau_ank = c.extra.get("tau_ank", c.tau) * m3
    ib = lambda m, a, b, cc: f"{m * (b*b + cc*cc) / 12:.6g} {m * (a*a + cc*cc) / 12:.6g} {m * (a*a + b*b) / 12:.6g}"
    tx, ty, tz = [v * s for v in TRUNK]
    rod = lambda m, L: f"{m * (0.06*s)**2 / 2:.6g} {m * L * L / 12:.6g} {m * L * L / 12:.6g}"
    lth, lsh = L_TH * s, L_SH * s
    fx0, fx1 = -HEEL * s, TOE * s; fl = fx1 - fx0; fh = FOOT_H * s; ah = ANK_H * s
    kp = c.kp * m3; kv = kp * 0.05 * np.sqrt(s); arm = c.armature * s**5
    jr = lambda r: f'range="{r[0]} {r[1]}" limited="true"'
    leg = ""
    for side, y in [("L", HIP_Y * s), ("R", -HIP_Y * s)]:
        leg += f"""
      <body name="thigh{side}" pos="0 {y} 0">
        <joint name="hip{side}" axis="0 -1 0" {jr(HIP_RANGE)} armature="{arm}"/>
        <inertial pos="0 0 {-lth/2}" mass="{M_TH*m3}" diaginertia="{rod(M_TH*m3, lth)}"/>
        <geom type="capsule" fromto="0 0 0 0 0 {-lth}" size="{0.06*s}" contype="0" conaffinity="0" rgba="0.3 0.5 0.9 1"/>
        <body name="shank{side}" pos="0 0 {-lth}">
          <joint name="knee{side}" axis="0 1 0" {jr(KNEE_RANGE)} armature="{arm}"/>
          <inertial pos="0 0 {-lsh/2}" mass="{M_SH*m3}" diaginertia="{rod(M_SH*m3, lsh)}"/>
          <geom name="shin{side}" type="capsule" fromto="0 0 0 0 0 {-lsh}" size="{0.045*s}" contype="0" conaffinity="0" rgba="0.3 0.5 0.9 1"/>
          <body name="foot{side}" pos="0 0 {-lsh}">
            <joint name="ank{side}" axis="0 -1 0" {jr(ANK_RANGE)} armature="{arm}"/>
            <site name="ank{side}" pos="0 0 0" size="0.01"/>
            <inertial pos="{(fx0+fx1)/2} 0 {-ah + fh/2}" mass="{M_FOOT*m3}" diaginertia="{ib(M_FOOT*m3, fl, FOOT_W*s, fh)}"/>
            <geom name="sole{side}" type="box" pos="{(fx0+fx1)/2} 0 {-ah + fh/2}" size="{fl/2} {FOOT_W*s/2} {fh/2}" friction="{c.mu} 0.005 0.0001" rgba="0.4 0.4 0.45 1"/>
          </body>
        </body>
      </body>"""
    tl = lambda j: tau_ank if j == "ank" else tau
    acts = "".join(f'<position name="{j}{sd}" joint="{j}{sd}" kp="{kp}" kv="{kv}" forcelimited="true" forcerange="{-tl(j)} {tl(j)}"/>'
                   for sd in "LR" for j in ("hip", "knee", "ank"))
    h = c.h; face = (c.step_fwd - HEEL - 0.03) * s; er = min(0.01 * s, h / 2) if h > 0 else 0
    step = (f'<geom name="step" type="box" pos="{face + 1.0*s} 0 {h/2}" size="{1.0*s} {0.6*s} {h/2}" friction="{c.mu} 0.005 0.0001" rgba="0.6 0.6 0.65 1"/>'
            if h > 0 else "")
    hip_z = ah + STAND * L_TH / 0.45 * s
    return f"""<mujoco model="biped_steps">
  <option gravity="0 0 {-G}" timestep="{c.dt}" integrator="{c.integrator}" cone="elliptic" impratio="10"/>
  <default><geom solref="{c.solref}" solimp="0.95 0.99 0.001"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="{8*s} {8*s} 0.1" friction="{c.mu} 0.005 0.0001" rgba="0.85 0.85 0.8 1"/>
    {step}
    <body name="trunk" pos="0 0 {hip_z}">
      <joint name="rx" type="slide" axis="1 0 0"/>
      <joint name="rz" type="slide" axis="0 0 1"/>
      <joint name="rp" type="hinge" axis="0 1 0"/>
      <inertial pos="0 0 {TRUNK_COM*s}" mass="{M_TRUNK*m3}" diaginertia="{ib(M_TRUNK*m3, tx, ty, tz)}"/>
      <geom type="box" pos="0 0 {tz/2}" size="{tx/2} {ty/2} {tz/2}" contype="0" conaffinity="0" rgba="0.65 0.65 0.7 1"/>
      {leg}
    </body>
  </worldbody>
  <actuator>{acts}</actuator>
</mujoco>"""


JOINTS = [f"{j}{sd}" for sd in "LR" for j in ("hip", "knee", "ank")]


class Robot:
    def __init__(self, c: Cfg):
        self.c = c
        self.m = mujoco.MjModel.from_xml_string(build_xml(c)); self.d = mujoco.MjData(self.m)
        m = self.m
        self.qa = {j: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS + ["rx", "rz", "rp"]}
        self.va = {j: m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS + ["rx", "rz", "rp"]}
        self.site = {sd: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"ank{sd}") for sd in "LR"}
        self.foot = {sd: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"foot{sd}") for sd in "LR"}
        self.hip_z0 = m.body_pos[1][2]
        self.lim = {j: np.radians(r) for j, r in zip(JOINTS, [HIP_RANGE, KNEE_RANGE, ANK_RANGE] * 2)}
        for sd in "LR":       # 逆運動学の初期値: 膝を少し曲げる（伸びきった特異点と、膝が逆に曲がる解を避ける）
            self.d.qpos[self.qa[f"hip{sd}"]], self.d.qpos[self.qa[f"knee{sd}"]], self.d.qpos[self.qa[f"ank{sd}"]] = R(12), R(24), R(12)

    # 逆運動学: 骨盤（rx, rz, rp）を与え、足首の位置 (x, z) と足の向き（水平）を合わせる
    def ik(self, root, feet, q0=None, iters=30):
        m, d = self.m, self.d
        q = d.qpos.copy() if q0 is None else q0.copy()
        q[self.qa["rx"]], q[self.qa["rz"]], q[self.qa["rp"]] = root[0], root[1] - self.hip_z0, root[2]
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
        for _ in range(iters):
            d.qpos[:] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            err_all = 0.0
            for sd in "LR":
                tgt = feet[sd]
                p = d.site_xpos[self.site[sd]]; Rm = d.xmat[self.foot[sd]].reshape(3, 3)
                ang = np.arctan2(Rm[0, 2], Rm[0, 0])
                e = np.array([tgt[0] - p[0], tgt[1] - p[2], 0.0 - ang])
                err_all = max(err_all, np.abs(e[:2]).max())
                mujoco.mj_jacSite(m, d, jacp, jacr, self.site[sd])
                cols = [self.va[f"{j}{sd}"] for j in ("hip", "knee", "ank")]
                J = np.vstack([jacp[0, cols], jacp[2, cols], jacr[1, cols]])
                dq = np.linalg.solve(J.T @ J + 1e-6 * np.eye(3), J.T @ e)
                for k, j in enumerate(("hip", "knee", "ank")):
                    a = self.qa[f"{j}{sd}"]
                    q[a] = np.clip(q[a] + np.clip(dq[k], -0.3, 0.3), *self.lim[f"{j}{sd}"])   # 可動域の中に限る
            if err_all < 1e-7:
                break
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        ok = err_all < 1e-4 * self.c.s and all(self.lim[j][0] - 1e-6 <= q[self.qa[j]] <= self.lim[j][1] + 1e-6 for j in JOINTS)
        return q, ok, err_all

    def com_x(self, q):
        self.d.qpos[:] = q; mujoco.mj_kinematics(self.m, self.d); mujoco.mj_comPos(self.m, self.d)
        return self.d.subtree_com[1][0]

    def solve_pelvis_x(self, z, pitch, feet, com_target, x_guess, q0):
        """重心の前後位置が com_target になる骨盤の前後位置（割線法）"""
        x0, x1 = x_guess, x_guess + 0.01 * self.c.s
        q, ok, _ = self.ik((x0, z, pitch), feet, q0); f0 = self.com_x(q) - com_target
        for _ in range(12):
            q, ok, _ = self.ik((x1, z, pitch), feet, q); f1 = self.com_x(q) - com_target
            if abs(f1) < 1e-6 * self.c.s:
                break
            x0, x1, f0 = x1, x1 - f1 * (x1 - x0) / (f1 - f0 + 1e-15), f1
        return x1, q, ok

    def static_torques(self, q, support):
        """準静的なつり合い: 支持する足（support の各足、平面の 3 成分）の床反力 f を未知数に、
        骨盤の 3 自由度の一般化力が 0 になるように解き、関節トルク τ = g(q) − J_cᵀ f を返す。両足支持は最小ノルム解"""
        m, d = self.m, self.d
        d.qpos[:] = q; d.qvel[:] = 0; mujoco.mj_forward(m, d)
        gq = d.qfrc_bias.copy()
        Js = []
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
        for sd in support:
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site[sd])
            Js.append(np.vstack([jacp[0], jacp[2], jacr[1]]))
        Jc = np.vstack(Js)
        root = [self.va["rx"], self.va["rz"], self.va["rp"]]
        f = np.linalg.lstsq(Jc[:, root].T, gq[root], rcond=None)[0]
        tau = gq - Jc.T @ f
        return {j: tau[self.va[j]] for j in JOINTS}, f


def plan(c: Cfg, n_per=60):
    """準静的な段の上り方の計画: 時刻ごとの関節角（と骨盤）、支持の足、重心の目標"""
    rb = Robot(c); s = c.s; h = c.h; ah = ANK_H * s
    Hst = ah + STAND * s; XF = c.step_fwd * s; cl = CLEAR * s
    xface = XF - (HEEL + 0.03) * s - TOE * s - 0.03 * s     # つま先が段の前面の 3 cm 手前に来る足首の前方位置（1 cm では前面に触れた: F32）
    xc_c = (TOE - HEEL) / 2 * s                          # 足裏の中央（足首から +0.025 s）
    xc_l = c.xc * s if c.mode == "lean" else xc_c
    lean = R(c.lean_deg) if c.mode == "lean" else 0.0
    # 持ち上げ始め（P2 の終わり）の骨盤の高さ: 左足（床、x = 0）に届き、右足（段）にも届く
    seq = []
    q = None
    Lmax = (L_TH + L_SH) * s
    feet0 = {"L": (0.0, ah), "R": (0.0, ah)}

    def reach_z(x, feet):
        """両足に届く（股関節〜足首が脚の長さの REACH 倍以内）骨盤の高さの上限"""
        return min(fz + np.sqrt(max((REACH * Lmax)**2 - (x - fx)**2, 0.0)) for fx, fz in feet.values())

    def add(phase, u, z, pitch, feet, com, support):
        nonlocal q
        xg = seq[-1]["x"] if seq else 0.0
        q0 = q if q is not None else rb.ik((0.0, Hst, 0.0), feet0)[0]
        for _ in range(3):      # 骨盤の高さを、両足に届く高さに下げる（前後位置と交互に解く）
            zz = min(z, reach_z(xg, feet))
            xg, q, ok = rb.solve_pelvis_x(zz, pitch, feet, com, xg, q0)
        x, z = xg, zz
        seq.append(dict(phase=phase, u=u, x=x, z=z, pitch=pitch, q=q.copy(), ok=ok, feet={k: tuple(v) for k, v in feet.items()},
                        com=com, support=support))
        return ok
    U = lambda: np.linspace(0, 1, n_per + 1)[1:]
    mj = lambda x: smooth_step(x)[0]
    add("P0", 0.0, Hst, 0.0, feet0, xc_c, "LR")
    for u in U():
        add("P0", u, Hst, 0.0, feet0, xc_c, "LR")
    # P1: 右足を振り上げて段の上へ。つま先が段の前面を越えるまでは前面の手前で上げ（斜めに）、越えたら前へ、最後に下ろす。重心は左足の上
    for u in U():
        if u < 0.45:
            v = mj(u / 0.45); fr = (min(xface, XF) * v, ah + (h + cl) * v)
        elif u < 0.75:
            v = mj((u - 0.45) / 0.3); fr = (min(xface, XF) + (XF - min(xface, XF)) * v, ah + h + cl)
        else:
            v = mj((u - 0.75) / 0.25); fr = (XF, ah + h + cl * (1 - v))
        add("P1", u, Hst, 0.0, {"L": (0.0, ah), "R": fr}, xc_c, "L")
    # P2: 重心を右足の上へ移し、上体を前傾する（骨盤の高さは両足に届く高さに下げる）
    for u in U():
        v = mj(u)
        add("P2", u, Hst, lean * v, {"L": (0.0, ah), "R": (XF, ah + h)}, xc_c + (XF + xc_l - xc_c) * v, "LR")
    zC = seq[-1]["z"]
    # P3: 右足で持ち上げる。左足は 2 cm 浮かせ（荷重を抜き）、体とともに上げる。つま先が段の高さを越えるまでは前面の手前に置く
    z_top = h + Hst
    for u in U():
        v = mj(u); w = mj(min(1.0, u / 0.15))
        zl = ah + 0.02 * s * w + max(0.0, (h + cl - 0.02 * s)) * mj(np.clip((u - 0.15) / 0.85, 0, 1))
        xp = seq[-1]["x"]
        xl = min(xp - 0.05 * s, max(0.0, xface)) if zl < ah + h + cl - 1e-9 else min(xp - 0.05 * s, XF)
        xl = max(xl, seq[-1]["feet"]["L"][0])                   # 後ろには戻さない
        add("P3", u, zC + (z_top - zC) * v, lean * (1 - v), {"L": (xl, zl), "R": (XF, ah + h)},
            XF + xc_l + (xc_c - xc_l) * v, "R")
    # P4: 左足を前へ運んで段に置く
    xl0, zl0 = seq[-1]["feet"]["L"]
    for u in U():
        if u < 0.6:
            v = mj(u / 0.6); fl = (xl0 + (XF - xl0) * v, zl0 + (ah + h + cl - zl0) * v)
        else:
            v = mj((u - 0.6) / 0.4); fl = (XF, ah + h + cl * (1 - v))
        add("P4", u, z_top, 0.0, {"L": fl, "R": (XF, ah + h)}, XF + xc_c, "R" if u < 0.999 else "LR")
    return rb, seq


def quasi_static(rb, seq):
    """1 層目 (b): 計画した姿勢の列の準静的な関節トルク（最大値と、その相）"""
    peak = {j: 0.0 for j in JOINTS}; where = {}
    for k in seq:
        tq, _ = rb.static_torques(k["q"], k["support"]); k["tau"] = tq
        for j, v in tq.items():
            if abs(v) > peak[j]:
                peak[j] = abs(v); where[j] = k["phase"]
    return peak, where


def execute(c: Cfg, rb=None, seq=None, record=False):
    """2 層目: 計画した関節角の列を位置サーボの目標にして、MuJoCo で上る"""
    if seq is None:
        rb, seq = plan(c)
    m, d = rb.m, rb.d
    Tp = c.T_phase * np.sqrt(c.s)
    phases = ["P0", "P1", "P2", "P3", "P4"]
    tk = np.array([phases.index(k["phase"]) * Tp + k["u"] * Tp for k in seq])
    Q = np.array([[k["q"][rb.qa[j]] for j in JOINTS] for k in seq])
    if c.extra.get("feedforward", True):
        # 重力の補償（フィードフォワード）: 準静的なトルク τ を出すように、目標角を τ / kp だけずらす（同じアクチュエータ、同じ上限）
        kp = c.kp * c.s**3
        for i, k in enumerate(seq):
            if "tau" not in k:
                k["tau"] = rb.static_torques(k["q"], k["support"])[0]
            Q[i] += np.array([k["tau"][j] for j in JOINTS]) / kp
    act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in JOINTS]
    adr = [rb.qa[j] for j in JOINTS]
    mujoco.mj_resetData(m, d)
    d.qpos[:] = seq[0]["q"]; d.ctrl[act] = Q[0]; mujoco.mj_forward(m, d)
    # 静かに置く: 床にわずかにめり込まない高さから（足裏の下端が床に触れる）
    T_end = tk[-1] + 1.0 * np.sqrt(c.s)
    n = int(round(T_end / m.opt.timestep))
    rec = {k: [] for k in ("t", "root", "q", "tau", "feet", "sat")}
    tau_lim = np.array([(c.extra.get("tau_ank", c.tau) if j.startswith("ank") else c.tau) * c.s**3 for j in JOINTS])
    sat_time = {j: 0.0 for j in JOINTS}
    zmin_err = 0.0
    soleL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "soleL"); c6 = np.zeros(6)
    t3a, t3b = 3 * Tp + 0.2 * Tp, 4 * Tp + 0.6 * Tp          # 左足が浮いているはずの区間（P3 の 20% 〜 P4 の 60%）
    push = scrape = 0.0; Mg = m.body_subtreemass[1] * G
    floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for k in range(n):
        t = d.time
        tgt = np.array([np.interp(t, tk, Q[:, i]) for i in range(len(JOINTS))])
        d.ctrl[act] = tgt
        mujoco.mj_step(m, d)
        f = d.actuator_force[act]
        for i, j in enumerate(JOINTS):
            if abs(f[i]) >= 0.999 * tau_lim[i]:
                sat_time[j] += m.opt.timestep
        if t3a <= t <= t3b:
            for ci in range(d.ncon):
                cc = d.contact[ci]
                if soleL in (cc.geom1, cc.geom2):
                    mujoco.mj_contactForce(m, d, ci, c6)
                    if floor_id in (cc.geom1, cc.geom2):
                        push = max(push, c6[0] / Mg)          # 後ろの足が床を押した（片脚で持ち上げていない）
                    else:
                        scrape = max(scrape, c6[0] / Mg)      # 後ろの足が段の上・角に触れた
        zp = np.interp(t, tk, [kk["z"] for kk in seq])
        zmin_err = min(zmin_err, d.qpos[rb.qa["rz"]] + rb.hip_z0 - zp)
        if record and k % 8 == 0:
            rec["t"].append(t); rec["root"].append([d.qpos[rb.qa["rx"]], d.qpos[rb.qa["rz"]] + rb.hip_z0, d.qpos[rb.qa["rp"]]])
            rec["q"].append(d.qpos[adr].copy()); rec["tau"].append(f.copy())
            rec["feet"].append([d.site_xpos[rb.site[sd]][[0, 2]].copy() for sd in "LR"])
        if not np.isfinite(d.qpos).all() or d.qpos[rb.qa["rz"]] + rb.hip_z0 < 0.3 * c.s:
            break
    fin = seq[-1]
    feet_err = max(np.hypot(*(d.site_xpos[rb.site[sd]][[0, 2]] - np.array(fin["feet"][sd]))) for sd in "LR")
    z_err = d.qpos[rb.qa["rz"]] + rb.hip_z0 - fin["z"]
    pitch = D(abs(d.qpos[rb.qa["rp"]]))
    out = dict(z_err=float(z_err), pitch_deg=float(pitch), feet_err=float(feet_err), zmin_err=float(zmin_err), trail_push=float(push), trail_scrape=float(scrape),
               sat_s={j: round(v, 3) for j, v in sat_time.items() if v > 0}, finished=bool(d.time >= T_end - 2 * m.opt.timestep))
    # 成功（補助あり）: 最後に段の上で立っている。 成功（片脚だけ、Sim10 の条件）: さらに、持ち上げの間に後ろの足が床にも段（前面・角・上面）にも触れず（体重の 2% 未満）、沈みが 3 cm 未満
    out["ok_any"] = bool(out["finished"] and abs(z_err) < 0.03 * c.s and pitch < 10 and feet_err < 0.03 * c.s)
    out["ok"] = bool(out["ok_any"] and push < 0.02 and scrape < 0.02 and zmin_err > -0.03 * c.s)
    if record:
        out["rec"] = {k: np.array(v) for k, v in rec.items()}; out["tk"] = tk; out["seq"] = seq
    return out


# ---- 1 層目 (a): Sim10 の式 ----------------------------------------------------------------
def sim10_limits(s):
    """sims/sim10_steps.py の Part B の関数（biped_limits）だけを取り出して実行する（Part A・C と図は実行しない）"""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim10_steps.py"), encoding="utf-8").read()
    a = src.index("L_TH, L_SH = 0.45, 0.45"); b = src.index("h_rom, h_tau_up, knee_tau = biped_limits(lean=False)")
    ns = {"np": np, "RAD": np.radians, "D": np.degrees, "G": G}
    exec(src[a:b], ns)
    bl = ns["biped_limits"]; tk, th = ns["TAU_KNEE"] * s**3, ns["TAU_HIP"] * s**3
    hr, hu, _ = bl(s, tk, th, lean=False)
    _, hl, _ = bl(s, tk, th, lean=True)
    return dict(rom=hr, upright=min(hr, hu), lean=min(hr, hl), tau_upright=hu, tau_lean=hl)


# ---- 1 層目 (b): 計画の探索と準静的なトルク --------------------------------------------------
LEANS = [0.0, 20.0, 40.0, 60.0]
XCS = [0.025, 0.06, 0.10]               # 重心の前方位置の候補（つま先 0.15 から 5 cm 以上の余裕。0.14 では足が前に倒れた: F33）
XFS = [0.30, 0.40, 0.50]


def search(base: Cfg, n_per=20):
    """段の高さ base.h で、計画のパラメータ（上体の前傾・重心の前方位置・踏み出し）を格子で探し、
    準静的なトルクの比 r = max |τ| / τ_max が小さい順に返す（逆運動学が成り立つ計画だけ）"""
    out = []
    leans = [0.0] if base.mode == "upright" else LEANS
    for ld in leans:
        for xc in XCS:
            for xf in XFS:
                c = Cfg(**{**base.__dict__, "lean_deg": ld, "xc": xc, "step_fwd": xf})
                rb, seq = plan(c, n_per=n_per)
                if not all(k["ok"] for k in seq):
                    continue
                pk, wh = quasi_static(rb, seq)
                lim = lambda j: c.extra.get("tau_ank", c.tau) if j.startswith("ank") else c.tau
                r = max(v / (lim(j) * c.s**3) for j, v in pk.items())
                out.append(dict(lean_deg=ld, xc=xc, step_fwd=xf, r=r, peak={j: round(v, 1) for j, v in pk.items()}, where=wh))
    return sorted(out, key=lambda x: x["r"])


def try_mj(base: Cfg, cands, k=3):
    """候補の計画（r の小さい順に最大 k 個）を MuJoCo で実行し、成功したものを返す"""
    best = None
    for cand in cands[:k]:
        c = Cfg(**{**base.__dict__, "lean_deg": cand["lean_deg"], "xc": cand["xc"], "step_fwd": cand["step_fwd"]})
        ex = execute(c)
        res = dict(cand=cand, ok=ex["ok"], ok_any=ex["ok_any"], trail_push=ex["trail_push"], trail_scrape=ex["trail_scrape"], pitch=ex["pitch_deg"], zmin_err=ex["zmin_err"], sat_s=ex["sat_s"])
        if ex["ok"]:
            return res
        if best is None or (ex["ok_any"] and not best["ok_any"]):
            best = res
    return best or dict(ok=False, ok_any=False)


def limits(args):
    """(mode, s, 変える設定) について、準静的な限界（1 層目 b）と MuJoCo の限界（2 層目、片脚・補助あり）を二分法で求める"""
    mode, s, over = args[:3]
    opts = args[3] if len(args) > 3 else {}
    base = Cfg(s=s, mode=mode, **(over or {}))
    cache = {}

    def get(h):
        if h not in cache:
            b = Cfg(**{**base.__dict__, "h": h})
            cands = search(b)
            cache[h] = (b, cands)
        return cache[h]
    qs_ok = lambda h: bool(get(h)[1]) and get(h)[1][0]["r"] <= 1.0
    mj_res = {}

    def mj(h):
        if h not in mj_res:
            b, cands = get(h)
            mj_res[h] = try_mj(b, cands) if cands else dict(ok=False, ok_any=False)
        return mj_res[h]

    def bis(f, lo, hi, it=7):
        if not f(lo):
            return (np.nan, lo)
        for _ in range(it):
            mid = round((lo + hi) / 2, 5)
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo, hi)
    H = 0.8 * s; lo0 = opts.get("lo", 0.0) * s; H = opts.get("hi", 0.8) * s; it = opts.get("iters", 7)
    if opts.get("mj_only"):
        rom = qs = mj2 = (np.nan, np.nan)
        mj1 = bis(lambda h: mj(h)["ok"], lo0, H, it)
    else:
        rom = bis(lambda h: bool(get(h)[1]), lo0, H, it)
        qs = bis(qs_ok, lo0, H, it)
        mj1 = bis(lambda h: mj(h)["ok"], lo0, H, it)
        mj2 = bis(lambda h: mj(h)["ok_any"], lo0, H, it)
    pick = lambda h: get(h)[1][0] if get(h)[1] else None
    return dict(mode=mode, s=s, over=over, rom=rom, qs=qs, mj=mj1, mj_any=mj2,
                qs_plan=pick(qs[0]) if np.isfinite(qs[0]) else None,
                mj_plan=mj(mj1[0]).get("cand") if np.isfinite(mj1[0]) else None,
                mj_detail={str(k): {kk: vv for kk, vv in v.items() if kk != "cand"} for k, v in mj_res.items()})


def figure():
    """図: (a) 上体を前傾して上る姿（MuJoCo の限界の高さ）、(b) 膝のトルク（準静的と MuJoCo）、(c) 限界の高さ / s と s"""
    res = json.load(open(DATA / "field20_biped_steps.json"))
    jl = {(r["mode"], r["s"]): r for r in res["jobs"]}
    fig = plt.figure(figsize=(15, 4.8))
    ax0 = fig.add_subplot(1, 3, 1); ax1 = fig.add_subplot(1, 3, 2); ax2 = fig.add_subplot(1, 3, 3)
    r1 = jl[("lean", 1.0)]; pl = r1["mj_plan"]
    h = r1["mj"][0]
    c = Cfg(h=h, mode="lean", lean_deg=pl["lean_deg"], xc=pl["xc"], step_fwd=pl["step_fwd"])
    rb, seq = plan(c); quasi_static(rb, seq)
    ex = execute(c, rb, seq, record=True)
    m, d = rb.m, rb.d; Tp = c.T_phase
    ax0.add_patch(plt.Rectangle(((c.step_fwd - HEEL - 0.03), 0), 1.0, h, color="0.8"))
    ax0.axhline(0, color="k", lw=0.8)
    names = ["trunk", "thighL", "shankL", "footL", "thighR", "shankR", "footR"]
    ids = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in names}
    for tt, col in [(1.0 * Tp, "0.6"), (2.0 * Tp, "tab:blue"), (3.0 * Tp, "tab:orange"), (3.4 * Tp, "tab:red"), (5.0 * Tp, "k")]:
        i = np.argmin(abs(ex["rec"]["t"] - tt))
        d.qpos[[rb.qa[j] for j in JOINTS]] = ex["rec"]["q"][i]
        d.qpos[rb.qa["rx"]], d.qpos[rb.qa["rz"]], d.qpos[rb.qa["rp"]] = ex["rec"]["root"][i, 0], ex["rec"]["root"][i, 1] - rb.hip_z0, ex["rec"]["root"][i, 2]
        mujoco.mj_kinematics(m, d)
        P = {n: d.xpos[ids[n]][[0, 2]] for n in names}
        top = P["trunk"] + 0.6 * np.array([np.sin(d.qpos[rb.qa["rp"]]), np.cos(d.qpos[rb.qa["rp"]])])
        ax0.plot(*np.array([top, P["trunk"]]).T, color=col, lw=4)
        for sd, ls in [("L", "--"), ("R", "-")]:
            ax0.plot(*np.array([P["trunk"], P[f"shank{sd}"], P[f"foot{sd}"]]).T, color=col, ls=ls, lw=2)
            ax0.plot(*np.array([P[f"foot{sd}"] + [-HEEL, -ANK_H], P[f"foot{sd}"] + [TOE, -ANK_H]]).T, color=col, ls=ls, lw=2)
        com = d.subtree_com[1] if False else None
        ax0.text(top[0] + (0.12 if tt == 1.0 * Tp else 0), top[1] + 0.03, f"t={tt:.1f}s", color=col, fontsize=7, ha="center")
    ax0.set_aspect("equal"); ax0.set_xlim(-0.5, 1.1); ax0.set_ylim(-0.05, 2.1)
    ax0.set_title(f"lean strategy, step {h:.2f} m (MuJoCo limit, s = 1)\nsolid: step leg, dashed: trailing leg", fontsize=9)
    st = np.array([k["tau"]["kneeR"] for k in seq])
    ax1.plot(ex["tk"], -st, "k--", label="quasi-static (plan)")
    ax1.plot(ex["rec"]["t"], -ex["rec"]["tau"][:, JOINTS.index("kneeR")], "tab:red", label="MuJoCo: step-leg knee")
    ax1.plot(ex["rec"]["t"], -ex["rec"]["tau"][:, JOINTS.index("hipR")], "tab:blue", lw=1, label="MuJoCo: step-leg hip")
    ax1.axhline(TAU, color="k", lw=0.8, ls=":"); ax1.text(10.5, TAU - 18, "limit\n250", fontsize=7)
    for k_, nm in enumerate(["P0 stand", "P1 swing", "P2 shift", "P3 lift", "P4 place"]):
        ax1.axvline(k_ * Tp, color="0.85", lw=0.8); ax1.text(k_ * Tp + 0.05, -60, nm, fontsize=7)
    ax1.set_xlabel("t [s]"); ax1.set_ylabel("extension torque [N·m]"); ax1.legend(fontsize=7, loc="lower left")
    ax1.set_title("knee / hip torque", fontsize=9)
    S = [0.5, 1.0, 2.0]
    for mode, col in [("upright", "tab:gray"), ("lean", "tab:red")]:
        ax2.plot(S, [res["sim10"][str(s_)][mode] / s_ for s_ in S], "o:", color=col, label=f"Sim10 formula ({mode})")
        ax2.plot(S, [jl[(mode, s_)]["qs"][0] / s_ for s_ in S], "s--", color=col, mfc="none", label=f"quasi-static plan ({mode})")
        ax2.plot(S, [jl[(mode, s_)]["mj"][0] / s_ for s_ in S], "*-", color=col, ms=10, label=f"MuJoCo, step leg only ({mode})")
    ax2.plot(S, [jl[("lean", s_)]["rom"][0] / s_ for s_ in S], "k^-", lw=0.8, label="range of motion (plan exists)")
    ax2.set_xscale("log"); ax2.minorticks_off(); ax2.set_xticks(S); ax2.set_xticklabels(["0.5", "1", "2"])
    ax2.annotate("s = 2: no step-up plan within\ntorque limit even at h = 0", (2.0, 0.02), (1.15, 0.03), fontsize=7,
                 arrowprops=dict(arrowstyle="->", lw=0.6))
    ax2.set_xlabel("scale s (torque limit ∝ s³)"); ax2.set_ylabel("max step height / s [m]")
    ax2.set_title("step height limit vs scale", fontsize=9); ax2.legend(fontsize=6, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "field20_biped_steps.png", dpi=130)
    print("図 → figures/field/field20_biped_steps.png")


def main():
    from multiprocessing import Pool
    jobs = [(m_, s_, None) for s_ in (1.0, 0.5, 2.0) for m_ in ("upright", "lean")]
    with Pool(4) as pool:
        res = pool.map(limits, jobs)
    out = {"jobs": res, "sim10": {str(s_): sim10_limits(s_) for s_ in (0.5, 1.0, 2.0)}}
    print("| s | 上体 | Sim10 の式 | 可動域（計画が作れる） | 準静的（計画の姿勢、1 層目 b） | MuJoCo 片脚だけ | MuJoCo 後ろ足の補助あり |")
    for r in res:
        s10 = out["sim10"][str(r["s"])][r["mode"]]
        f = lambda x: f"{x[0]:.3f}" if np.isfinite(x[0]) else "—"
        print(f"| {r['s']:g} | {r['mode']} | {s10:.2f} | {f(r['rom'])} | {f(r['qs'])} | {f(r['mj'])} | {f(r['mj_any'])} |", flush=True)
        print(f"    準静的の計画 {r['qs_plan']}\n    MuJoCo の計画 {r['mj_plan']}", flush=True)
    json.dump(out, open(DATA / "field20_biped_steps.json", "w"), indent=1, ensure_ascii=False, default=float)
    print(f"[{time.time() - T_START:.0f} s]")


VARIANTS = {   # 名前: Cfg の上書き（上体の前傾、s = 1）
    "基準": {},
    "計算: 時間刻み 0.5 ms": {"dt": 5e-4},
    "計算: 時間刻み 0.1 ms": {"dt": 1e-4},
    "系統: 速く（1 相 1 s）": {"T_phase": 1.0},
    "系統: ゆっくり（1 相 4 s）": {"T_phase": 4.0},
    "系統: サーボ kp 1e4": {"kp": 1e4},
    "系統: サーボ kp 1e5": {"kp": 1e5},
    "系統: 重力の補償なし": {"extra": {"feedforward": False}},
    "系統: 接触を柔らかく（solref 0.02）": {"solref": "0.02 1"},
    "系統: 摩擦 μ 0.4": {"mu": 0.4},
    "系統: 回転子の慣性 0.005 / 0.5": None,
    "系統: 足首を強く（1000 N·m）": {"extra": {"tau_ank": 1000.0}},
}


def errors_main():
    from multiprocessing import Pool
    names = [k for k in VARIANTS if VARIANTS[k] is not None]
    jobs = [("lean", 1.0, VARIANTS[k], {"mj_only": True, "lo": 0.2, "hi": 0.6, "iters": 6}) for k in names]
    names += ["系統: 回転子の慣性 0.005", "系統: 回転子の慣性 0.5"]
    jobs += [("lean", 1.0, {"armature": 0.005}, {"mj_only": True, "lo": 0.2, "hi": 0.6, "iters": 6}),
             ("lean", 1.0, {"armature": 0.5}, {"mj_only": True, "lo": 0.2, "hi": 0.6, "iters": 6})]
    rng = np.random.default_rng(0)
    mc = [{"trunk": 1 + rng.uniform(-.1, .1), "thigh": 1 + rng.uniform(-.1, .1), "shank": 1 + rng.uniform(-.1, .1),
           "foot": 1 + rng.uniform(-.15, .15)} for _ in range(16)]
    jobs_mc = [("lean", 1.0, {"extra": {"mass": m_}}, {"mj_only": True, "lo": 0.2, "hi": 0.6, "iters": 5}) for m_ in mc]
    with Pool(4) as pool:
        res = pool.map(limits, jobs + jobs_mc)
    out = {}
    print("=== 誤差の評価：上体の前傾、s = 1、MuJoCo（片脚だけ）の限界 ===")
    for k, r in zip(names, res[:len(names)]):
        out[k] = r
        print(f"| {k} | {r['mj'][0]:.3f}（{r['mj'][0]:.3f}〜{r['mj'][1]:.3f}） | {r['mj_plan']} |", flush=True)
    hs = np.array([r["mj"][0] for r in res[len(names):]])
    out["mc"] = {"masses": mc, "h": hs.tolist()}
    print(f"入力（部品表の質量、16 回）: {np.nanmean(hs):.3f} ± {np.nanstd(hs, ddof=1):.3f} m（{100 * np.nanstd(hs, ddof=1) / np.nanmean(hs):.1f}%）")
    json.dump(out, open(DATA / "field20_errors.json", "w"), indent=1, ensure_ascii=False, default=float)


if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else "main"
    {"main": main, "errors": errors_main, "figure": lambda: figure()}[arg]()
