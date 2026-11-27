"""Field18: 宇宙の脚の円錐 AMBAC を MuJoCo で照合する（下半身の段 1、2 層目の照合）.

Sim09 Part A2 の 3 次元の結果（式のモデル）
  ・腕（肩）  : 前下方 30° の軸、半頂角 30° → 24.8°/周
  ・脚（股関節）: 前下方 30° の軸、半頂角 30° → 48.2°/周（腕の 1.9 倍）、半頂角 45° → 96.8°/周
を、自由に浮かぶ胴体＋2 軸の股関節（肩）を持つ MuJoCo の機体で確かめる。

照合の層
  1 層目 (a) Sim09 の式のモデル（ambac.spatial、40 点の点質量、球面座標、中点則 3000 歩）
         (b) 同じ物理を、方向ベクトルで書き直して高精度の常微分方程式（四元数, rtol 1e-12）で解いたもの
  2 層目 MuJoCo（自由関節の胴体、円錐の軸に垂直な 2 本のヒンジ = 自在継手、位置サーボで追従）
         ・MuJoCo が実際にたどった脚の向きを 1 層目に入れ直した値とも比べる（サーボの追従誤差に左右されない比較）
誤差の評価
  計算誤差: MuJoCo の時間刻み・積分法（RK4 / implicitfast / Euler）、1 層目の積分の歩数と点質量の数
  系統誤差: 脚の太さ（軸まわりの慣性）、棒の離散化、速さの曲線（幾何学的な量なので変わらないはず）、サーボの硬さ
  入力の不確かさ: 脚の質量 ±15%、胴体の慣性 ±10%（段階 4 の答えを隠した仮想の実機と同じ幅）のモンテカルロ
可動域・干渉: 股関節の屈曲・伸展・外転・内転の角度と、もう 1 本の脚との距離を円錐の上で調べる
"""
import os
os.environ.setdefault("MUJOCO_GL", "disable")
import json
import time

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from common import DATA, FIG, smooth_step
from ambac.spatial import Arm3D, FloatingBody3D, sph
from sim07_helpers import cone_holonomy

D = np.degrees
RAD = np.radians
OUT = FIG / "field"; OUT.mkdir(exist_ok=True)
t_start = time.time()

# ---- 機体（Sim09 Part A2 と同じ） ----------------------------------------------------
TORSO_DIMS = (0.25, 0.4, 0.8)
ARM = dict(shoulder=(0.0, -0.2, 0.30), mass=4.0, length=0.7)
LEG = dict(shoulder=(0.0, -0.1, -0.40), mass=12.0, length=0.9)
OTHER_LEG_HIP = np.array([0.0, +0.1, -0.40])          # 固定したもう 1 本の脚（干渉の確認用、真下に伸ばす）
LIMB_R = 0.06                                         # 脚の半径（干渉と軸まわりの慣性の系統誤差に使う）
AXIS = sph(0, RAD(-60))                               # 前下方 30°（真下から前へ 30°）


def body_of(limb, base_mass, n_pts=40):
    return FloatingBody3D(base_mass, TORSO_DIMS, Arm3D(**limb, n_pts=n_pts))


CASES = {   # name: (limb, base_mass（胴体＋固定した手足）, 半頂角)
    "arm_b30": (ARM, 46.0 + 16.0 + 12.0, RAD(30)),
    "leg_b30": (LEG, 46.0 + 16.0 + 4.0, RAD(30)),
    "leg_b45": (LEG, 46.0 + 16.0 + 4.0, RAD(45)),
}
SIM09 = {"arm_b30": 24.8, "leg_b30": 48.2, "leg_b45": 96.8}     # 報告書の値（表 tab:legspace）


def frame(a):
    a = np.asarray(a, float) / np.linalg.norm(a)
    t_ = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(t_, a); e1 /= np.linalg.norm(e1)
    return a, e1, np.cross(a, e1)


def cone_dir(beta, phase, a=AXIS):
    """円錐の上の肢の向き（phase: 0→2π）。Sim07 の cone_holonomy と同じ向き・同じ始点"""
    a, e1, e2 = frame(a)
    phase = np.asarray(phase, float)[..., None]
    return np.cos(beta) * a + np.sin(beta) * (np.cos(phase) * e1 + np.sin(phase) * e2)


# ---- 1 層目 (b): 方向ベクトルで書いた式を高精度で解く --------------------------------------
def omega_dir(body, u, ud, axial=0.0, spin=0.0):
    """胴体座標の角速度（系の角運動量 0）。u: 肢の向き, ud: その時間微分。
    axial: 肢の軸まわりの慣性（太さ）、spin: 胴体に対する肢の軸まわりの回転の速さ（継手の形で決まる）"""
    arm = body.arm
    d = (np.arange(arm.n_pts) + 0.5) / arm.n_pts * arm.length
    m = np.full(arm.n_pts, arm.mass / arm.n_pts)
    P = np.vstack([np.zeros(3), np.array(arm.shoulder) + d[:, None] * u])
    V = np.vstack([np.zeros(3), d[:, None] * ud])
    M = np.r_[body.base_mass, m]
    c = (M[:, None] * P).sum(0) / M.sum(); vc = (M[:, None] * V).sum(0) / M.sum()
    r = P - c; v = V - vc
    A = body.I_base + np.einsum("i,ij->", M, r * r) * np.eye(3) - np.einsum("i,ij,ik->jk", M, r, r)
    L = np.einsum("i,ij->j", M, np.cross(r, v))
    if axial:   # 軸まわりの角運動量 axial·(u·ω + spin)·u
        A = A + axial * np.outer(u, u); L = L + axial * spin * u
    return np.linalg.solve(A, -L)


def phase_profile(T, kind="minjerk"):
    if kind == "uniform":
        return (lambda t: 2 * np.pi * t / T), (lambda t: 2 * np.pi / T)
    return (lambda t: 2 * np.pi * smooth_step(t / T)[0]), (lambda t: 2 * np.pi * smooth_step(t / T)[1] / T)


def holonomy_ode(body, beta, T=2.0, kind="minjerk", axial=0.0, a=AXIS, rtol=1e-12):
    ph, phd = phase_profile(T, kind)
    _, e1, e2 = frame(a)

    def rhs(t, y):
        p = ph(t); pd = phd(t)
        u = cone_dir(beta, p, a)
        ud = np.sin(beta) * pd * (-np.sin(p) * e1 + np.cos(p) * e2)
        spin = 0.0
        if axial:   # 自在継手（局所 z → 局所 y）のねじれ: spin = j1' · u_local,z
            a_, b_, c_ = frame(a); ul = np.array([u @ a_, u @ b_, u @ c_]); uld = np.array([ud @ a_, ud @ b_, ud @ c_])
            spin = (ul[0] * uld[1] - ul[1] * uld[0]) / (ul[0]**2 + ul[1]**2) * ul[2]
        w = omega_dir(body, u, ud, axial, spin)
        qx, qy, qz, qw = y                                     # 胴体座標の角速度: q' = q ⊗ (0, w)/2
        return 0.5 * np.array([qw * w[0] + qy * w[2] - qz * w[1], qw * w[1] + qz * w[0] - qx * w[2],
                               qw * w[2] + qx * w[1] - qy * w[0], -qx * w[0] - qy * w[1] - qz * w[2]])
    sol = solve_ivp(rhs, (0, T), [0, 0, 0, 1.0], method="DOP853", rtol=rtol, atol=rtol * 1e-2)
    return Rotation.from_quat(sol.y[:, -1] / np.linalg.norm(sol.y[:, -1])).as_rotvec()


def holonomy_samples(body, us, dt, axial=0.0, js=None):
    """記録された肢の向きの列（胴体座標）から、中点則で胴体の回転を積分する（js: 継手の角、ねじれの計算用）"""
    R = Rotation.identity()
    for k, (a_, b_) in enumerate(zip(us[:-1], us[1:])):
        um = a_ + b_; um /= np.linalg.norm(um)
        spin = 0.0
        if axial:
            spin = (js[k + 1, 0] - js[k, 0]) / dt * (-np.sin((js[k + 1, 1] + js[k, 1]) / 2))
        R = R * Rotation.from_rotvec(omega_dir(body, um, (b_ - a_) / dt, axial, spin) * dt)
    return R.as_rotvec()


# ---- 2 層目: MuJoCo ------------------------------------------------------------------
def mjcf(body, a=AXIS, timestep=2.5e-4, integrator="RK4", kp=2.0e5, kv=None, axial=1e-7, n_rod=None):
    """胴体（自由関節）＋自在継手の肢。肢の局所 x 軸 = 円錐の軸、ヒンジは局所 z → 局所 y の順"""
    arm = body.arm; L = arm.length; m = arm.mass
    n = arm.n_pts if n_rod is None else n_rod
    I_perp = m * L**2 / 12 * (1 - 1 / n**2) if np.isfinite(n) else m * L**2 / 12   # 点質量の列と同じ慣性
    Ib = np.diag(body.I_base)
    a, e1, _ = frame(a)
    I_hip = m * L**2 / 3
    kv = 2 * np.sqrt(kp * I_hip) if kv is None else kv
    sx, sy, sz = arm.shoulder; bx, by, bz = TORSO_DIMS
    return f"""<mujoco model="leg_cone">
  <option gravity="0 0 0" timestep="{timestep}" integrator="{integrator}"><flag contact="disable"/></option>
  <worldbody>
    <body name="base">
      <freejoint name="base"/>
      <inertial pos="0 0 0" mass="{body.base_mass}" diaginertia="{Ib[0]} {Ib[1]} {Ib[2]}"/>
      <geom type="box" size="{bx/2} {by/2} {bz/2}" rgba="0.7 0.7 0.7 1"/>
      <body name="limb" pos="{sx} {sy} {sz}" xyaxes="{a[0]} {a[1]} {a[2]} {e1[0]} {e1[1]} {e1[2]}">
        <joint name="j1" type="hinge" axis="0 0 1" damping="0" armature="0"/>
        <joint name="j2" type="hinge" axis="0 1 0" damping="0" armature="0"/>
        <inertial pos="{L/2} 0 0" mass="{m}" diaginertia="{axial} {I_perp} {I_perp}"/>
        <geom type="capsule" fromto="0 0 0 {L} 0 0" size="{LIMB_R}" rgba="0.3 0.5 0.9 1"/>
      </body>
    </body>
  </worldbody>
  <actuator><position name="j1" joint="j1" kp="{kp}" kv="{kv}"/><position name="j2" joint="j2" kp="{kp}" kv="{kv}"/></actuator>
</mujoco>"""


def ik(u_base, a=AXIS):
    """胴体座標の肢の向き → 2 本のヒンジの角（u_local = Rz(j1) Ry(j2) x̂ = (c2 c1, c2 s1, −s2)）"""
    a, e1, e2 = frame(a)
    ul = np.array([u_base @ a, u_base @ e1, u_base @ e2])
    return np.array([np.arctan2(ul[1], ul[0]), -np.arcsin(np.clip(ul[2], -1, 1))])


def run_mj(body, beta, T=2.0, settle=0.5, kind="minjerk", record=True, a=AXIS, **kw):
    m = mujoco.MjModel.from_xml_string(mjcf(body, a=a, **kw)); d = mujoco.MjData(m)
    ph, _ = phase_profile(T, kind)
    ref = lambda t: ik(cone_dir(beta, ph(min(t, T)), a), a)
    d.qpos[7:9] = ref(0.0); mujoco.mj_forward(m, d)
    bid = m.body("limb").id; base = m.body("base").id
    us, err, L, js = [], [], [], []
    n = int(round((T + settle) / m.opt.timestep))
    for k in range(n + 1):
        if record:
            Rb = d.xmat[base].reshape(3, 3); Rl = d.xmat[bid].reshape(3, 3)
            us.append(Rb.T @ Rl[:, 0]); js.append(d.qpos[7:9].copy())
            err.append(np.abs(d.qpos[7:9] - ref(d.time)).max())
            mujoco.mj_subtreeVel(m, d); L.append(np.linalg.norm(d.subtree_angmom[0]))
        d.ctrl[:] = ref(d.time)
        mujoco.mj_step(m, d)
    w, x, y, z = d.qpos[3:7]
    rv = Rotation.from_quat([x, y, z, w]).as_rotvec()
    return dict(rv=rv, us=np.array(us), js=np.array(js), track_err=float(np.max(err)) if err else np.nan,
                L_max=float(np.max(L)) if L else np.nan, dt=m.opt.timestep, j_final=d.qpos[7:9].copy())


res = {"cases": {}}
lines = []


def log(s):
    print(s); lines.append(s)


# ---- 0. 自在継手の向きの確認（IK と MuJoCo の姿勢が一致するか） ---------------------------------
b0 = body_of(LEG, 66.0)
m0 = mujoco.MjModel.from_xml_string(mjcf(b0)); d0 = mujoco.MjData(m0)
chk = 0.0
for p in np.linspace(0, 2 * np.pi, 13):
    u = cone_dir(RAD(45), p); d0.qpos[7:9] = ik(u); mujoco.mj_forward(m0, d0)
    chk = max(chk, np.linalg.norm(d0.xmat[m0.body("limb").id].reshape(3, 3)[:, 0] - u))
log(f"[確認] 自在継手の IK と MuJoCo の肢の向きの差 max {chk:.1e}")

# ---- 1. 主結果: 1 層目 (a)(b) と 2 層目 -------------------------------------------------------
log("\n=== 1 周あたりの胴体の回転 [deg]（前下方 30° の軸） ===")
log(f"{'case':8s} {'報告値':>7s} {'(a)Sim09式':>11s} {'(b)ODE':>10s} {'MuJoCo':>10s} {'MJ経路→式':>10s}  "
    f"{'MJ−(b)':>9s} {'追従誤差':>8s} {'|L|max':>8s}  回転軸（胴体座標）")
main_runs = {}
for name, (limb, mb, beta) in CASES.items():
    body = body_of(limb, mb)
    rv_a = cone_holonomy(body, AXIS, beta)                       # Sim09 と同じ計算（一様な速さ）
    rv_b = holonomy_ode(body, beta)
    o = run_mj(body, beta)
    rv_r = holonomy_samples(body, o["us"], o["dt"])
    main_runs[name] = o
    ang = lambda v: D(np.linalg.norm(v))
    ax_ = o["rv"] / np.linalg.norm(o["rv"])
    res["cases"][name] = dict(report=SIM09[name], sim09_formula=ang(rv_a), ode=ang(rv_b), mujoco=ang(o["rv"]),
                              mujoco_path_formula=ang(rv_r), mj_minus_ode=ang(o["rv"]) - ang(rv_b),
                              axis_angle_mj_vs_ode=D(np.arccos(np.clip(ax_ @ rv_b / np.linalg.norm(rv_b), -1, 1))),
                              rv_mujoco=o["rv"].tolist(), rv_ode=rv_b.tolist(), track_err_deg=D(o["track_err"]),
                              L_max=o["L_max"], residual_joint_deg=D(np.abs(o["j_final"] - ik(cone_dir(beta, 0))).max()))
    c = res["cases"][name]
    log(f"{name:8s} {c['report']:7.1f} {c['sim09_formula']:11.4f} {c['ode']:10.4f} {c['mujoco']:10.4f} {c['mujoco_path_formula']:10.4f}  "
        f"{c['mj_minus_ode']:+9.1e} {c['track_err_deg']:7.2f}° {c['L_max']:8.1e}  {np.round(ax_, 3)}")
    c["mj_minus_path"] = ang(o["rv"]) - ang(rv_r)
    log(f"         MuJoCo − (MuJoCo の経路を式に入れた値) = {c['mj_minus_path']:+.2e}°, 回転軸の向きの差（MJ と (b)）{c['axis_angle_mj_vs_ode']:.3f}°")
ratio = res["cases"]["leg_b30"]["mujoco"] / res["cases"]["arm_b30"]["mujoco"]
res["leg_over_arm_mujoco"] = ratio
log(f"脚 / 腕（半頂角 30°、MuJoCo）= {ratio:.3f}（Sim09: 1.9）")

# ---- 2. 計算誤差 ------------------------------------------------------------------------------
log("\n=== 計算誤差: MuJoCo の積分法と時間刻み（leg_b30, kp 2e5）。追従誤差を除くため、MuJoCo の経路を式に入れた値とも比べる ===")
ref_b30 = res["cases"]["leg_b30"]["ode"]
body_l = body_of(LEG, 66.0)
comp = []
for integ in ("RK4", "implicitfast", "Euler"):
    for dt in (2e-3, 1e-3, 5e-4, 2.5e-4, 1e-4):
        o = run_mj(body_l, RAD(30), timestep=dt, integrator=integ)
        v = D(np.linalg.norm(o["rv"])); e = v - ref_b30
        ok = o["track_err"] < RAD(10)
        e_path = v - D(np.linalg.norm(holonomy_samples(body_l, o["us"], o["dt"]))) if ok else np.nan
        comp.append(dict(integrator=integ, dt=dt, value=v, err_deg=e, rel=e / ref_b30, mj_minus_path=e_path,
                         rel_path=e_path / ref_b30, L_max=o["L_max"], track_err_deg=D(o["track_err"]), stable=bool(ok)))
        log(f"  {integ:12s} dt={dt*1e3:5.2f} ms: {v:10.5f}°  (b) との差 {e / ref_b30 * 100:+9.4f}%, "
            + (f"MJ経路→式との差 {e_path / ref_b30 * 100:+.2e}%, |L|max {o['L_max']:.1e}, 追従誤差 {D(o['track_err']):.2f}°" if ok
               else f"サーボが発振（追従誤差 {D(o['track_err']):.0f}°）"))
res["comp_mujoco"] = comp

log("\n=== 計算誤差: 1 層目 (a) の歩数・点質量の数（leg_b30） ===")
comp1 = []
for n in (500, 1000, 3000, 10000):
    v = D(np.linalg.norm(cone_holonomy(body_l, AXIS, RAD(30), n=n)))
    comp1.append(dict(kind="steps", n=n, value=v)); log(f"  中点則 {n:6d} 歩: {v:.5f}°（(b) との差 {v - ref_b30:+.1e}°）")
for npt in (10, 40, 160, 640):
    v = D(np.linalg.norm(holonomy_ode(body_of(LEG, 66.0, n_pts=npt), RAD(30), rtol=1e-10)))
    comp1.append(dict(kind="n_pts", n=npt, value=v)); log(f"  点質量 {npt:4d} 個: {v:.5f}°（40 個との差 {v - ref_b30:+.1e}°）")
res["comp_formula"] = comp1

# ---- 3. 系統誤差（モデルの仮定を 1 つずつ変える） --------------------------------------------------
log("\n=== 系統誤差: 仮定を 1 つずつ変える（leg_b30、MuJoCo RK4 0.25 ms と 1 層目 (b)） ===")
syst = []
I_ax = LEG["mass"] * LIMB_R**2 / 2
for label, kwmj, kwode in (
        ("基準（細い棒, 最小ジャーク 2 s）", {}, {}),
        (f"脚の太さ（半径 {LIMB_R} m, 軸まわり {I_ax:.3f} kg·m²）", dict(axial=I_ax), dict(axial=I_ax)),
        ("連続な棒（点質量の列でなく）", dict(n_rod=np.inf), None),
        ("一様な速さ（Sim09 と同じ, 始めと終わりで急に動く）", dict(kind="uniform"), dict(kind="uniform")),
        ("ゆっくり 1 周 8 s", dict(T=8.0), dict(T=8.0)),
        ("速く 1 周 0.5 s", dict(T=0.5), dict(T=0.5)),
        ("サーボ kp 2e6（10 倍硬く）", dict(kp=2.0e6), None),
        ("サーボ kp 2e4（1/10）", dict(kp=2.0e4), None),
        ("サーボ kp 2e3（1/100）", dict(kp=2.0e3), None),
        ("サーボ kp 2e2（1/1000）", dict(kp=2.0e2), None)):
    T_ = kwmj.pop("T", 2.0); kind = kwmj.pop("kind", "minjerk")
    o = run_mj(body_l, RAD(30), T=T_, kind=kind, settle=0.5 if kwmj.get("kp", 2e5) > 1e3 else 3.0, **kwmj)
    v_mj = D(np.linalg.norm(o["rv"]))
    v_ode = D(np.linalg.norm(holonomy_ode(body_l, RAD(30), **kwode))) if kwode is not None else np.nan
    ax_in = kwmj.get("axial", 0.0) if kwmj.get("axial", 0) > 1e-6 else 0.0
    v_path = D(np.linalg.norm(holonomy_samples(body_l, o["us"], o["dt"], ax_in, o["js"])))
    syst.append(dict(label=label, mujoco=v_mj, ode=v_ode, path_formula=v_path, shift_pct=(v_mj / ref_b30 - 1) * 100,
                     mj_minus_path=v_mj - v_path, track_err_deg=D(o["track_err"])))
    log(f"  {label:30s}: MuJoCo {v_mj:.4f}°（{(v_mj / ref_b30 - 1) * 100:+.3f}%）, 式 {v_ode:.4f}°, "
        f"MJ経路→式 {v_path:.4f}°（MJ との差 {v_mj - v_path:+.1e}°）, 追従誤差 {D(o['track_err']):.2f}°")
res["systematic"] = syst

# ---- 4. 入力（部品表）の不確かさ: 脚の質量 ±15%、胴体の慣性 ±10% -------------------------------------
log("\n=== 入力の不確かさの伝搬（leg_b30, 1 層目 (b), 一様分布, 200 回） ===")
rng = np.random.default_rng(18)


class ScaledBody(FloatingBody3D):
    def __init__(self, *a, inertia_scale=1.0):
        super().__init__(*a); self.inertia_scale = inertia_scale

    @property
    def I_base(self):
        return super().I_base * self.inertia_scale


mc = []
for k in range(200):
    fl = rng.uniform(0.85, 1.15); fi = rng.uniform(0.9, 1.1)
    bb = ScaledBody(66.0 + (1 - fl) * 12.0, TORSO_DIMS, Arm3D(**dict(LEG, mass=12.0 * fl)), inertia_scale=fi)
    mc.append((fl, fi, D(np.linalg.norm(holonomy_ode(bb, RAD(30), rtol=1e-8)))))
mc = np.array(mc)
J = np.linalg.lstsq(np.c_[np.ones(len(mc)), mc[:, 0] - 1, mc[:, 1] - 1], mc[:, 2], rcond=None)[0]
res["input_mc"] = dict(mean=float(mc[:, 2].mean()), sd=float(mc[:, 2].std(ddof=1)), sd_pct=float(mc[:, 2].std(ddof=1) / ref_b30 * 100),
                       min=float(mc[:, 2].min()), max=float(mc[:, 2].max()),
                       dtheta_per_unit_leg=float(J[1]), dtheta_per_unit_inertia=float(J[2]))
log(f"  {mc[:, 2].mean():.2f} ± {mc[:, 2].std(ddof=1):.2f}°（{res['input_mc']['sd_pct']:.1f}%）, 範囲 {mc[:, 2].min():.1f}〜{mc[:, 2].max():.1f}°;"
    f" 感度: 脚の質量 +1% → {J[1] / 100:+.3f}°, 胴体の慣性 +1% → {J[2] / 100:+.3f}°")

# ---- 5. 可動域と、もう 1 本の脚との干渉 ---------------------------------------------------------
log("\n=== 股関節の可動域と、もう 1 本の脚との干渉（前下方 30° の軸） ===")


def seg_dist(p0, p1, q0, q1, n=60):
    s = np.linspace(0, 1, n)[:, None]
    P = p0 + s * (p1 - p0); Q = q0 + s * (q1 - q0)
    return np.min(np.linalg.norm(P[:, None] - Q[None], axis=-1))


def torso_pen(shoulder, L, u, r=0.04):
    """肢（半径 r、付け根から 0.15 m より先）が胴体の箱にめり込む深さ [m]（負ならめり込まない）"""
    half = np.array(TORSO_DIMS) / 2 + r
    pts = np.array(shoulder) + np.linspace(0.15, L, 40)[:, None] * u
    return (half - np.abs(pts)).min(axis=1).max()


def hip_angles(u):
    """股関節の角度: 屈曲（前、+）・伸展（後ろ、−）は矢状面、外転（外 = −y、+）・内転（−）は前額面"""
    down = -u[2]
    return D(np.arctan2(u[0], down)), D(np.arctan2(-u[1], down))


rom = {}
other0, other1 = OTHER_LEG_HIP, OTHER_LEG_HIP + np.array([0, 0, -LEG["length"]])
hip = np.array(LEG["shoulder"])
for beta_deg in (30, 45):
    for tilt_deg in (0, 20, 30):   # 円錐の軸を外へ傾けて内転と干渉を避ける（可動域の形の系統誤差）
        a = np.array([np.sin(RAD(30)), -np.tan(RAD(tilt_deg)) * np.cos(RAD(30)), -np.cos(RAD(30))]); a /= np.linalg.norm(a)
        ph = np.linspace(0, 2 * np.pi, 361)
        us = cone_dir(RAD(beta_deg), ph, a)
        fl, ab = np.array([hip_angles(u) for u in us]).T
        dist = np.array([seg_dist(hip, hip + LEG["length"] * u, other0, other1) for u in us])
        body = body_of(LEG, 66.0)
        v = D(np.linalg.norm(holonomy_ode(body, RAD(beta_deg), a=a, rtol=1e-9)))
        key = f"b{beta_deg}_tilt{tilt_deg}"
        rom[key] = dict(flex_max=fl.max(), ext_max=-fl.min(), abd_max=ab.max(), add_max=-ab.min(),
                        min_gap=dist.min() - 2 * LIMB_R, collide_frac=float(np.mean(dist < 2 * LIMB_R)), deg_per_cycle=v)
        r_ = rom[key]
        ok = "可動域内" if (r_["flex_max"] <= 120 and r_["ext_max"] <= 20 and r_["abd_max"] <= 45 and r_["add_max"] <= 30) else "可動域外"
        log(f"  半頂角 {beta_deg}°, 軸を外へ {tilt_deg:2d}°: 屈曲 {r_['flex_max']:5.1f}° 伸展 {r_['ext_max']:5.1f}° 外転 {r_['abd_max']:5.1f}° "
            f"内転 {r_['add_max']:5.1f}°（{ok}）, 脚どうしの最小すき間 {r_['min_gap'] * 100:+6.1f} cm"
            f"（干渉 {r_['collide_frac'] * 100:4.0f}% の区間）→ {v:6.2f}°/周")
res["rom"] = rom
log("  ※ 人の股関節の目安: 屈曲 120°, 伸展 20°, 外転 45°, 内転 30°。すき間が負 = 2 本の脚（半径 6 cm）がぶつかる")

# 可動域の中で、もう 1 本の脚にぶつからない円錐のうち、最も回るもの（軸の前傾・外への傾き・半頂角を格子で探す）
log("\n=== 可動域内・干渉なしで最も回る円錐（格子探索, 1 層目 (b)） ===")
best = None; feas = []
body = body_of(LEG, 66.0)
for fwd in (15, 30, 45, 60):
    for tilt in (0, 10, 15, 20, 25, 30):
        for beta_deg in (15, 20, 25, 30, 35, 40):
            a = np.array([np.sin(RAD(fwd)), -np.tan(RAD(tilt)) * np.cos(RAD(fwd)), -np.cos(RAD(fwd))]); a /= np.linalg.norm(a)
            us = cone_dir(RAD(beta_deg), np.linspace(0, 2 * np.pi, 121), a)
            fl, ab = np.array([hip_angles(u) for u in us]).T
            gap = min(seg_dist(hip, hip + LEG["length"] * u, other0, other1) for u in us) - 2 * LIMB_R
            pen = max(torso_pen(hip, LEG["length"], u) for u in us)
            if fl.max() <= 120 and -fl.min() <= 20 and ab.max() <= 45 and -ab.min() <= 30 and gap > 0.01 and pen < 0:
                v = D(np.linalg.norm(holonomy_ode(body, RAD(beta_deg), a=a, rtol=1e-7)))
                feas.append((fwd, tilt, beta_deg, v, gap))
                if best is None or v > best[3]:
                    best = (fwd, tilt, beta_deg, v, gap)
res["rom_best"] = dict(fwd=best[0], tilt=best[1], beta=best[2], deg_per_cycle=best[3], gap=best[4], n_feasible=len(feas),
                       ratio_to_arm=best[3] / res["cases"]["arm_b30"]["ode"])
log(f"  可能な組 {len(feas)} 個。最大: 軸の前傾 {best[0]}°, 外へ {best[1]}°, 半頂角 {best[2]}° → {best[3]:.2f}°/周"
    f"（すき間 {best[4] * 100:.1f} cm、腕の 24.8° の {best[3] / res['cases']['arm_b30']['ode']:.2f} 倍）")
for f_ in sorted(feas, key=lambda z: -z[3])[:5]:
    log(f"    前傾 {f_[0]:2d}°, 外へ {f_[1]:2d}°, 半頂角 {f_[2]:2d}° → {f_[3]:6.2f}°/周")
o = run_mj(body, RAD(best[2]), a=np.array([np.sin(RAD(best[0])), -np.tan(RAD(best[1])) * np.cos(RAD(best[0])), -np.cos(RAD(best[0]))])
           / np.linalg.norm([np.sin(RAD(best[0])), -np.tan(RAD(best[1])) * np.cos(RAD(best[0])), -np.cos(RAD(best[0]))]))
res["rom_best"]["mujoco"] = D(np.linalg.norm(o["rv"]))
log(f"  MuJoCo で確認: {res['rom_best']['mujoco']:.3f}°/周（式との差 {(res['rom_best']['mujoco'] / best[3] - 1) * 100:+.2f}%, 追従誤差 {D(o['track_err']):.2f}°）")

# 腕も同じように確かめる: Sim09 の腕の円錐（前下方 30°、半頂角 30°）は胴体にめり込まないか
us = cone_dir(RAD(30), np.linspace(0, 2 * np.pi, 361))
pens = np.array([torso_pen(ARM["shoulder"], ARM["length"], u) for u in us])
res["arm_ref_torso"] = dict(max_pen=float(pens.max()), frac=float(np.mean(pens > 0)))
log(f"\n  腕（Sim09 の円錐）: 胴体の箱に最大 {pens.max() * 100:.1f} cm めり込む（{np.mean(pens > 0) * 100:.0f}% の区間）")
bodyA = body_of(ARM, 74.0); bestA = None; nA = 0
for fwd in (15, 30, 45, 60, 75, 90):
    for tilt in (0, 10, 20, 30, 40, 50):
        for beta_deg in (15, 20, 25, 30, 35, 40, 45):
            a = np.array([np.sin(RAD(fwd)), -np.tan(RAD(tilt)) * np.cos(RAD(fwd)) if fwd < 90 else -np.sin(RAD(tilt)), -np.cos(RAD(fwd))])
            a /= np.linalg.norm(a)
            us = cone_dir(RAD(beta_deg), np.linspace(0, 2 * np.pi, 121), a)
            if max(torso_pen(ARM["shoulder"], ARM["length"], u) for u in us) < -0.01:
                nA += 1
                v = D(np.linalg.norm(holonomy_ode(bodyA, RAD(beta_deg), a=a, rtol=1e-7)))
                if bestA is None or v > bestA[3]:
                    bestA = (fwd, tilt, beta_deg, v)
res["arm_best"] = dict(fwd=bestA[0], tilt=bestA[1], beta=bestA[2], deg_per_cycle=bestA[3], n_feasible=nA)
res["feasible_leg_over_arm"] = best[3] / bestA[3]
log(f"  腕で胴体にめり込まない円錐（{nA} 組）の最大: 軸の前傾 {bestA[0]}°, 外へ {bestA[1]}°, 半頂角 {bestA[2]}° → {bestA[3]:.2f}°/周")
log(f"  → 実現できる円錐どうしで比べると、脚 / 腕 = {best[3] / bestA[3]:.2f}。ただし腕の値は胴体を箱で近似した暫定値"
    "（肩が箱の側面にあるため内向きの成分がすべて禁止される。Sim08 の楕円柱の胸郭で詰める必要がある）")

res["elapsed_s"] = time.time() - t_start
json.dump(res, open(DATA / "field18_legcone.json", "w"), indent=2, ensure_ascii=False, default=float)
(DATA / "field18_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

# ---- 図 ---------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
names = list(CASES); xs = np.arange(len(names)); w = 0.2
for i, (key, lab) in enumerate((("report", "Sim09 report"), ("ode", "layer 1 (ODE)"), ("mujoco", "layer 2 (MuJoCo)"),
                                ("mujoco_path_formula", "MuJoCo path → formula"))):
    vals = [res["cases"][n][key] for n in names]
    ax[0].bar(xs + (i - 1.5) * w, vals, w, label=lab)
for i, n in enumerate(names):
    ax[0].text(i, res["cases"][n]["mujoco"] + 2, f"{res['cases'][n]['mujoco']:.2f}°", ha="center", fontsize=9)
ax[0].set_xticks(xs, ["arm, β=30°", "leg, β=30°", "leg, β=45°"]); ax[0].set_ylabel("body rotation per cone cycle [deg]")
ax[0].set_title("Cone AMBAC in space: formula vs MuJoCo"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis="y")
for integ, mk in (("RK4", "o-"), ("implicitfast", "s-"), ("Euler", "^-")):
    cc = [c for c in comp if c["integrator"] == integ]
    cc = [c for c in cc if c["stable"]]
    ax[1].loglog([c["dt"] * 1e3 for c in cc], [max(abs(c["rel_path"]), 1e-12) for c in cc], mk, label=integ)
ax[1].axvline(0.25, color="k", ls=":", lw=.8)
ax[1].set_xlabel("MuJoCo timestep [ms]"); ax[1].set_ylabel("|MuJoCo − formula on MuJoCo path| (relative)")
ax[1].set_title("Computational error (leg, β=30°, stable runs)"); ax[1].legend(); ax[1].grid(alpha=.3, which="both")
o = main_runs["leg_b30"]
th = [0.0]; R = Rotation.identity(); body = body_of(LEG, 66.0)
for a_, b_ in zip(o["us"][:-1:10], o["us"][10::10]):
    um = (a_ + b_) / np.linalg.norm(a_ + b_)
    R = R * Rotation.from_rotvec(omega_dir(body, um, (b_ - a_) / (10 * o["dt"])) * 10 * o["dt"]); th.append(D(R.magnitude()))
tt = np.arange(len(th)) * 10 * o["dt"]
ax[2].plot(tt, th, label="leg β=30° (formula on MuJoCo path)")
ax[2].axhline(res["cases"]["leg_b30"]["mujoco"], color="C1", ls="--", lw=.8, label="MuJoCo final")
ax[2].set_xlabel("time [s]"); ax[2].set_ylabel("|body rotation| [deg]"); ax[2].set_title("One cone cycle (min-jerk, 2 s)")
ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "field18_legcone.png", dpi=150)
print(f"\n図: {OUT / 'field18_legcone.png'}, 経過 {res['elapsed_s']:.0f} s")
