"""宇宙の試験場（重力なし）と、平面モデルと同じ質量特性を持つ MuJoCo の機体.

平面モデル（x–z 面内、角度は x 軸から z 軸へ向かう向きが正）を 3 次元に置くため、
関節の回転軸は −y とする（−y 軸まわりの正の回転が、x → z の向きの回転にあたる）。
胴体は自由関節で浮かび、外力はない（重力ゼロ、減衰ゼロ）。
"""
from __future__ import annotations

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None


def planar_robot_mjcf(base_mass, base_dims, limbs, timestep=5e-4, kp=4000.0, kv=150.0):
    """limbs: [(name, attach_xz, mass, length, q0_rad)]. 腕は一様な棒（重心は関節から length/2）。"""
    bx, bz = base_dims
    Iyy = base_mass * (bx**2 + bz**2) / 12
    Ixx = base_mass * (0.25**2 + bz**2) / 12
    Izz = base_mass * (bx**2 + 0.25**2) / 12
    bodies, acts = [], []
    for name, (ax, az), m, L, _ in limbs:
        I = m * L**2 / 12
        bodies.append(f"""
      <body name="{name}" pos="{ax} 0 {az}">
        <joint name="{name}" type="hinge" axis="0 -1 0" damping="0" armature="0"/>
        <inertial pos="{L/2} 0 0" mass="{m}" diaginertia="{1e-6} {I} {I}"/>
        <geom type="capsule" fromto="0 0 0 {L} 0 0" size="0.03" contype="0" conaffinity="0" rgba="0.3 0.5 0.9 1"/>
      </body>""")
        acts.append(f'<position name="{name}" joint="{name}" kp="{kp}" kv="{kv}"/>')
    return f"""<mujoco model="space_test_field">
  <option gravity="0 0 0" timestep="{timestep}" integrator="RK4"/>
  <worldbody>
    <light pos="0 -3 2" dir="0 1 -0.5"/>
    <body name="base" pos="0 0 0">
      <freejoint name="base"/>
      <inertial pos="0 0 0" mass="{base_mass}" diaginertia="{Ixx} {Iyy} {Izz}"/>
      <geom type="box" size="{bx/2} 0.125 {bz/2}" contype="0" conaffinity="0" rgba="0.7 0.7 0.7 1"/>{''.join(bodies)}
    </body>
  </worldbody>
  <actuator>{''.join(acts)}</actuator>
</mujoco>"""


def base_angle(qpos):
    """自由関節の四元数（w, x, y, z）から、−y 軸まわりの回転角（平面モデルの θ）を求める"""
    w, x, y, z = qpos[3:7]
    return 2 * np.arctan2(-y, w)


def run(model_xml, q_init, q_ref_fun, T, record_every=10):
    """関節の目標角 q_ref_fun(t) を位置サーボで追従させて T 秒動かす.
    戻り値: dict(t, q（実際の関節角）, theta（胴体の角度）, L（重心まわりの角運動量の y 成分）, gyro, tau（関節トルク）)"""
    m = mujoco.MjModel.from_xml_string(model_xml)
    d = mujoco.MjData(m)
    nq = len(q_init)
    d.qpos[7:7 + nq] = q_init
    mujoco.mj_forward(m, d)
    out = {k: [] for k in ("t", "q", "theta", "L", "gyro", "tau")}
    n = int(round(T / m.opt.timestep))
    for k in range(n + 1):
        if k % record_every == 0:
            mujoco.mj_subtreeVel(m, d)
            out["t"].append(d.time)
            out["q"].append(d.qpos[7:7 + nq].copy())
            out["theta"].append(base_angle(d.qpos))
            out["L"].append(-d.subtree_angmom[1][1])          # −y 成分 = 平面の角運動量
            out["gyro"].append(-d.qvel[4])                     # 胴体の角速度（−y 成分、ワールド座標）
            out["tau"].append(d.actuator_force.copy())          # 関節トルク（位置サーボの出力）
        d.ctrl[:] = q_ref_fun(d.time)
        mujoco.mj_step(m, d)
    return {k: np.array(v) for k, v in out.items()}
