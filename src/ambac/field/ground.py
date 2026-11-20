"""地上の試験場（重力・床・摩擦あり）と、戦車型台車＋腕の機体（Sim06 と同じ質量特性）.

座標: x 前方, y 左, z 上。腕は x–z 面内で動く（関節の回転軸は −y）。台車は箱形で、床との接触は箱の角で生じる。
"""
from __future__ import annotations

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None

G_EARTH, G_MOON = 9.80665, 1.625


def tank_arm_mjcf(g=G_EARTH, mu=0.6, timestep=5e-4, half=(0.4, 0.3), base_mass=80.0, com_z=0.35,
                  shoulder_z=0.75, l1=0.5, m1=4.0, l2=0.5, m2=3.0, payload=10.0, kp=(30000, 15000), kv=(1500, 800),
                  turret=False, arm_len=1.0, arm_mass=7.0, floor_extra=""):
    """turret=False: 矢状面の 2 関節の腕（荷物運び）。turret=True: 鉛直軸まわりの旋回関節＋水平の棒（ヨーの試験）"""
    a, b = half
    Ib = [base_mass * (4 * b * b + 0.5**2) / 12, base_mass * (4 * a * a + 0.5**2) / 12, base_mass * (4 * a * a + 4 * b * b) / 12]
    if not turret:
        I1, I2 = m1 * l1**2 / 12, m2 * l2**2 / 12
        arm = f"""
      <body name="upper" pos="0 0 {shoulder_z}">
        <joint name="q1" type="hinge" axis="0 -1 0"/>
        <inertial pos="{l1/2} 0 0" mass="{m1}" diaginertia="1e-5 {I1} {I1}"/>
        <geom type="capsule" fromto="0 0 0 {l1} 0 0" size="0.04" contype="0" conaffinity="0" rgba="0.3 0.5 0.9 1"/>
        <body name="fore" pos="{l1} 0 0">
          <joint name="q2" type="hinge" axis="0 -1 0"/>
          <inertial pos="{l2/2} 0 0" mass="{m2}" diaginertia="1e-5 {I2} {I2}"/>
          <geom type="capsule" fromto="0 0 0 {l2} 0 0" size="0.035" contype="0" conaffinity="0" rgba="0.3 0.5 0.9 1"/>
          <body name="payload" pos="{l2} 0 0">
            <inertial pos="0 0 0" mass="{payload}" diaginertia="1e-4 1e-4 1e-4"/>
            <geom type="sphere" size="0.08" contype="0" conaffinity="0" rgba="0.95 0.6 0.1 1"/>
          </body>
        </body>
      </body>"""
        acts = f'<position joint="q1" kp="{kp[0]}" kv="{kv[0]}"/><position joint="q2" kp="{kp[1]}" kv="{kv[1]}"/>'
    else:
        Ia = arm_mass * arm_len**2 / 12
        arm = f"""
      <body name="turret" pos="0 0 {shoulder_z}">
        <joint name="yaw" type="hinge" axis="0 0 1"/>
        <inertial pos="{arm_len/2} 0 0" mass="{arm_mass}" diaginertia="1e-5 {Ia} {Ia}"/>
        <geom type="capsule" fromto="0 0 0 {arm_len} 0 0" size="0.04" contype="0" conaffinity="0" rgba="0.3 0.5 0.9 1"/>
        <body name="payload" pos="{arm_len} 0 0">
          <inertial pos="0 0 0" mass="{payload}" diaginertia="1e-4 1e-4 1e-4"/>
          <geom type="sphere" size="0.08" contype="0" conaffinity="0" rgba="0.95 0.6 0.1 1"/>
        </body>
      </body>"""
        acts = f'<position joint="yaw" kp="{kp[0]}" kv="{kv[0]}"/>'
    return f"""<mujoco model="ground_test_field">
  <option gravity="0 0 {-g}" timestep="{timestep}" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <default><geom solref="0.004 1" solimp="0.95 0.99 0.001"/></default>
  <worldbody>
    <light pos="1 -3 3" dir="-0.3 1 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1" friction="{mu} 0.005 0.0001" rgba="0.85 0.85 0.8 1"/>{floor_extra}
    <body name="base" pos="0 0 0.0005">
      <freejoint name="base"/>
      <inertial pos="0 0 {com_z}" mass="{base_mass}" diaginertia="{Ib[0]} {Ib[1]} {Ib[2]}"/>
      <geom name="tank" type="box" pos="0 0 0.25" size="{a} {b} 0.25" friction="{mu} 0.005 0.0001" rgba="0.55 0.6 0.55 1"/>{arm}
    </body>
  </worldbody>
  <actuator>{acts}</actuator>
</mujoco>"""


def pitch_yaw(qpos):
    """自由関節の四元数から、ピッチ（y 軸まわり、前のめりが正）とヨー（z 軸まわり）を求める"""
    w, x, y, z = qpos[3:7]
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return pitch, yaw


def roll(qpos):
    w, x, y, z = qpos[3:7]
    return np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))


def run(xml, q_init, q_ref_fun, T, settle=0.5, hold=1.0, record_every=4):
    """settle 秒静止させてから、q_ref_fun(τ) で T 秒動かし、hold 秒保持する.
    記録: t（動作開始からの時刻）, q, pitch, yaw, roll, xy（台車の水平位置）, ncon（床との接触点の数）"""
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    nq = m.nu
    d.qpos[7:7 + nq] = q_init; d.ctrl[:] = q_init
    mujoco.mj_forward(m, d)
    out = {k: [] for k in ("t", "q", "pitch", "yaw", "xy", "roll", "ncon")}
    n = int(round((settle + T + hold) / m.opt.timestep))
    for k in range(n + 1):
        t = d.time - settle
        tau = np.clip(t / T, 0, 1)
        d.ctrl[:] = q_ref_fun(tau, t)
        if k % record_every == 0:
            p, y = pitch_yaw(d.qpos)
            out["t"].append(t); out["q"].append(d.qpos[7:7 + nq].copy())
            out["pitch"].append(p); out["yaw"].append(y); out["xy"].append(d.qpos[0:2].copy())
            out["roll"].append(roll(d.qpos)); out["ncon"].append(d.ncon)
        mujoco.mj_step(m, d)
    return {k: np.array(v) for k, v in out.items()}
