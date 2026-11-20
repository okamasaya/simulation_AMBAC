"""アセンブル: 機体の部品表（JSON）とステージの定義（JSON）から MuJoCo のモデル（MJCF）を組み立てる.

機体の部品表（例は configs/field/robots/）
  base     : {"type": "fixed" | "tank" | "floating", "mass", "size": [x, y, z], "com": [x, y, z]}
             fixed=地面に固定（アームだけの試験）、tank=床に置いた箱形の台車、floating=宇宙に浮かぶ胴体
  waist    : 省略可。{"height", "torso_mass", "torso_size", "torso_com", "range": [deg, deg]}（y 軸まわりに前後に傾く上体）
  arms     : [{"name", "mount": [x, y, z], "shoulder": "pitch" | "yaw_pitch" | "ball3",
               "links": [{"length", "mass"}, ...], "payload"}]
             肩: pitch=1 軸（x–z 面内）、yaw_pitch=2 軸ジンバル（旋回＋俯仰）、ball3=3 軸（旋回＋俯仰＋ひねり）。
             リンクが 2 本以上なら、間に肘（俯仰）を入れる。
  actuator : {"kp", "kv"}（位置サーボのゲイン、関節ごとに一律）
ステージの定義（例は configs/field/stages/）
  gravity  : m/s²（0 なら床なし＝宇宙）、friction、obstacles: [{"type": "step" | "slope" | "wall" | "pillar", ...}]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs" / "field"


def load(kind, name):
    """kind: "robots" または "stages" """
    return json.loads((CONFIG / kind / f"{name}.json").read_text(encoding="utf-8"))


def _box_inertia(m, sx, sy, sz):
    return [m * (sy**2 + sz**2) / 12, m * (sx**2 + sz**2) / 12, m * (sx**2 + sy**2) / 12]


def _arm_xml(arm, color="0.3 0.5 0.9 1"):
    """腕の body の入れ子と、関節名のリストを返す"""
    joints = []
    name = arm["name"]; sh = arm.get("shoulder", "pitch")
    mx, my, mz = arm["mount"]
    open_tags, close_tags = [], []
    axes = {"pitch": [("pitch", "0 -1 0")], "yaw_pitch": [("yaw", "0 0 1"), ("pitch", "0 -1 0")],
            "ball3": [("yaw", "0 0 1"), ("pitch", "0 -1 0"), ("roll", "1 0 0")]}[sh]
    # 肩の関節は同じ点に重ねる（質量のない中間 body を使う）
    open_tags.append(f'<body name="{name}_shoulder" pos="{mx} {my} {mz}">')
    close_tags.append("</body>")
    for k, (jn, ax) in enumerate(axes):
        if k > 0:
            open_tags.append(f'<body name="{name}_{jn}_frame">'); close_tags.append("</body>")
        open_tags.append(f'<joint name="{name}_{jn}" type="hinge" axis="{ax}"/>')
        joints.append(f"{name}_{jn}")
        if k < len(axes) - 1:
            open_tags.append('<inertial pos="0 0 0" mass="1e-4" diaginertia="1e-7 1e-7 1e-7"/>')
    for i, link in enumerate(arm["links"]):
        L, m = link["length"], link["mass"]; I = m * L**2 / 12
        if i > 0:
            open_tags.append(f'<body name="{name}_link{i}" pos="{arm["links"][i-1]["length"]} 0 0">')
            close_tags.append("</body>")
            open_tags.append(f'<joint name="{name}_elbow{i}" type="hinge" axis="0 -1 0"/>')
            joints.append(f"{name}_elbow{i}")
        else:
            open_tags.append(f'<body name="{name}_link0">'); close_tags.append("</body>")
        open_tags.append(f'<inertial pos="{L/2} 0 0" mass="{m}" diaginertia="1e-5 {I} {I}"/>')
        open_tags.append(f'<geom type="capsule" fromto="0 0 0 {L} 0 0" size="0.035" contype="0" conaffinity="0" rgba="{color}"/>')
    if arm.get("payload", 0) > 0:
        open_tags.append(f'<body name="{name}_payload" pos="{arm["links"][-1]["length"]} 0 0">'
                         f'<inertial pos="0 0 0" mass="{arm["payload"]}" diaginertia="1e-4 1e-4 1e-4"/>'
                         f'<geom type="sphere" size="0.07" contype="0" conaffinity="0" rgba="0.95 0.6 0.1 1"/></body>')
    return "\n".join(open_tags + close_tags[::-1]), joints


def _obstacles_xml(stage):
    out = []
    mu = stage.get("friction", 0.6)
    for k, ob in enumerate(stage.get("obstacles", [])):
        t = ob["type"]
        if t == "step":           # x の位置から先が高さ h の段
            x0, h, w = ob["x"], ob["height"], ob.get("width", 4.0)
            out.append(f'<geom name="step{k}" type="box" pos="{x0 + 1.0} 0 {h/2}" size="1.0 {w/2} {h/2}" friction="{mu} 0.005 0.0001" rgba="0.6 0.6 0.65 1"/>')
        elif t == "slope":        # x の位置から始まる上り坂（角度 deg、長さ L）
            x0, ang, L = ob["x"], np.radians(ob["angle"]), ob.get("length", 3.0)
            cx = x0 + L / 2 * np.cos(ang); cz = L / 2 * np.sin(ang) - 0.05 * np.cos(ang)
            out.append(f'<geom name="slope{k}" type="box" pos="{cx} 0 {cz}" size="{L/2} 2 0.05" euler="0 {-np.degrees(ang)} 0" friction="{mu} 0.005 0.0001" rgba="0.6 0.65 0.6 1"/>')
        elif t == "wall":
            x0, h = ob["x"], ob.get("height", 1.5)
            out.append(f'<geom name="wall{k}" type="box" pos="{x0} 0 {h/2}" size="0.05 2 {h/2}" rgba="0.7 0.6 0.6 1"/>')
        elif t == "pillar":
            x0, y0, h = ob["x"], ob.get("y", 0.0), ob.get("height", 1.2)
            out.append(f'<geom name="pillar{k}" type="cylinder" pos="{x0} {y0} {h/2}" size="0.1 {h/2}" rgba="0.6 0.6 0.7 1"/>')
    return "\n    ".join(out)


def build(robot, stage, timestep=5e-4):
    """機体の部品表とステージの定義から MJCF の文字列と、関節名のリストを返す"""
    g = stage.get("gravity", 9.80665); mu = stage.get("friction", 0.6)
    b = robot["base"]; sx, sy, sz = b["size"]; com = b.get("com", [0, 0, sz / 2])
    Ib = _box_inertia(b["mass"], sx, sy, sz)
    color = "0.55 0.6 0.55 1"
    if b["type"] == "floating":
        base_open = f'<body name="base" pos="0 0 {stage.get("float_height", 1.5)}"><freejoint name="base"/>'
        geom_z = 0.0; com = b.get("com", [0, 0, 0]); contype = 'contype="0" conaffinity="0"'
    elif b["type"] == "tank":
        base_open = '<body name="base" pos="0 0 0.0005"><freejoint name="base"/>'; geom_z = sz / 2; contype = ""
    else:  # fixed
        base_open = '<body name="base" pos="0 0 0">'; geom_z = sz / 2; contype = ""
    parts = [base_open,
             f'<inertial pos="{com[0]} {com[1]} {com[2]}" mass="{b["mass"]}" diaginertia="{Ib[0]} {Ib[1]} {Ib[2]}"/>',
             f'<geom name="base" type="box" pos="0 0 {geom_z}" size="{sx/2} {sy/2} {sz/2}" friction="{mu} 0.005 0.0001" {contype} rgba="{color}"/>']
    joints = []
    arm_parent_close = ""
    w = robot.get("waist")
    if w:
        tsx, tsy, tsz = w["torso_size"]; tc = w.get("torso_com", [0, 0, tsz / 2])
        It = _box_inertia(w["torso_mass"], tsx, tsy, tsz)
        lo, hi = w.get("range", [-40, 40])
        parts += [f'<body name="torso" pos="0 0 {w["height"]}">',
                  f'<joint name="waist" type="hinge" axis="0 1 0" range="{lo} {hi}" limited="true"/>',
                  f'<inertial pos="{tc[0]} {tc[1]} {tc[2]}" mass="{w["torso_mass"]}" diaginertia="{It[0]} {It[1]} {It[2]}"/>',
                  f'<geom type="box" pos="0 0 {tsz/2}" size="{tsx/2} {tsy/2} {tsz/2}" contype="0" conaffinity="0" rgba="0.65 0.65 0.7 1"/>']
        joints.append("waist"); arm_parent_close = "</body>"
    for arm in robot.get("arms", []):
        x, j = _arm_xml(arm); parts.append(x); joints += j
    parts.append(arm_parent_close + "</body>")
    a = robot.get("actuator", {}); kp, kv = a.get("kp", 20000), a.get("kv", 1000)
    acts = "".join(f'<position name="{j}" joint="{j}" kp="{kp}" kv="{kv}"/>' for j in joints)
    floor = "" if g == 0 else f'<geom name="floor" type="plane" size="8 8 0.1" friction="{mu} 0.005 0.0001" rgba="0.85 0.85 0.8 1"/>'
    xml = f"""<mujoco model="{robot.get('name', 'robot')}_on_{stage.get('name', 'stage')}">
  <option gravity="0 0 {-g}" timestep="{timestep}" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <default><geom solref="0.004 1" solimp="0.95 0.99 0.001"/></default>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <worldbody>
    <light pos="1 -3 4" dir="-0.3 1 -1" directional="true"/>
    {floor}
    {_obstacles_xml(stage)}
    {"".join(parts)}
  </worldbody>
  <actuator>{acts}</actuator>
</mujoco>"""
    return xml, joints


def mass_properties(xml):
    """組み立てたモデルの総質量と、全体の重心（初期姿勢）"""
    import mujoco
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    total = float(m.body_mass.sum())
    com = (m.body_mass[:, None] * d.xipos).sum(0) / total
    return total, com
