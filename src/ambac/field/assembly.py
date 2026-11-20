"""アセンブル: 機体の部品表（JSON）とステージの定義（JSON）から MuJoCo のモデル（MJCF）を組み立てる.

機体の部品表（例は configs/field/robots/）
  base     : {"type": "fixed" | "tank" | "floating", "mass", "size": [x, y, z], "com": [x, y, z], "inertia_scale"（省略時 1）}
             fixed=地面に固定（アームだけの試験）、tank=床に置いた箱形の台車、floating=宇宙に浮かぶ胴体、
             tracked=走行できる履帯の台車（履帯を、駆動する転輪の列と、前端の持ち上がった部分の小さな転輪で近似）。
             tracked の追加項目 "track": {"wheel_radius", "front_height"（前端の高さ）, "approach_deg"（進入角）, "drive_force"（N、全体）, "kv", "wheel_mass", "spacing"}
             tracked の車体（箱）は接触しない（スカート扱い）。接地・段差との接触は転輪だけが受け持つ。
  waist    : 省略可。{"height", "torso_mass", "torso_size", "torso_com", "range": [deg, deg]}（y 軸まわりに前後に傾く上体）
  arms     : [{"name", "mount": [x, y, z], "shoulder": "pitch" | "yaw_pitch" | "ball3",
               "links": [{"length", "mass"}, ...], "payload"}]
             肩: pitch=1 軸（x–z 面内）、yaw_pitch=2 軸ジンバル（旋回＋俯仰）、ball3=3 軸（旋回＋俯仰＋ひねり）。
             リンクが 2 本以上なら、間に肘（俯仰）を入れる。
  actuator : {"kp", "kv"}（位置サーボのゲイン、関節ごとに一律）
             省略可: "torque_limit"（N·m、数値なら全関節、辞書なら {関節名の末尾: 値}、例 {"pitch": 150, "elbow1": 80}）、
                     "speed_limit_deg_s"（目標角の変化の速さの上限、同じ書き方）。モーターと減速機の定格を表す簡易モデル。
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
        if t == "step":           # x の位置から先が高さ h の段（角は半径 r で丸める。鋭い角は接触が不安定になるため）
            x0, h, w = ob["x"], ob["height"], ob.get("width", 4.0); r = min(ob.get("edge_radius", 0.03), h / 2)
            fr = f'friction="{mu} 0.005 0.0001" rgba="0.6 0.6 0.65 1"'
            out.append(f'<geom name="step{k}" type="box" pos="{x0 + r + 1.0} 0 {h/2}" size="1.0 {w/2} {h/2}" {fr}/>')
            out.append(f'<geom name="step{k}_front" type="box" pos="{x0 + r/2} 0 {(h - r)/2}" size="{r/2} {w/2} {(h - r)/2}" {fr}/>')
            out.append(f'<geom name="step{k}_edge" type="capsule" fromto="{x0 + r} {-w/2} {h - r} {x0 + r} {w/2} {h - r}" size="{r}" {fr}/>')
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
    Ib = [x * b.get("inertia_scale", 1.0) for x in _box_inertia(b["mass"], sx, sy, sz)]
    color = "0.55 0.6 0.55 1"
    if b["type"] == "floating":
        base_open = f'<body name="base" pos="0 0 {stage.get("float_height", 1.5)}"><freejoint name="base"/>'
        geom_z = 0.0; com = b.get("com", [0, 0, 0]); contype = 'contype="0" conaffinity="0"'
    elif b["type"] == "tank":
        base_open = '<body name="base" pos="0 0 0.0005"><freejoint name="base"/>'; geom_z = sz / 2; contype = ""
    elif b["type"] == "tracked":
        base_open = '<body name="base" pos="0 0 0.002"><freejoint name="base"/>'; contype = 'contype="0" conaffinity="0"'
        tr = b.get("track", {}); R = tr.get("wheel_radius", 0.08); H = tr.get("front_height", 0.25)
        geom_z = R + sz / 2   # 車体の箱は転輪の上に載る
    else:  # fixed
        base_open = '<body name="base" pos="0 0 0">'; geom_z = sz / 2; contype = ""
    parts = [base_open,
             f'<inertial pos="{com[0]} {com[1]} {com[2]}" mass="{b["mass"]}" diaginertia="{Ib[0]} {Ib[1]} {Ib[2]}"/>',
             f'<geom name="base" type="box" pos="0 0 {geom_z}" size="{sx/2} {sy/2} {sz/2}" friction="{mu} 0.005 0.0001" {contype} rgba="{color}"/>']
    joints = []
    wheels = []
    equality = ""
    if b["type"] == "tracked":
        # 接地部の転輪（後端 −sx/2 から 前端 sx/2 − R まで）と、前端の斜めの部分の転輪（高さ R → H）
        n_road = max(4, int(round((sx - 2 * R) / (tr.get("spacing", 0.4) * R))) + 1)   # 転輪を密に並べ、ベルトの直線に近づける
        xs = np.linspace(-sx / 2 + R, sx / 2 - R, n_road); pts = [(x, R) for x in xs]
        ang = np.radians(tr.get("approach_deg", 40.0)); dx = (H - R) / np.tan(ang)   # 前端の斜めの部分（進入角）
        n_sl = max(2, int(np.ceil((H - R) / np.sin(ang) / (tr.get("spacing", 0.4) * R))))
        for k in range(1, n_sl + 1):
            f = k / n_sl; pts.append((sx / 2 - R + f * dx, R + f * (H - R)))
        mu_w = mu
        wm = tr.get("wheel_mass", 3.0); wi = 0.5 * wm * R**2   # 転輪の質量（軽すぎると駆動が振動する）
        for i, (x, z) in enumerate(pts):
            parts.append(f'<body name="wheel{i}" pos="{x:.4f} 0 {z:.4f}"><joint name="wheel{i}" type="hinge" axis="0 1 0" damping="0.01"/>'
                         f'<inertial pos="0 0 0" mass="{wm}" diaginertia="{wi:.5f} {wi:.5f} {wi:.5f}"/>'
                         f'<geom type="capsule" fromto="0 {-(sy/2 - R)} 0 0 {sy/2 - R} 0" size="{R}" contype="1" conaffinity="0" friction="{mu_w} 0.005 0.0001" rgba="0.2 0.2 0.2 1"/></body>')
            wheels.append(f"wheel{i}")
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

    def per_joint(spec, j):
        if spec is None:
            return None
        if isinstance(spec, dict):
            for k, v in spec.items():
                if j == k or j.endswith("_" + k):
                    return v
            return None
        return spec
    acts = ""
    for j in joints:
        tl = per_joint(a.get("torque_limit"), j)
        fr = f' forcelimited="true" forcerange="{-tl} {tl}"' if tl else ""
        acts += f'<position name="{j}" joint="{j}" kp="{kp}" kv="{kv}"{fr}/>'
    if wheels:
        tr = b.get("track", {}); R = tr.get("wheel_radius", 0.08)
        fw = tr.get("drive_force", 2000.0) * R / len(wheels)     # 転輪 1 個あたりのトルクの上限
        for wn in wheels:
            acts += f'<velocity name="drive_{wn}" joint="{wn}" kv="{tr.get("kv", 30.0)}" forcelimited="true" forcerange="{-fw} {fw}"/>'
        # 履帯のベルト: すべての転輪を同じ回転にそろえる（駆動力が、接触しているどの転輪にも伝わる）
        equality = "<equality>" + "".join(f'<joint joint1="{wn}" joint2="{wheels[0]}" polycoef="0 1 0 0 0"/>' for wn in wheels[1:]) + "</equality>"
    sl = [per_joint(a.get("speed_limit_deg_s"), j) for j in joints]
    custom = ""
    if any(sl):
        custom = '<custom><numeric name="speed_limit" data="' + " ".join(str(np.radians(v) if v else 0.0) for v in sl) + '"/></custom>'
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
  {equality}
  {custom}
</mujoco>"""
    return xml, joints


def mass_properties(xml):
    """組み立てたモデルの総質量と、全体の重心（初期姿勢）"""
    import mujoco
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    total = float(m.body_mass.sum())
    com = (m.body_mass[:, None] * d.xipos).sum(0) / total
    return total, com
