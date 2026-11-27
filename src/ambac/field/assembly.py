"""アセンブル: 機体の部品表（JSON）とステージの定義（JSON）から MuJoCo のモデル（MJCF）を組み立てる.

機体の部品表（例は configs/field/robots/）
  base     : {"type": "fixed" | "tank" | "floating", "mass", "size": [x, y, z], "com": [x, y, z], "inertia_scale"（省略時 1）}
             fixed=地面に固定（アームだけの試験）、tank=床に置いた箱形の台車、floating=宇宙に浮かぶ胴体、
             tracked=走行できる履帯の台車（履帯を、駆動する転輪の列と、前端の持ち上がった部分の小さな転輪で近似）。
             tracked の追加項目 "track": {"wheel_radius", "front_height"（前端の高さ）, "approach_deg"（進入角）, "drive_force"（N、全体）, "kv", "wheel_mass", "spacing"}
             tracked の車体（箱）は接触しない（スカート扱い）。接地・段差との接触は転輪だけが受け持つ。
             linked_track=板をつないだ履帯（ambac.field.track）。"track": {"front_height", "approach_deg", "pitch", "plate_mass",
             "sprocket_radius", "wheel_radius", "n_road", "drive_force"（N、1 本あたり）, "kv", "width", "n_tracks"（1 または 2）, "gauge"（左右の中心の間隔）}。
             駆動は起動輪（後ろ）の摩擦で伝える。2 本のとき、駆動は drive_L_sprocket（左、+y）と drive_R_sprocket（右、−y）。
  ankle    : 省略可（base が tank のとき）。{"foot_size", "foot_mass", "height"（足首の高さ）, "axes": "pitch" | "pitch_roll",
             "kp", "kv"（足首のサーボ、省略時は actuator と同じ）, "armature"（kg·m²、回転子の見かけの慣性）, "contact": "box" | "corners"}：base を足と脚に分け、足首の関節でつなぐ
  waist    : 省略可。{"height", "torso_mass", "torso_size", "torso_com", "range": [deg, deg]}（y 軸まわりに前後に傾く上体）
  arms     : [{"name", "mount": [x, y, z], "shoulder": "pitch" | "yaw_pitch" | "ball3",
               "links": [{"length", "mass"}, ...], "payload"}]
             肩: pitch=1 軸（x–z 面内）、yaw_pitch=2 軸ジンバル（旋回＋俯仰）、ball3=3 軸（旋回＋俯仰＋ひねり）。
             省略可 "sea": {関節（yaw など）: {"stiffness" [N·m/rad], "damping", "rotor_inertia"}}：モーター側と腕の間にばね＋減衰（しなり）。
             リンクが 2 本以上なら、間に肘（俯仰）を入れる。
  flywheel : 省略可。{"inertia", "mass", "radius", "pos", "torque_limit", "bearing_damping"}：車体に付けた、鉛直軸まわりに回る円盤（駆動は "flywheel" のトルク）
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
    color = arm.get("color", color)
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
        sea = arm.get("sea", {}).get(jn)
        if sea:
            # しなり（直列弾性）: モーター側（ローター）の関節の先に、ばね＋減衰の受動関節を挟む
            Ir = sea.get("rotor_inertia", 0.5)
            open_tags.append(f'<inertial pos="0 0 0" mass="0.5" diaginertia="{Ir} {Ir} {Ir}"/>')
            open_tags.append(f'<body name="{name}_{jn}_out">'); close_tags.append("</body>")
            open_tags.append(f'<joint name="{name}_{jn}_spring" type="hinge" axis="{ax}" stiffness="{sea["stiffness"]}" damping="{sea.get("damping", 0.0)}"/>')
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
    elif b["type"] == "linked_track":
        base_open = '<body name="base" pos="0 0 0.002"><freejoint name="base"/>'; contype = 'contype="0" conaffinity="0"'
        geom_z = b.get("track", {}).get("front_height", 0.25) + sz / 2 - 0.1
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
    track_world = ""
    if b["type"] == "linked_track":
        from .track import linked_track
        tr = b.get("track", {})
        if tr.get("n_tracks", 1) == 2:   # 左右 2 本（中心の間隔 gauge、それぞれの幅 width）
            g2 = tr.get("gauge", sy - tr.get("width", 0.15)) / 2
            outs = [linked_track(tr, sx, tr.get("width", 0.15), mu, z0=0.002, y=yy, prefix=pf) for yy, pf in [(g2, "L_"), (-g2, "R_")]]
        else:
            outs = [linked_track(tr, sx, tr.get("width", sy * 0.8), mu, z0=0.002)]
        parts.append("".join(o[0] for o in outs)); track_world = "".join(o[1] for o in outs)
        eq_t = "".join(o[2] for o in outs); act_t = "".join(o[3] for o in outs)
        equality = "<equality>" + eq_t + "</equality>"
    fw = robot.get("flywheel")
    if fw:   # 車体に付けた、鉛直軸まわりに回る円盤（逆回転するおもり）。トルクで駆動する
        If = fw["inertia"]; fp = fw.get("pos", [0, 0, 0.4])
        parts.append(f'<body name="flywheel" pos="{fp[0]} {fp[1]} {fp[2]}"><joint name="flywheel" type="hinge" axis="0 0 1" damping="{fw.get("bearing_damping", 0.0)}"/>'
                     f'<inertial pos="0 0 0" mass="{fw.get("mass", 10.0)}" diaginertia="{If / 2} {If / 2} {If}"/>'
                     f'<geom type="cylinder" size="{fw.get("radius", 0.2)} 0.02" contype="0" conaffinity="0" rgba="0.9 0.3 0.3 1"/></body>')
    arm_parent_close = ""
    ank = robot.get("ankle") if b["type"] == "tank" else None
    leg_close = ""; z_off = 0.0
    if ank:
        # 足首: base を足（足裏の箱、自由関節、接触する）に替え、その上に足首の関節（pitch、省略可で roll）で脚をつなぐ。
        # 脚の質量 = base の質量 − 足の質量。全体の重心は足首なしの base と同じになるように脚の重心を置く。
        fx, fy, fh = ank.get("foot_size", [sx, sy, 0.05]); fm = ank.get("foot_mass", 2.0); ha = ank.get("height", 0.08)
        arm_a = ank.get("armature", 0.0)   # モーターの回転子の慣性を減速比の 2 乗倍した値（kg·m²、関節の見かけの慣性）
        lm = b["mass"] - fm; lz = (b["mass"] * com[2] - fm * fh / 2) / lm - ha
        Il = [x * lm / b["mass"] for x in Ib]; If_ = _box_inertia(fm, fx, fy, fh)
        foot_geoms = ""
        if ank.get("contact", "box") == "corners":   # 足裏の 4 隅の小球（box の代わり。接触点の系統誤差を見る）
            rr = 0.01
            for cx in (-fx / 2 + rr, fx / 2 - rr):
                for cy in (-fy / 2 + rr, fy / 2 - rr):
                    foot_geoms += f'<geom type="sphere" pos="{cx} {cy} {rr}" size="{rr}" friction="{mu} 0.005 0.0001"/>'
            foot_geoms += f'<geom type="box" pos="0 0 {fh/2 + 0.005}" size="{fx/2} {fy/2} {fh/2 - 0.005}" contype="0" conaffinity="0" rgba="{color}"/>'
        else:
            foot_geoms = f'<geom name="foot" type="box" pos="0 0 {fh/2}" size="{fx/2} {fy/2} {fh/2}" friction="{mu} 0.005 0.0001" rgba="0.4 0.4 0.45 1"/>'
        parts = [base_open, f'<inertial pos="0 0 {fh/2}" mass="{fm}" diaginertia="{If_[0]} {If_[1]} {If_[2]}"/>', foot_geoms,
                 f'<body name="leg" pos="0 0 {ha}">', f'<joint name="ankle_pitch" type="hinge" axis="0 1 0" armature="{arm_a}"/>']
        joints.append("ankle_pitch"); leg_close = "</body>"
        if ank.get("axes", "pitch") == "pitch_roll":
            parts += ['<inertial pos="0 0 0" mass="1e-4" diaginertia="1e-7 1e-7 1e-7"/>', '<body name="leg_roll_frame">',
                      f'<joint name="ankle_roll" type="hinge" axis="1 0 0" armature="{arm_a}"/>']
            joints.append("ankle_roll"); leg_close = "</body></body>"
        parts += [f'<inertial pos="{com[0]} {com[1]} {lz}" mass="{lm}" diaginertia="{Il[0]} {Il[1]} {Il[2]}"/>',
                  f'<geom type="box" pos="0 0 {(fh + sz) / 2 - ha}" size="{sx/2 - 0.01} {sy/2 - 0.01} {(sz - fh) / 2 - 0.002}" contype="0" conaffinity="0" rgba="{color}"/>']
        z_off = ha
    w = robot.get("waist")
    if w:
        tsx, tsy, tsz = w["torso_size"]; tc = w.get("torso_com", [0, 0, tsz / 2])
        It = _box_inertia(w["torso_mass"], tsx, tsy, tsz)
        lo, hi = w.get("range", [-40, 40])
        parts += [f'<body name="torso" pos="0 0 {w["height"] - z_off}">',
                  f'<joint name="waist" type="hinge" axis="0 1 0" range="{lo} {hi}" limited="true"/>',
                  f'<inertial pos="{tc[0]} {tc[1]} {tc[2]}" mass="{w["torso_mass"]}" diaginertia="{It[0]} {It[1]} {It[2]}"/>',
                  f'<geom type="box" pos="0 0 {tsz/2}" size="{tsx/2} {tsy/2} {tsz/2}" contype="0" conaffinity="0" rgba="0.65 0.65 0.7 1"/>']
        joints.append("waist"); arm_parent_close = "</body>"
    for arm in robot.get("arms", []):
        x, j = _arm_xml(arm); parts.append(x); joints += j
    parts.append(arm_parent_close + leg_close + "</body>")
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
        kpj, kvj = (ank.get("kp", kp), ank.get("kv", kv)) if (ank and j.startswith("ankle")) else (kp, kv)
        acts += f'<position name="{j}" joint="{j}" kp="{kpj}" kv="{kvj}"{fr}/>'
    if wheels:
        tr = b.get("track", {}); R = tr.get("wheel_radius", 0.08)
        fw = tr.get("drive_force", 2000.0) * R / len(wheels)     # 転輪 1 個あたりのトルクの上限
        for wn in wheels:
            acts += f'<velocity name="drive_{wn}" joint="{wn}" kv="{tr.get("kv", 30.0)}" forcelimited="true" forcerange="{-fw} {fw}"/>'
        # 履帯のベルト: すべての転輪を同じ回転にそろえる（駆動力が、接触しているどの転輪にも伝わる）
        equality = "<equality>" + "".join(f'<joint joint1="{wn}" joint2="{wheels[0]}" polycoef="0 1 0 0 0"/>' for wn in wheels[1:]) + "</equality>"
    if b["type"] == "linked_track":
        acts += act_t
    if fw:
        acts += f'<motor name="flywheel" joint="flywheel" gear="1" ctrllimited="true" ctrlrange="{-fw.get("torque_limit", 1000)} {fw.get("torque_limit", 1000)}"/>'
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
    {track_world}
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
