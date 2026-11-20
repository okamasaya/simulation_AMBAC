"""履帯（板をつないだベルト）のモデル: 履板の輪を、起動輪・転輪・誘導輪に巻き付ける.

assembly.build の base.type = "linked_track" から使う。座標は車体（base）の座標で、x が前、z が上。
  ・履板: 厚さ t、長さ p（ピッチ）の箱を、ヒンジ（y 軸）で輪につなぐ。最初の履板は自由関節、最後と最初は connect 拘束で閉じる
  ・起動輪（後ろ、半径 Rs、駆動）、転輪（半径 Rr、空転）、誘導輪（前、高さ H、半径 Ri、空転）は車体に付く
  ・接触: 履板 ↔ 床・段・障害物、履板 ↔ 車輪。履板どうし、車輪 ↔ 床は接触しない
  ・履板と地面の接触はやや柔らかくする（ground_solref、既定 0.1 s：静止時に約 3.5 mm 沈む）。硬い接触のままだと、
    荷重が履帯の前後の端に集中し（不静定な多点接触のため）、旋回の抵抗が約 2 倍になる。実際の地面の沈み込みやサスペンションに相当する
  ・誘導輪はばね（張り装置、既定 1500 N で前へ押す）で前後に動く。起動輪には歯がないため、摩擦（既定 μ = 2.0）と張りで履板を送る
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull


def belt_path(circles, pitch):
    """円（中心 x, z、半径 r）の列の凸包をベルトの経路とし、等間隔 pitch に近い N 点に分ける（反時計回り：上から見て前→上→後→下）"""
    pts = []
    for (cx, cz, r) in circles:
        a = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        pts.append(np.c_[cx + r * np.cos(a), cz + r * np.sin(a)])
    pts = np.vstack(pts)
    hull = pts[ConvexHull(pts).vertices]                 # 反時計回り（x–z 平面）
    seg = np.linalg.norm(np.diff(np.vstack([hull, hull[:1]]), axis=0), axis=1)
    L = seg.sum(); N = int(round(L / pitch))
    s = np.r_[0, np.cumsum(seg)]
    tgt = np.linspace(0, L, N, endpoint=False)
    closed = np.vstack([hull, hull[:1]])
    P = np.c_[np.interp(tgt, s, closed[:, 0]), np.interp(tgt, s, closed[:, 1])]
    return P, L


def linked_track(spec, length, width, mu, z0=0.0, y=0.0, prefix=""):
    """履帯の MJCF 部品を返す: (車体に付ける車輪の XML, ワールドに置く履板の鎖の XML, equality の XML, actuator の XML, 情報)"""
    t = spec.get("plate_thickness", 0.02); pitch = spec.get("pitch", 0.04)
    Rs = spec.get("sprocket_radius", 0.10); Rr = spec.get("wheel_radius", 0.08); Ri = spec.get("idler_radius", 0.08)
    H = spec.get("front_height", 0.25); n_road = spec.get("n_road", 5)
    half = length / 2
    zc = t + Rr                                          # 転輪の中心の高さ（履板の厚さの上）
    sprocket = (-half + Rs, t + Rs, Rs)
    ang = np.radians(spec.get("approach_deg", 40.0))
    idler = (half - Rr + (H - zc) / np.tan(ang), H, Ri)
    road = [(x, zc, Rr) for x in np.linspace(-half + 2 * Rs, half - Rr, n_road)]
    circles = [sprocket, idler] + road
    # 履板の中心線は、車輪の表面から t/2 外側。さらに、まっすぐな履板が円弧の内側に食い込まないよう、
    # 頂点を外側にずらす（弦の中央が円弧に接する: 弦の矢高 ≈ p²/(8r)）。遊び（clearance）も少し加える。
    # これを入れないと、ベルトが車輪を締め付けて大きな張力が生じ、荷重が前後の端の車輪に集中する
    clear = spec.get("clearance", 0.0005)
    P, L = belt_path([(cx, cz, r + t / 2 + pitch**2 / (8 * (r + t / 2)) + clear) for (cx, cz, r) in circles], pitch)
    N = len(P); p_eff = L / N
    mass = spec.get("plate_mass", 0.3); w = width
    Ip = [mass * (w**2 + t**2) / 12, mass * (p_eff**2 + t**2) / 12, mass * (p_eff**2 + w**2) / 12]
    # 車輪（車体の子）
    wheels = []
    for k, (cx, cz, r) in enumerate(circles):
        name = prefix + ("sprocket" if k == 0 else ("idler" if k == 1 else f"road{k - 2}"))
        fr = spec.get("sprocket_friction", 2.0) if k == 0 else 0.3
        # 誘導輪は前後に動けるようにし、ばねで前に押してベルトに張りを与える（張り装置）
        tens = ""
        if k == 1 and spec.get("tensioner", True):
            kt = spec.get("tensioner_stiffness", 1.0e5); pre = spec.get("tensioner_preload", 1500.0)
            tens = (f'<joint name="{name}_tension" type="slide" axis="1 0 0" stiffness="{kt}" springref="{pre / kt:.5f}" '
                    f'damping="500" range="-0.02 0.03" limited="true"/>')
        wheels.append(f'<body name="{name}" pos="{cx:.4f} {y:.4f} {cz + z0:.4f}">{tens}<joint name="{name}" type="hinge" axis="0 1 0" damping="0.05"/>'
                      f'<inertial pos="0 0 0" mass="2" diaginertia="{r*r:.4f} {r*r:.4f} {r*r:.4f}"/>'
                      f'<geom type="cylinder" size="{r} {w / 2}" euler="90 0 0" contype="4" conaffinity="2" friction="{fr} 0.005 0.0001" rgba="0.3 0.3 0.3 1"/></body>')
    # 履板の鎖（ワールドの子、入れ子）
    chain, close = [], []
    for i in range(N):
        a, b = P[i], P[(i + 1) % N]
        th = np.arctan2(b[1] - a[1], b[0] - a[0])         # x–z 平面での向き
        if i == 0:
            chain.append(f'<body name="{prefix}plate0" pos="{a[0]:.5f} {y:.4f} {a[1] + z0:.5f}" euler="0 {-np.degrees(th):.4f} 0"><freejoint name="{prefix}track"/>')
            prev = th
        else:
            d = th - prev; d = (d + np.pi) % (2 * np.pi) - np.pi
            chain.append(f'<body name="{prefix}plate{i}" pos="{p_eff:.5f} 0 0" euler="0 {-np.degrees(d):.4f} 0"><joint name="{prefix}link{i}" type="hinge" axis="0 1 0" damping="0.02"/>')
            prev = th
        # 外向きの法線（反時計回りの経路では、進行方向の右手 = 外側）: 局所座標の −z 側…ではなく、経路の外側に厚みを置く
        chain.append(f'<inertial pos="{p_eff / 2:.5f} 0 0" mass="{mass}" diaginertia="{Ip[0]:.6f} {Ip[1]:.6f} {Ip[2]:.6f}"/>'
                     f'<geom type="box" pos="{p_eff / 2:.5f} 0 0" size="{p_eff / 2 * 0.98:.5f} {w / 2:.4f} {t / 2:.4f}" contype="2" conaffinity="1" '
                     f'friction="{mu} 0.005 0.0001" solref="{spec.get("ground_solref", 0.1)} 1" rgba="0.25 0.25 0.22 1"/>')
        close.append("</body>")
    chain_xml = "".join(chain) + "".join(close)
    equality = f'<connect body1="{prefix}plate{N - 1}" body2="{prefix}plate0" anchor="{p_eff:.5f} 0 0" solref="0.002 1"/>'
    tq = spec.get("drive_force", 2000.0) * Rs
    act = f'<velocity name="drive_{prefix}sprocket" joint="{prefix}sprocket" kv="{spec.get("kv", 200.0)}" forcelimited="true" forcerange="{-tq} {tq}"/>'
    info = dict(n_plates=N, pitch=p_eff, perimeter=L, sprocket_radius=Rs, circles=circles)
    return "".join(wheels), chain_xml, equality, act, info
