"""番外編 03: トリプルアクセルを腕を伸ばしたまま跳べるか（体格・技術・運動能力の切り分け）.

空中では角運動量 L が一定なので、回転数は N = (L / 2π) ∫ dt / I(t)。
  - 運動能力: 踏み切りで作る L と、跳ぶ高さで決まる滞空時間 t_air
  - 体格    : 姿勢ごとの慣性モーメント I（身長・体重・比率）
  - 技術    : 腕をいつ・どれだけ速く・どの姿勢にするか（I(t) の時間の形）、左右の非対称（回転軸の傾き = AMBAC）
の 3 つに分けて、「腕を頭の上に伸ばす（up）」「片腕だけ上げる（tano）」「横に広げる（side）」を、
腕を胸の前で組む普通の締め（tuck）と比べる。特定の選手の解析ではない（本人の測定値はない）。

人体: de Leva (1996) の体節の質量・重心・回転半径（configs/body/*.json）。身長・体重で拡大縮小する。
ノギスで測った人形（ボディちゃん）は configs/body/bodychan_template.json の形で入れると、楕円の錐台（一様密度）から同じ形式を作る。
MuJoCo: 自由に浮いた多体（胴体＋頭、腕は肩 3 軸＋肘、脚は固定）を RK4 で飛ばし、腕は位置サーボで動かす。

文献値（docs/16 に出典）:
  Knoll & Hildebrand 1998: 3A の滞空 0.68〜0.76 s、最後に氷を離れるときの L 32.8 ± 2.0 kg m²/s（最大 37.5 ± 4.7）
  King ら（USOC）: 3A の回転速度 4.9 rev/s（5 人）、高さ 0.58〜0.81 m
  Kunverji & Lester 2023: 2019 年世界選手権の 3A の高さ 0.59 ± 0.05 m

実行: python bangai03_axel_arms.py [poses|flight|factors|errors|anneal|figure|all]
       python bangai03_axel_arms.py poses --body ../configs/body/bodychan_template.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from common import DATA, ROOT, smooth_step

OUT = ROOT / "figures" / "field"
OUT.mkdir(parents=True, exist_ok=True)
BODY_DIR = ROOT / "configs" / "body"
G = 9.81
SEGS = ["head", "trunk", "upperarm", "forearm", "hand", "thigh", "shank", "foot"]

# 基準の選手（仮定）: 男子、de Leva の比率、身長 1.70 m・体重 62 kg
REF = dict(body="deleva1996_male.json", height=1.70, mass=62.0, boot=1.3)
# 文献の運動能力（Knoll & Hildebrand 1998）
L_NOM, L_SD, L_MAX = 32.8, 2.0, 37.5
T_AIR_NOM, T_AIR_LO, T_AIR_HI = 0.70, 0.68, 0.76
T_TR = 0.10     # 腕の姿勢を変える時間（仮定。King らは「腕は脚より速く締まる」）
T_OPEN = 0.12   # 着氷前に腕を開く時間（仮定）


# ---------------------------------------------------------------- 体のパラメータ
def _frustum(len_m, w0, d0, w1, d1, n=200):
    """楕円の錐台（一様密度 1）の体積・重心（近位からの比）・回転半径の比 [ML 軸, AP 軸, 長軸]."""
    s = (np.arange(n) + 0.5) / n
    a = 0.5 * (w0 + (w1 - w0) * s)   # 左右の半径
    b = 0.5 * (d0 + (d1 - d0) * s)   # 前後の半径
    dz = len_m / n
    dm = np.pi * a * b * dz
    m = dm.sum()
    z = s * len_m
    zc = (dm * z).sum() / m
    # 薄い楕円板: 長軸まわり m(a²+b²)/4、左右の軸(ML) まわり m b²/4 + m z²、前後の軸(AP) まわり m a²/4 + m z²
    I_long = (dm * (a**2 + b**2) / 4).sum()
    I_ml = (dm * (b**2 / 4 + (z - zc) ** 2)).sum()
    I_ap = (dm * (a**2 / 4 + (z - zc) ** 2)).sum()
    r = np.sqrt(np.array([I_ml, I_ap, I_long]) / m) / len_m
    return m, zc / len_m, r


def load_body(path=None, height=None, mass=None, boot=None, scale=None, dI_frac=None):
    """体の JSON を読み、SI の体節の表にする. scale は {"shoulder": 1.1, "arm": 1.0, ...} の比率の変更（感度用）."""
    path = Path(path or BODY_DIR / REF["body"])
    if not path.is_absolute() and not path.exists():
        path = BODY_DIR / path
    cfg = json.load(open(path))
    H = float(height or cfg["height_m"])
    M = float(mass or cfg["mass_kg"])
    boot = float(cfg.get("boot_mass_kg", 0.0) if boot is None else boot)
    scale = scale or {}
    if cfg.get("type") == "caliper":
        k = 1.0 / cfg["figure_height_mm"]                 # 人形の長さ → 身長の比
        segs, vol = {}, {}
        for nm in SEGS:
            q = cfg["measure_mm"][nm]
            v, c, r = _frustum(q["len"] * k, q["w0"] * k, q["d0"] * k, q["w1"] * k, q["d1"] * k)
            nmul = 2 if nm in ("upperarm", "forearm", "hand", "thigh", "shank", "foot") else 1
            vol[nm] = v * nmul
            segs[nm] = dict(len=q["len"] * k, com=c, r=list(r))
        tot = sum(vol.values())
        for nm in SEGS:
            nmul = 2 if nm in ("upperarm", "forearm", "hand", "thigh", "shank", "foot") else 1
            segs[nm]["mass"] = vol[nm] / tot / nmul
        lm = {key: val * k for key, val in cfg["landmarks_mm"].items()}
    else:
        segs = {nm: dict(cfg["segments"][nm]) for nm in SEGS}
        lm = dict(cfg["landmarks_frac"])
    lm.setdefault("shoulder_below_c7", 0.03)
    # 比率の変更（感度）
    if "arm" in scale:
        for nm in ("upperarm", "forearm", "hand"):
            segs[nm] = dict(segs[nm], len=segs[nm]["len"] * scale["arm"])
    if "shoulder" in scale:
        lm["shoulder_jc_half_width"] *= scale["shoulder"]
    if "girth" in scale:   # 胴体の太さ（回転半径）
        segs["trunk"] = dict(segs["trunk"], r=[x * scale["girth"] for x in segs["trunk"]["r"]])
    if "armmass" in scale:
        for nm in ("upperarm", "forearm", "hand"):
            segs[nm] = dict(segs[nm], mass=segs[nm]["mass"] * scale["armmass"])
        tot = segs["head"]["mass"] + segs["trunk"]["mass"] + 2 * sum(segs[n]["mass"] for n in SEGS[2:])
        for nm in SEGS:
            segs[nm] = dict(segs[nm], mass=segs[nm]["mass"] / tot)
    out = {}
    for nm in SEGS:
        s = segs[nm]
        L = (s["len"] or 0.0) * H
        out[nm] = dict(len=L, mass=s["mass"] * M, com=s["com"], r=np.array(s["r"], float))
    hipH = lm["hip_jc_height"] * H
    out["trunk"]["len"] = H - out["head"]["len"] - hipH
    if dI_frac is None:
        dI_frac = calib()["dI_frac"]
    return dict(name=cfg["name"], H=H, M=M, boot=boot, seg=out, dI=dI_frac * M * H**2, dI_frac=dI_frac,
                hipH=hipH, wsh=lm["shoulder_jc_half_width"] * H, whip=lm["hip_jc_half_width"] * H,
                chest=lm["chest_half_depth"] * H, sh_below=lm["shoulder_below_c7"] * H)


# ---------------------------------------------------------------- 較正（締めの緩さ）
N_AIR_NOM = 4.9 * T_AIR_NOM      # King らの 4.9 rev/s × Knoll らの 0.70 s = 3.43 回転（空中の回転数の目安）
_CAL = {}


def calib(n_target=N_AIR_NOM, recompute=False):
    """理想の締め（胸の前で組む）だと回りすぎる（F64）。全姿勢に同じ ΔI（胴体の長軸）を足し、
    基準の選手が L 32.8・滞空 0.70 s・普通の締めで n_target 回転になるようにする. ΔI / (M H²) で保存."""
    key = round(n_target, 4)
    if key in _CAL and not recompute:
        return _CAL[key]
    f = DATA / "bangai03_calib.json"
    allc = json.load(open(f)) if f.exists() else {}
    if str(key) in allc and not recompute:
        _CAL[key] = allc[str(key)]
        return _CAL[key]
    _CAL[key] = dict(dI_frac=0.0)       # 再帰を避ける仮の値
    from scipy.optimize import brentq
    def n_of(x):
        B = load_body(REF["body"], REF["height"], REF["mass"], REF["boot"], dI_frac=x)
        return Skater(B).fly(std_schedule("tuck"), L_NOM, T_AIR_NOM, dt=2e-3, torque=False)["turns"]
    x1 = brentq(lambda x: n_of(x) - n_target, 0.0, 0.01, xtol=1e-7)
    B = load_body(REF["body"], REF["height"], REF["mass"], REF["boot"], dI_frac=x1)
    res = dict(dI_frac=x1, dI=x1 * REF["mass"] * REF["height"] ** 2, n_target=n_target, n_ideal=n_of(0.0),
               I_tuck_ideal=Skater(load_body(REF["body"], REF["height"], REF["mass"], REF["boot"], dI_frac=0.0)).inertia("tuck")[0][2, 2],
               I_tuck_cal=Skater(B).inertia("tuck")[0][2, 2])
    _CAL[key] = res
    allc[str(key)] = res
    json.dump(allc, open(f, "w"), indent=1)
    return res


# ---------------------------------------------------------------- MuJoCo のモデル
def _inertial(seg, z0, sign=-1.0, extra=""):
    """長さ方向が局所 z（sign=-1 なら下向き）の体節の inertial 要素."""
    L, m, r = seg["len"], seg["mass"], seg["r"]
    c = sign * seg["com"] * L + z0
    ixx = m * (r[1] * L) ** 2   # 前後の軸 (x) まわり = transverse
    iyy = m * (r[0] * L) ** 2   # 左右の軸 (y) まわり = sagittal
    izz = m * (r[2] * L) ** 2
    return f'<inertial pos="0 0 {c:.6f}" mass="{m:.6f}" diaginertia="{ixx:.6e} {iyy:.6e} {izz:.6e}" {extra}/>'


def _quat_from_z(v):
    """局所 z を v に向ける四元数 (w x y z)."""
    v = np.asarray(v, float) / np.linalg.norm(v)
    z = np.array([0, 0, 1.0])
    ax = np.cross(z, v)
    s = np.linalg.norm(ax)
    if s < 1e-12:
        return np.array([1, 0, 0, 0.0]) if v[2] > 0 else np.array([0, 1, 0, 0.0])
    ang = np.arctan2(s, v @ z)
    ax /= s
    return np.r_[np.cos(ang / 2), np.sin(ang / 2) * ax]


def build_xml(B, kp=1.0, kv=1.0, dt=2e-4, integrator="RK4"):
    """kp・kv は関節ごとの基準（固有角振動数 80 rad/s、臨界減衰）への倍率."""
    S = B["seg"]
    zc7 = B["H"] - S["head"]["len"] - B["hipH"]          # 腰の関節の中心を原点にした頸椎の高さ
    zsh = zc7 - B["sh_below"]
    trunk = S["trunk"]
    # 胴体＋頭を 1 つの剛体に（頭は頸椎から上）
    mt, mh = trunk["mass"], S["head"]["mass"]
    Lt, Lh = trunk["len"], S["head"]["len"]
    ct = zc7 - trunk["com"] * Lt
    ch = zc7 + (1 - S["head"]["com"]) * Lh
    def diag(seg):
        L, m, r = seg["len"], seg["mass"], seg["r"]
        return np.array([m * (r[1] * L) ** 2, m * (r[0] * L) ** 2, m * (r[2] * L) ** 2])
    zt = (mt * ct + mh * ch) / (mt + mh)
    It = diag(trunk) + diag(S["head"]) + np.array([1, 1, 0]) * (mt * (ct - zt) ** 2 + mh * (ch - zt) ** 2)
    It = It + B.get("dI", 0.0) * np.array([0.5, 0.5, 1.0])    # 締めの緩さ（較正、docs/16）。薄い輪と同じ配分
    arm = []
    for side, sg in (("L", 1), ("R", -1)):
        ua, fa, hd = S["upperarm"], S["forearm"], S["hand"]
        hand_in = _inertial(hd, -fa["len"])
        arm.append(f"""
      <body name="ua{side}" pos="0 {sg * B['wsh']:.5f} {zsh:.5f}">
        <joint name="sf{side}" type="hinge" axis="0 1 0" damping="0"/>
        <joint name="sa{side}" type="hinge" axis="{sg} 0 0"/>
        <joint name="sr{side}" type="hinge" axis="0 0 1"/>
        {_inertial(ua, 0.0)}
        <geom type="capsule" fromto="0 0 0 0 0 {-ua['len']:.4f}" size="0.04" contype="0" conaffinity="0" rgba=".8 .6 .5 1"/>
        <body name="fa{side}" pos="0 0 {-ua['len']:.5f}">
          <joint name="el{side}" type="hinge" axis="0 -1 0"/>
          {_inertial(fa, 0.0)}
          <geom type="capsule" fromto="0 0 0 0 0 {-fa['len']:.4f}" size="0.032" contype="0" conaffinity="0" rgba=".8 .6 .5 1"/>
          <body name="hd{side}" pos="0 0 0">{hand_in}
            <geom type="sphere" pos="0 0 {-fa['len'] - 0.5 * hd['len']:.4f}" size="0.04" contype="0" conaffinity="0" rgba=".8 .6 .5 1"/>
          </body>
        </body>
      </body>""")
    legs = []
    th, sh, ft = S["thigh"], S["shank"], S["foot"]
    Lleg = th["len"] + sh["len"]
    for side, sg in (("L", 1), ("R", -1)):
        # 足をそろえて交差（空中の締めの姿勢）: 足首が中心線に来るよう内に傾け、左足を少し前に
        foot_pt = np.array([0.03 * sg, 0.0, -Lleg])
        hip = np.array([0, sg * B["whip"], 0])
        q = _quat_from_z(hip - foot_pt)     # 局所 z は上向き（近位）→ 体節は -z へ伸ばす
        qs = " ".join(f"{x:.6f}" for x in q)
        boot = ""
        if B["boot"] > 0:
            boot = (f'<body name="boot{side}" pos="0 0 {-(Lleg + 0.06):.4f}"><inertial pos="0 0 0" mass="{B["boot"]:.4f}" '
                    f'diaginertia="{B["boot"] * 0.08**2:.4e} {B["boot"] * 0.08**2:.4e} {B["boot"] * 0.03**2:.4e}"/></body>')
        legs.append(f"""
      <body name="th{side}" pos="0 {sg * B['whip']:.5f} 0" quat="{qs}">
        {_inertial(th, 0.0)}
        <geom type="capsule" fromto="0 0 0 0 0 {-th['len']:.4f}" size="0.06" contype="0" conaffinity="0" rgba=".3 .3 .6 1"/>
        <body name="sh{side}" pos="0 0 {-th['len']:.5f}">{_inertial(sh, 0.0)}
          <geom type="capsule" fromto="0 0 0 0 0 {-sh['len']:.4f}" size="0.045" contype="0" conaffinity="0" rgba=".3 .3 .6 1"/>
          <body name="ft{side}" pos="0 0 {-sh['len']:.5f}">{_inertial(ft, 0.0)}</body>
        </body>
        {boot}
      </body>""")
    acts = []
    wn = 80.0
    ua, fa, hd = S["upperarm"], S["forearm"], S["hand"]
    I_sh = ua["mass"] * (0.5 * ua["len"]) ** 2 + (fa["mass"] + hd["mass"]) * (ua["len"] + 0.5 * fa["len"]) ** 2
    I_el = fa["mass"] * (0.5 * fa["len"]) ** 2 + hd["mass"] * (fa["len"] + 0.4 * hd["len"]) ** 2
    I_eff = dict(sf=I_sh, sa=I_sh, sr=0.4 * I_el, el=I_el)
    for side in "LR":
        for j in ("sf", "sa", "sr", "el"):
            acts.append(f'<position name="a{j}{side}" joint="{j}{side}" kp="{kp * I_eff[j] * wn**2:.3f}" kv="{kv * 2 * I_eff[j] * wn:.4f}"/>')
    return f"""<mujoco model="skater">
  <option timestep="{dt}" integrator="{integrator}" gravity="0 0 {-G}"/>
  <compiler angle="radian"/>
  <worldbody>
    <body name="trunk" pos="0 0 1.5">
      <freejoint name="root"/>
      <inertial pos="0 0 {zt:.6f}" mass="{mt + mh:.6f}" diaginertia="{It[0]:.6e} {It[1]:.6e} {It[2]:.6e}"/>
      <geom type="capsule" fromto="0 0 0 0 0 {zc7:.4f}" size="{B['chest']:.4f}" contype="0" conaffinity="0" rgba=".9 .9 .9 1"/>
      <geom type="sphere" pos="0 0 {ch:.4f}" size="0.1" contype="0" conaffinity="0" rgba=".8 .6 .5 1"/>
      {''.join(arm)}
      {''.join(legs)}
    </body>
  </worldbody>
  <actuator>{''.join(acts)}</actuator>
</mujoco>"""


class Skater:
    def __init__(self, B, **kw):
        import mujoco
        self.mj = mujoco
        self.B = B
        self.m = mujoco.MjModel.from_xml_string(build_xml(B, **kw))
        self.d = mujoco.MjData(self.m)
        self.jadr = {}
        for side in "LR":
            self.jadr[side] = [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"{j}{side}")]
                               for j in ("sf", "sa", "sr", "el")]
        self.trunk = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.poses = {side: {} for side in "LR"}
        for p in ("open", "tuck", "up", "side", "uppar", "half"):
            self.add_pose(p)

    @staticmethod
    def mirror(q):
        return np.array([q[0], q[1], -q[2], q[3]])

    def add_pose(self, p):
        """左腕を逆運動学で解き、右腕は鏡像（腕を組む tuck だけは右腕も解く。左が上）."""
        if p in self.poses["L"]:
            return
        qL = self.ik("L", p)
        self.poses["L"][p] = qL
        self.poses["R"][p] = self.ik("R", p, guess=self.mirror(qL)) if p == "tuck" else self.mirror(qL)

    # ---- 腕の姿勢: 上腕と前腕の向き（胴体の座標、左腕。右は y を反転）
    def pose_dirs(self, p, side):
        B = self.B
        S = B["seg"]
        La = S["upperarm"]["len"] + S["forearm"]["len"] + 0.5 * S["hand"]["len"]
        sg = 1 if side == "L" else -1
        if p == "open":       # 踏み切り・着氷: 横やや前に伸ばす
            u = f = np.array([0.35, 1.0, 0.0])
        elif p == "side":     # 真横に伸ばす（T の字）
            u = f = np.array([0.0, 1.0, 0.0])
        elif p == "half":     # 半分締め: 肘を横に張り、手を胸の前
            u = np.array([0.15, 1.0, -0.6]); f = np.array([0.6, -0.9, 0.25])
        elif p == "tuck":     # 胸の前で腕を組む（左腕を上に）
            u = np.array([0.42, -0.05, -0.9]); f = np.array([0.25, -1.0, 0.30 + (0.06 if side == "L" else -0.02)])
        elif p == "up":       # 頭の上で手を合わせる
            u = f = np.array([0.0, -B["wsh"] + 0.02, np.sqrt(La**2 - (B["wsh"] - 0.02) ** 2)])
        elif p == "uppar":    # 肩幅のまま真上
            u = f = np.array([0.0, 0.0, 1.0])
        elif p.startswith("v"):  # 真上から外へ α 度（v0 = 肩幅で真上、v90 = 真横）
            a = np.radians(float(p[1:]))
            u = f = np.array([0.0, np.sin(a), np.cos(a)])
        else:
            raise KeyError(p)
        u = u / np.linalg.norm(u); f = f / np.linalg.norm(f)
        u = u * np.array([1, sg, 1]); f = f * np.array([1, sg, 1])
        return u, f

    def _fk(self, side, q):
        """肩からの肘・手首の位置（胴体の座標）. MJCF の関節の並びと同じ."""
        sg = 1 if side == "L" else -1
        S = self.B["seg"]
        def R(axis, a):
            axis = np.asarray(axis, float)
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
        R1 = R([0, 1, 0], q[0]) @ R([sg, 0, 0], q[1]) @ R([0, 0, 1], q[2])
        e = R1 @ np.array([0, 0, -S["upperarm"]["len"]])
        R2 = R1 @ R([0, -1, 0], q[3])
        w = e + R2 @ np.array([0, 0, -S["forearm"]["len"]])
        return e, w

    def ik(self, side, p, guess=None):
        u, f = self.pose_dirs(p, side)
        S = self.B["seg"]
        e_t = u * S["upperarm"]["len"]
        w_t = e_t + f * S["forearm"]["len"]
        best = None
        starts = [guess] if guess is not None else [[0, 0.3, 0, 0.3], [-1.0, 0.5, 0.5, 1.5], [0.5, 1.5, -0.5, 0.5], [0, 2.5, 0, 0.1], [-0.8, 1.2, 1.2, 1.8]]
        for q0 in starts:
            res = least_squares(lambda q: np.r_[self._fk(side, q)[0] - e_t, self._fk(side, q)[1] - w_t, 1e-3 * np.asarray(q)],
                                q0, bounds=([-4, -4, -4, 0.0], [4, 4, 4, 2.7]))
            if best is None or res.cost < best.cost:
                best = res
        return best.x

    def set_arms(self, qL, qR):
        for side, q in (("L", qL), ("R", qR)):
            for a, v in zip(self.jadr[side], q):
                self.d.qpos[a] = v

    def inertia(self, pL, pR=None, qL=None, qR=None):
        """姿勢の全身の慣性テンソル（重心まわり、胴体の座標）と重心."""
        mj = self.mj
        m, d = self.m, self.d
        d.qpos[:] = 0; d.qpos[3] = 1.0
        qL = self.poses["L"][pL] if qL is None else qL
        qR = self.poses["R"][pR or pL] if qR is None else qR
        self.set_arms(qL, qR)
        mj.mj_kinematics(m, d); mj.mj_comPos(m, d)
        mass = m.body_mass[1:]
        pos = d.xipos[1:]
        c = (mass[:, None] * pos).sum(0) / mass.sum()
        I = np.zeros((3, 3))
        for i in range(1, m.nbody):
            Rm = d.ximat[i].reshape(3, 3)
            r = d.xipos[i] - c
            I += Rm @ np.diag(m.body_inertia[i]) @ Rm.T + m.body_mass[i] * (r @ r * np.eye(3) - np.outer(r, r))
        return I, c

    # ---- 飛ぶ
    def arm_plan(self, schedule, t_air, t_tr=T_TR, t_open=T_OPEN, p0="open"):
        """腕の関節角の指令を、5 次の多項式の移行の列にする. 返り値は t → (q, qd, qdd)（左 4 + 右 4）.

        移行の途中で次の姿勢に変わるときは、その時点の位置・速度・加速度を引き継いで 5 次式を作り直す（F70）。
        """
        nb = len(schedule)
        tb = (t_air - t_open) / nb
        Minv = np.linalg.inv(np.array([[1.0, 1, 1], [3, 4, 5], [6, 12, 20]]))
        def seg_make(t0, q0, v0, a0, q1):
            T = t_tr
            rhs = np.array([q1 - q0 - v0 * T - 0.5 * a0 * T**2, -(v0 * T + a0 * T**2), -a0 * T**2])
            c = Minv @ rhs
            return (t0, q0, v0, a0, c)
        def seg_eval(sg, t):
            t0, q0, v0, a0, c = sg
            T = t_tr
            s = min(max((t - t0) / T, 0.0), 1.0)
            q = q0 + v0 * T * s + 0.5 * a0 * T**2 * s**2 + c[0] * s**3 + c[1] * s**4 + c[2] * s**5
            if s >= 1.0:
                return q, 0 * q, 0 * q
            dq = v0 * T + a0 * T**2 * s + 3 * c[0] * s**2 + 4 * c[1] * s**3 + 5 * c[2] * s**4
            ddq = a0 * T**2 + 6 * c[0] * s + 12 * c[1] * s**2 + 20 * c[2] * s**3
            return q, dq / T, ddq / T**2
        segs = {}
        for k, side in enumerate("LR"):
            P = self.poses[side]
            events = [(i * tb, P[schedule[i][k]]) for i in range(nb)] + [(t_air - t_open, P["open"])]
            z = 0 * P[p0]
            lst = [seg_make(-1.0, P[p0], z, z, P[p0])]
            goal = P[p0]
            for te, tgt in events:
                if np.allclose(tgt, goal):
                    continue
                q, dq, ddq = seg_eval(lst[-1], te)
                lst.append(seg_make(te, q, dq, ddq, tgt))
                goal = tgt
            segs[side] = lst
        def f(t):
            out = []
            for side in "LR":
                lst = segs[side]
                i = max(j for j in range(len(lst)) if lst[j][0] <= t)
                out.append(seg_eval(lst[i], t))
            return tuple(np.r_[out[0][k], out[1][k]] for k in range(3))
        return f

    def _base_vel(self, qpos, qa_d, Lw, Pw):
        """腕の速さ qa_d を与え、全体の角運動量（重心まわり、世界）Lw と運動量 Pw を満たす自由関節の速度を解く."""
        mj = self.mj
        m, d = self.m, self.d
        d.qpos[:] = qpos
        mj.mj_kinematics(m, d); mj.mj_comPos(m, d); mj.mj_crb(m, d)
        M = np.zeros((m.nv, m.nv))
        mj.mj_fullM(m, d, M)
        R = d.xmat[self.trunk].reshape(3, 3)
        o = d.xpos[self.trunk]
        c = d.subtree_com[self.trunk]
        # 自由関節の一般化運動量 = [世界の運動量 P, 胴体の座標で表した原点 o まわりの角運動量]
        h = np.r_[Pw, R.T @ (Lw + np.cross(c - o, Pw))]
        vb = np.linalg.solve(M[:6, :6], h - M[:6, 6:] @ qa_d)
        return vb

    def fly(self, schedule, L, t_air, t_tr=T_TR, t_open=T_OPEN, dt=1e-3, return_trace=False, torque=True, p0="open"):
        """schedule: 区間ごとの (左の姿勢, 右の姿勢) の列（[0, t_air − t_open] を等分）. 最初は open から、最後は open へ.

        腕の関節角を時間の関数として与え（技術）、胴体の動きは運動量と角運動量の保存から解く（RK4）。
        返り値: 回転数（L の向き＝鉛直のまわりの胴体の向き）、着氷時の胴体の軸の傾き（度）、L の保存のずれ、肩の最大トルク.
        """
        mj = self.mj
        m, d = self.m, self.d
        for p in {x for pair in schedule for x in pair} | {p0}:
            self.add_pose(p)
        plan = self.arm_plan(schedule, t_air, t_tr, t_open, p0)
        mtot = m.body_subtreemass[self.trunk]
        Lw = np.array([0.0, 0.0, L])
        P0 = np.array([0.0, 0.0, mtot * G * t_air / 2])
        ia = np.r_[self.jadr["L"], self.jadr["R"]]          # 腕の qpos の番地（qvel も同じ並び: 7..14 → 6..13）
        iv = ia - 1
        def rhs(t, y):
            qa, qad, _ = plan(t)
            qpos = np.zeros(m.nq); qpos[:7] = y; qpos[ia] = qa
            vb = self._base_vel(qpos, qad, Lw, P0 + np.array([0, 0, -mtot * G * t]))
            qd = np.zeros(7)
            qd[:3] = vb[:3]
            w = vb[3:]
            qw = y[3:7]
            qd[3:7] = 0.5 * np.array([-qw[1:] @ w,
                                      qw[0] * w[0] + qw[2] * w[2] - qw[3] * w[1],
                                      qw[0] * w[1] + qw[3] * w[0] - qw[1] * w[2],
                                      qw[0] * w[2] + qw[1] * w[1] - qw[2] * w[0]])
            return qd, vb
        nst = int(round(t_air / dt))
        y = np.r_[0, 0, 1.5, 1.0, 0, 0, 0]
        heading, tilt, tr, vbs, ys = [], [], [], [], []
        for k in range(nst + 1):
            t = k * dt
            k1, vb = rhs(t, y)
            qw = y[3:7]
            R = np.zeros(9); mj.mju_quat2Mat(R, qw); R = R.reshape(3, 3)
            heading.append(np.arctan2(R[1, 0], R[0, 0]))
            tilt.append(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
            vbs.append(vb); ys.append(y.copy())
            if return_trace:
                tr.append([t, vb[5], tilt[-1]])
            if k == nst:
                break
            k2, _ = rhs(t + dt / 2, y + dt / 2 * k1)
            k3, _ = rhs(t + dt / 2, y + dt / 2 * k2)
            k4, _ = rhs(t + dt, y + dt * k3)
            y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            y[3:7] /= np.linalg.norm(y[3:7])
        out = dict(turns=float((np.unwrap(heading)[-1] - heading[0]) / (2 * np.pi)), tilt_land=float(tilt[-1]),
                   tilt_max=float(max(tilt)))
        if torque:
            # 肩・肘のトルク（逆動力学。胴体の加速度は差分）と、L の保存の確認（MuJoCo 自身の subtree_angmom）
            vbs = np.array(vbs)
            ab = np.gradient(vbs, dt, axis=0)
            taus, Ld = [], []
            stride = max(1, int(round(2e-3 / dt)))
            for k in range(0, nst + 1, stride):
                qa, qad, qadd = plan(k * dt)
                d.qpos[:7] = ys[k]; d.qpos[ia] = qa
                d.qvel[:6] = vbs[k]; d.qvel[iv] = qad
                d.qacc[:6] = ab[k]; d.qacc[iv] = qadd
                mj.mj_inverse(m, d)
                taus.append(d.qfrc_inverse[iv].copy())
                mj.mj_subtreeVel(m, d)
                Ld.append(np.abs(d.subtree_angmom[self.trunk] - Lw).max() / L)
            taus = np.array(taus)
            tt = np.arange(0, nst + 1, stride) * dt
            hold = (tt > t_tr + 0.02) & (tt < t_air - t_open - 0.02)
            tsh = np.linalg.norm(taus[:, [0, 1, 2]], axis=1), np.linalg.norm(taus[:, [4, 5, 6]], axis=1)
            tsh = np.maximum(*tsh)
            tel = np.abs(taus[:, [3, 7]]).max(1)
            close = tt <= t_tr + 0.02
            opn = tt >= t_air - t_open - 0.02
            out.update(tau_sh=float(tsh.max()), tau_el=float(tel.max()),
                       tau_sh_close=float(tsh[close].max()), tau_sh_open=float(tsh[opn].max()),
                       tau_sh_flight=float(tsh[~opn].max()),
                       tau_sh_hold=float(tsh[hold].max()) if hold.any() else 0.0,
                       tau_el_hold=float(tel[hold].max()) if hold.any() else 0.0, Ldev=float(max(Ld)))
            if return_trace:
                out["tau_trace"] = np.c_[tt, tsh, tel]
        if return_trace:
            out["trace"] = np.array(tr)
        return out


def std_schedule(p="tuck", nb=8):
    return [(p, p)] * nb


def quasi_static_turns(sk, p, L, t_air, t_tr=T_TR, t_open=T_OPEN, n=2000):
    """腕が動く間の相対の角運動量を無視した N = (L/2π)∫dt/I_zz（左右対称の姿勢）."""
    t = np.linspace(0, t_air, n)
    qo, qp = sk.poses["L"]["open"], sk.poses["L"][p]
    s1 = smooth_step(t / t_tr)[0]
    s2 = smooth_step((t - (t_air - t_open)) / t_tr)[0]
    lam = s1 * (1 - s2)
    I = np.array([sk.inertia(None, qL=qo + (qp - qo) * l, qR=sk.poses["R"]["open"] + (sk.poses["R"][p] - sk.poses["R"]["open"]) * l)[0][2, 2]
                  for l in lam])
    return float(np.trapezoid(L / I, t) / (2 * np.pi))


# ---------------------------------------------------------------- 1. 姿勢ごとの慣性モーメント
def run_poses(body=None, height=None, mass=None, boot=None, quiet=False):
    B = load_body(body or REF["body"], height or (None if body else REF["height"]),
                  mass or (None if body else REF["mass"]), REF["boot"] if boot is None and not body else boot)
    sk = Skater(B)
    res = {}
    for name, (pl, pr) in dict(tuck=("tuck", "tuck"), up=("up", "up"), uppar=("uppar", "uppar"),
                               tano=("up", "tuck"), half=("half", "half"), side=("side", "side"),
                               open=("open", "open")).items():
        I, c = sk.inertia(pl, pr)
        ev, V = np.linalg.eigh(I)
        ax = V[:, 0] * np.sign(V[2, 0])
        res[name] = dict(Izz=float(I[2, 2]), Ixx=float(I[0, 0]), Iyy=float(I[1, 1]), Ixz=float(I[0, 2]), Iyz=float(I[1, 2]),
                         Imin=float(ev[0]), axis_tilt_deg=float(np.degrees(np.arccos(abs(ax[2])))), com_z=float(c[2]))
    if not quiet:
        print(f"体: {B['name']}  身長 {B['H']:.3f} m  体重 {B['M']:.1f} kg  靴 {B['boot']:.1f} kg×2  肩の半幅 {B['wsh']:.3f} m")
        print(f"{'姿勢':6s} {'Izz':>7s} {'/tuck':>6s} {'Ixx':>6s} {'Iyy':>6s} {'Iyz':>7s} {'主軸の傾き':>8s}")
        for k, v in res.items():
            print(f"{k:6s} {v['Izz']:7.3f} {v['Izz'] / res['tuck']['Izz']:6.2f} {v['Ixx']:6.2f} {v['Iyy']:6.2f} {v['Iyz']:7.3f} {v['axis_tilt_deg']:8.2f}°")
    return B, sk, res


# ---------------------------------------------------------------- 2. MuJoCo の飛行と準静的な式の比較
def run_flight():
    B, sk, P = run_poses(quiet=True)
    lines = []
    out = {}
    for p in ("tuck", "half", "up", "uppar", "side"):
        r = sk.fly(std_schedule(p), L_NOM, T_AIR_NOM)
        qs = quasi_static_turns(sk, p, L_NOM, T_AIR_NOM)
        out[p] = dict(r, quasi=qs)
        lines.append(f"{p:6s} MuJoCo {r['turns']:.3f} 回転  準静的 {qs:.3f}  傾き(着氷) {r['tilt_land']:.2f}°  "
                     f"|ΔL|/L {r['Ldev']:.1e}  肩のトルク 締め {r['tau_sh_close']:.0f}／保持 {r['tau_sh_hold']:.0f}／開き {r['tau_sh_open']:.0f} N·m")
    r = sk.fly([("up", "tuck")] * 8, L_NOM, T_AIR_NOM)
    out["tano"] = r
    lines.append(f"tano   MuJoCo {r['turns']:.3f} 回転  傾き(着氷) {r['tilt_land']:.1f}°  最大 {r['tilt_max']:.1f}°  肩 {r['tau_sh']:.0f} N·m（肘 {r['tau_el']:.0f}）")
    r = sk.fly([("up", "tuck")] * 4 + [("tuck", "up")] * 4, L_NOM, T_AIR_NOM)
    out["tano_swap"] = r
    lines.append(f"tano 左右を途中で入れ替え {r['turns']:.3f} 回転  傾き(着氷) {r['tilt_land']:.1f}°  最大 {r['tilt_max']:.1f}°")
    # 計算の誤差: 時間の刻みと積分法
    for dt in (2.5e-4, 1e-3, 2e-3):
        r2 = sk.fly(std_schedule("up"), L_NOM, T_AIR_NOM, dt=dt)
        out[f"up_dt{dt}"] = r2
        lines.append(f"up RK4 dt {dt * 1e3:.2f} ms {r2['turns']:.5f} 回転  |ΔL|/L {r2['Ldev']:.1e}")
    print("\n".join(lines))
    json.dump(out, open(DATA / "bangai03_flight.json", "w"), indent=1, default=float)
    return out


# ---------------------------------------------------------------- 3. 体格・技術・運動能力の切り分け
def k_factor(sk, p, L=L_NOM, t_air=T_AIR_NOM, t_tr=T_TR, ref=None):
    """腕の姿勢 p で、普通の締め（tuck）と同じ回転数にするのに要る L·t の倍率（N は L に比例するので比で出る）."""
    n_ref = ref if ref is not None else sk.fly(std_schedule("tuck"), L, t_air, t_tr=t_tr)["turns"]
    n_p = sk.fly(std_schedule(p), L, t_air, t_tr=t_tr)["turns"]
    return n_ref / n_p, n_ref, n_p


def run_factors():
    out = {}
    B0 = load_body(REF["body"], REF["height"], REF["mass"], REF["boot"])
    sk0 = Skater(B0)
    base = {p: k_factor(sk0, p) for p in ("up", "uppar", "half", "side")}
    out["base"] = {p: dict(K=v[0], n_tuck=v[1], n_p=v[2]) for p, v in base.items()}
    print("基準の体（1.70 m, 62 kg）: tuck と同じ回転数に要る L·t の倍率 K")
    for p, v in base.items():
        print(f"  {p:6s} K = {v[0]:.3f}  （tuck {v[1]:.2f} 回転、{p} {v[2]:.2f} 回転）")
    # 体格: 身長（同じ体型で相似に大きく）と比率
    body = {}
    for H in (1.55, 1.62, 1.70, 1.78, 1.85):
        M = REF["mass"] * (H / REF["height"]) ** 2     # BMI 一定
        sk = Skater(load_body(REF["body"], H, M, REF["boot"]))
        k, nt, nu = k_factor(sk, "up")
        It = sk.inertia("tuck")[0][2, 2]
        body[f"H{H}"] = dict(H=H, M=M, K=k, n_tuck=nt, n_up=nu, I_tuck=It)
    for key, vals in (("shoulder", (0.9, 1.1)), ("arm", (0.95, 1.05)), ("girth", (0.9, 1.1)), ("armmass", (0.85, 1.15))):
        for v in vals:
            sk = Skater(load_body(REF["body"], REF["height"], REF["mass"], REF["boot"], scale={key: v}))
            k, nt, nu = k_factor(sk, "up")
            body[f"{key}{v}"] = dict(K=k, n_tuck=nt, n_up=nu)
    for nm, f in (("female", "deleva1996_female.json"),):
        sk = Skater(load_body(f, REF["height"], REF["mass"], REF["boot"]))
        k, nt, nu = k_factor(sk, "up")
        body[nm] = dict(K=k, n_tuck=nt, n_up=nu)
    out["body"] = body
    print("体格（up の K）:")
    for k, v in body.items():
        extra = f"  tuck {v['n_tuck']:.2f} 回転（同じ L のとき）" if "H" in v else ""
        print(f"  {k:12s} K = {v['K']:.3f}{extra}")
    # 技術: 腕を変える速さ、上げている時間の割合
    tech = {}
    for ttr in (0.06, 0.10, 0.15, 0.20):
        k, nt, nu = k_factor(sk0, "up", t_tr=ttr)
        tech[f"ttr{ttr}"] = dict(K=k, n_tuck=nt, n_up=nu)
    nt = sk0.fly(std_schedule("tuck"), L_NOM, T_AIR_NOM)["turns"]
    for nup in range(0, 9, 2):
        sch = [("tuck", "tuck")] * ((8 - nup) // 2) + [("up", "up")] * nup + [("tuck", "tuck")] * ((8 - nup) - (8 - nup) // 2)
        r = sk0.fly(sch, L_NOM, T_AIR_NOM)
        tech[f"upbins{nup}"] = dict(K=nt / r["turns"], frac=nup / 8)
    out["tech"] = tech
    print("技術:")
    for k, v in tech.items():
        print(f"  {k:10s} K = {v['K']:.3f}")
    # 運動能力: 文献の範囲で L·t がどれだけ動くか
    ath = dict(L_sd=L_SD / L_NOM, L_max=L_MAX / L_NOM,
               t_lo=T_AIR_LO / T_AIR_NOM, t_hi=T_AIR_HI / T_AIR_NOM,
               Lt_hi=(L_NOM + L_SD) * T_AIR_HI / (L_NOM * T_AIR_NOM),
               Lt_max=L_MAX * T_AIR_HI / (L_NOM * T_AIR_NOM))
    # 実際の回転数（t が変わると移行の時間の割合も変わるので MuJoCo で）
    for L, t in ((L_NOM, T_AIR_NOM), (L_NOM + L_SD, T_AIR_NOM), (L_NOM, T_AIR_HI), (L_NOM + L_SD, T_AIR_HI), (L_MAX, T_AIR_HI)):
        ath[f"n_up_L{L}_t{t}"] = sk0.fly(std_schedule("up"), L, t)["turns"]
        ath[f"n_tuck_L{L}_t{t}"] = sk0.fly(std_schedule("tuck"), L, t)["turns"]
    out["ath"] = ath
    print("運動能力（文献の範囲）:", {k: round(v, 3) for k, v in ath.items()})
    # 腕の角度: 真上から外へ α 度（肩幅の V 字）
    vs = {}
    for a in (0, 5, 10, 15, 20, 30, 45, 60, 90):
        r = sk0.fly(std_schedule(f"v{a}"), L_NOM, T_AIR_NOM)
        vs[a] = dict(K=nt / r["turns"], Izz=float(sk0.inertia(f"v{a}")[0][2, 2]), tau_hold=r["tau_sh_hold"])
    out["vangle"] = vs
    print("V 字の角度:", {k: round(v["K"], 3) for k, v in vs.items()})
    # 肩の力（接触なし、腕を肩の関節のトルクだけで動かす・保つ）: 移行の時間ごと
    st = {}
    for p in ("tuck", "up", "uppar", "half"):
        for ttr in (0.10, 0.15, 0.20, 0.30):
            r = sk0.fly(std_schedule(p), L_NOM, T_AIR_NOM, t_tr=ttr)
            st[f"{p}_{ttr}"] = dict(turns=r["turns"], close=r["tau_sh_close"], hold=r["tau_sh_hold"], open=r["tau_sh_open"],
                                    el=r["tau_el"])
    # 腕を氷の上で先に上げておく（踏み切りの瞬間に up）
    for p0 in ("up", "tuck"):
        r = sk0.fly(std_schedule("up"), L_NOM, T_AIR_NOM, t_tr=0.2, p0=p0)
        st[f"p0{p0}_up_0.2"] = dict(turns=r["turns"], close=r["tau_sh_close"], hold=r["tau_sh_hold"], open=r["tau_sh_open"],
                                    el=r["tau_el"])
    out["strength"] = st
    print("肩のトルク（N·m、片腕、締め/保持/開き）:")
    for k, v in st.items():
        print(f"  {k:14s} {v['turns']:.3f} 回転  {v['close']:.0f} / {v['hold']:.0f} / {v['open']:.0f}")
    # 踏み切りの慣性モーメントのうち腕の分（L を作る腕の役割の目安。脚はそろえたままなので上限側）
    Io, Iu = sk0.inertia("open")[0][2, 2], sk0.inertia("up")[0][2, 2]
    out["takeoff"] = dict(I_open=Io, I_up=Iu, omega_open=L_NOM / Io / (2 * np.pi), omega_up=L_NOM / Iu / (2 * np.pi))
    json.dump(out, open(DATA / "bangai03_factors.json", "w"), indent=1, default=float)
    return out


# ---------------------------------------------------------------- 4. 誤差の評価
def run_errors(n_mc=200, seed=7):
    """統計（体節の値のばらつき・文献の L と t のばらつき）、計算（刻み・積分法・準静的）、系統（仮定の選び方）."""
    rng = np.random.default_rng(seed)
    B0 = load_body(REF["body"], REF["height"], REF["mass"], REF["boot"])
    sk0 = Skater(B0)
    It0 = {p: sk0.inertia(p)[0][2, 2] for p in ("tuck", "up", "open")}
    # 統計: de Leva は平均だけなので、質量の比 ±10%、回転半径 ±5%、目印 ±5%（1σ、仮定）で体を揺らす。
    # 1 回の飛行は 30〜60 ms なので、MuJoCo の K を直接 MC する。
    Ks, Its, Ius = [], [], []
    t0 = time.time()
    path = BODY_DIR / REF["body"]
    cfg0 = json.load(open(path))
    tmp = DATA / "bangai03_tmp_body.json"
    for i in range(n_mc):
        cfg = json.loads(json.dumps(cfg0))
        for nm in SEGS:
            s = cfg["segments"][nm]
            s["mass"] *= np.exp(0.10 * rng.standard_normal())
            ft, fl = np.exp(0.05 * rng.standard_normal(2))     # 横の 2 つの回転半径は同じ倍率（三角不等式を保つ、F68）
            r0, r1, r2 = s["r"][0] * ft, s["r"][1] * ft, s["r"][2] * fl
            r2 = max(r2, 1.001 * np.sqrt(abs(r0**2 - r1**2)))   # 手は de Leva の値でも境目に近い
            s["r"] = [r0, r1, r2]
            if s["len"] is not None:
                s["len"] *= np.exp(0.03 * rng.standard_normal())
        for k in cfg["landmarks_frac"]:
            cfg["landmarks_frac"][k] *= np.exp(0.05 * rng.standard_normal())
        json.dump(cfg, open(tmp, "w"))
        sk = Skater(load_body(tmp, REF["height"], REF["mass"], REF["boot"]))
        k, nt, nu = k_factor(sk, "up")
        Ks.append(k); Its.append(sk.inertia("tuck")[0][2, 2]); Ius.append(sk.inertia("up")[0][2, 2])
    tmp.unlink()
    Ks, Its, Ius = map(np.array, (Ks, Its, Ius))
    stat = dict(K_mean=Ks.mean(), K_sd=Ks.std(ddof=1), I_tuck_mean=Its.mean(), I_tuck_sd=Its.std(ddof=1),
                I_up_mean=Ius.mean(), I_up_sd=Ius.std(ddof=1), n=n_mc, sec=time.time() - t0)
    # 文献の L と t のばらつき → 普通の 3A の回転数のばらつき（MC、N は L に比例し t には MuJoCo で）
    n_t = {t: sk0.fly(std_schedule("tuck"), L_NOM, t)["turns"] for t in (0.66, 0.70, 0.74, 0.78)}
    ts = np.array(list(n_t)); ns = np.array(list(n_t.values()))
    cf = np.polyfit(ts, ns, 2)
    Lm = L_NOM + L_SD * rng.standard_normal(4000)
    tm = rng.uniform(T_AIR_LO, T_AIR_HI, 4000)
    nm_ = Lm / L_NOM * np.polyval(cf, tm)
    stat.update(n_tuck_lit_mean=nm_.mean(), n_tuck_lit_sd=nm_.std(ddof=1))
    # 計算: 刻みと積分法、準静的な式
    comp = {}
    ref = sk0.fly(std_schedule("up"), L_NOM, T_AIR_NOM)["turns"]
    for dt in (2.5e-4, 1e-3, 2e-3):
        r = sk0.fly(std_schedule("up"), L_NOM, T_AIR_NOM, dt=dt)
        comp[f"dt{dt}"] = dict(dturns=r["turns"] - ref, Ldev=r["Ldev"])
    comp["quasi_up"] = quasi_static_turns(sk0, "up", L_NOM, T_AIR_NOM) - ref
    # 系統: 仮定を 1 つずつ変える
    syst = {}
    def K_of(**kw):
        b = load_body(kw.get("body", REF["body"]), REF["height"], REF["mass"], kw.get("boot", REF["boot"]), scale=kw.get("scale"))
        return k_factor(Skater(b), "up", t_tr=kw.get("t_tr", T_TR))[0]
    K0 = K_of()
    syst["boot0"] = K_of(boot=0.0) - K0
    syst["female_table"] = K_of(body="deleva1996_female.json") - K0
    syst["shoulder+10%"] = K_of(scale={"shoulder": 1.1}) - K0
    syst["t_tr0.15"] = K_of(t_tr=0.15) - K0
    syst["caliper_template"] = K_of(body="bodychan_template.json") - K0
    # 締めの姿勢の定義（tuck を半分締め half に）→ 要る L の絶対値への影響
    n_half = sk0.fly(std_schedule("half"), L_NOM, T_AIR_NOM)["turns"]
    n_tuck = sk0.fly(std_schedule("tuck"), L_NOM, T_AIR_NOM)["turns"]
    syst["tuck_as_half_ntuck_ratio"] = n_half / n_tuck
    out = dict(K0=K0, stat=stat, comp=comp, syst=syst, It0=It0, n_t=n_t)
    json.dump(out, open(DATA / "bangai03_errors.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=lambda x: round(float(x), 5)))
    return out


# ---------------------------------------------------------------- 5. UC-AutoMaiL の焼きなましで腕の時間割
def run_anneal(budget=250, seeds=(1, 2, 3, 4, 5), L=None, t_tr=0.15):
    """8 区間 × 左右 2 ビット（1 = その腕を頭上、0 = 胸の前）の 16 ビットを、UC-AutoMaiL の search_subsets で探す.

    評価 = 腕を頭上にしている時間の割合（左右の平均）
           − 4 ×（普通の 3A の回転数 − 0.02 に足りない回転）
           − 0.02 ×（着氷の傾き − 10°）₊ − 0.004 ×（肩のトルク − 普通の 3A の 1.5 倍）₊
    肩のトルクは着氷前に腕を開く区間を除いた最大（接触なし）。
    """
    import uc_automail as UC
    ANNEALING, RANDOM = UC.annealing.ANNEALING, UC.annealing.RANDOM
    SubsetSpace, search_subsets = UC.annealing.SubsetSpace, UC.annealing.search_subsets   # diy_automl.annealing
    B0 = load_body(REF["body"], REF["height"], REF["mass"], REF["boot"])
    sk = Skater(B0)
    L = L or L_NOM
    ref = sk.fly(std_schedule("tuck"), L_NOM, T_AIR_NOM, t_tr=t_tr, dt=2e-3)
    n_req = ref["turns"] - 0.02
    tau_cap = 1.5 * ref["tau_sh_flight"]
    nb, width = 8, 16
    cache = {}
    def schedule(mask):
        return [("up" if mask >> (2 * i) & 1 else "tuck", "up" if mask >> (2 * i + 1) & 1 else "tuck") for i in range(nb)]
    def evaluate(mask):
        t0 = time.time()
        if mask not in cache:
            r = sk.fly(schedule(mask), L, T_AIR_NOM, t_tr=t_tr, dt=2e-3)
            frac = bin(mask).count("1") / width
            short = max(0.0, n_req - r["turns"])
            score = (frac - 4.0 * short - 0.02 * max(0.0, r["tilt_land"] - 10.0)
                     - 0.004 * max(0.0, r["tau_sh_flight"] - tau_cap))
            cache[mask] = (score, r, frac)
        return cache[mask][0], time.time() - t0
    space = SubsetSpace(width, width, budget)
    res = {}
    t_start = time.time()
    for method in (ANNEALING, RANDOM):
        best = []
        for sd in seeds:
            settings = dict(search_seed=sd, budget=budget, initial_temperature=0.05, final_temperature=0.002)
            hist = search_subsets(space, method, settings, evaluate)
            b = hist[hist[-1].best]
            sc, r, frac = cache[b.mask]
            best.append(dict(mask=b.mask, score=sc, frac=frac, turns=r["turns"], tilt_land=r["tilt_land"],
                             tilt_max=r["tilt_max"], tau=r["tau_sh_flight"],
                             sched=["".join("U" if x == "up" else "t" for x in s) for s in schedule(b.mask)],
                             curve=[cache[h.mask][0] for h in hist]))
        res[method] = best
        sc = np.array([b["score"] for b in best])
        print(f"{method:22s} 最良の点 {sc.mean():.3f} ± {sc.std(ddof=1):.3f}（{len(seeds)} 回）")
        for b in best:
            print(f"   {' '.join(b['sched'])}  頭上の割合 {b['frac']:.2f}  {b['turns']:.3f} 回転  傾き {b['tilt_land']:.1f}°  肩 {b['tau']:.0f} N·m")
    hand = {}
    cand = {"all_up": (1 << width) - 1, "all_tuck": 0}
    for k in range(1, nb + 1):
        cand[f"sym_last{k}"] = sum(3 << (2 * i) for i in range(nb - k, nb))
        cand[f"sym_first{k}"] = sum(3 << (2 * i) for i in range(k))
        cand[f"L_only{k}"] = sum(1 << (2 * i) for i in range(k))
    for nm, mask in cand.items():
        evaluate(mask)
        sc, r, frac = cache[mask]
        hand[nm] = dict(score=sc, turns=r["turns"], frac=frac, tilt=r["tilt_land"], tau=r["tau_sh_flight"])
    bestn = max(hand, key=lambda k: hand[k]["score"])
    print("手で作った候補の最良:", bestn, {k: round(v, 3) for k, v in hand[bestn].items()})
    out = dict(L=L, t_tr=t_tr, n_req=n_req, tau_cap=tau_cap, budget=budget, results=res, hand=hand, n_eval=len(cache),
               uc_version=UC.__version__, sec=time.time() - t_start)
    json.dump(out, open(DATA / "bangai03_anneal.json", "w"), indent=1, default=float)
    return out


# ---------------------------------------------------------------- 6. 図
def run_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    B, sk, P = run_poses(quiet=True)
    fac = json.load(open(DATA / "bangai03_factors.json"))
    ann = json.load(open(DATA / "bangai03_anneal.json")) if (DATA / "bangai03_anneal.json").exists() else None
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    # (a) 姿勢の慣性モーメント
    ax = axs[0, 0]
    names = ["tuck", "half", "up", "uppar", "tano", "side", "open"]
    jp = dict(tuck="胸で組む", half="半分", up="頭上で合わせる", uppar="肩幅で真上", tano="片腕だけ上", side="真横", open="踏み切り")
    v = [P[n]["Izz"] for n in names]
    ax.bar(range(len(names)), v, color=["C0", "C7", "C3", "C3", "C1", "C7", "C7"])
    ax.set_xticks(range(len(names))); ax.set_xticklabels([jp[n] for n in names], rotation=30, fontsize=9)
    ax.set_ylabel("縦の軸まわりの慣性モーメント [kg m²]")
    ax.axhline(L_NOM / (4.9 * 2 * np.pi), color="k", ls="--", lw=1)
    ax.text(0.02, L_NOM / (4.9 * 2 * np.pi) + 0.05, "文献の L / ω から逆算（32.8 / 4.9 rev/s）", fontsize=8)
    ax.set_title("(a) 腕の姿勢と慣性モーメント（1.70 m・62 kg）")
    # (b) 飛行中の回転速度
    ax = axs[0, 1]
    for p, c in (("tuck", "C0"), ("up", "C3"), ("side", "C7")):
        r = sk.fly(std_schedule(p), L_NOM, T_AIR_NOM, return_trace=True)
        tr = r["trace"]
        ax.plot(tr[:, 0], tr[:, 1] / (2 * np.pi), c, label=f"{jp[p]}: {r['turns']:.2f} 回転")
    r = sk.fly([("up", "tuck")] * 4 + [("tuck", "up")] * 4, L_NOM, T_AIR_NOM, return_trace=True)
    ax.plot(r["trace"][:, 0], r["trace"][:, 1] / (2 * np.pi), "C1", label=f"片腕（途中で左右交代）: {r['turns']:.2f}")
    ax.set_xlabel("踏み切りからの時間 [s]"); ax.set_ylabel("胴体の縦の軸まわりの回転速度 [rev/s]")
    ax.legend(fontsize=8); ax.set_title(f"(b) MuJoCo の飛行（L {L_NOM} kg m²/s, 滞空 {T_AIR_NOM} s）")
    # (c) 片腕の非対称と軸の傾き
    ax = axs[0, 2]
    for sch, lab, c in (([("up", "tuck")] * 8, "片腕だけ上（ずっと）", "C1"),
                        ([("up", "tuck")] * 4 + [("tuck", "up")] * 4, "途中で左右を入れ替え", "C2"),
                        (std_schedule("up"), "両腕を上", "C3")):
        r = sk.fly(sch, L_NOM, T_AIR_NOM, return_trace=True)
        ax.plot(r["trace"][:, 0], r["trace"][:, 2], c, label=lab)
    ax.set_xlabel("時間 [s]"); ax.set_ylabel("胴体の軸の傾き [°]"); ax.legend(fontsize=8)
    ax.set_title("(c) 左右の非対称が回転軸を傾ける（AMBAC）")
    # (d) 腕の角度（真上から外へ）と K、運動能力で出せる倍率
    ax = axs[1, 0]
    vs = fac["vangle"]
    aa = sorted(int(k) for k in vs)
    ax.plot(aa, [vs[str(k)]["K"] for k in aa], "o-", color="C3", label="肩幅の V 字（MuJoCo）")
    ax.axhline(fac["base"]["up"]["K"], color="C3", ls="--", lw=1)
    ax.text(40, fac["base"]["up"]["K"] + 0.02, f"頭上で手を合わせる K {fac['base']['up']['K']:.2f}", color="C3", fontsize=8)
    ath = fac["ath"]
    n0 = ath[f"n_tuck_L{L_NOM}_t{T_AIR_NOM}"]
    for nm, val in (("L +1σ", ath[f"n_tuck_L{L_NOM + L_SD}_t{T_AIR_NOM}"] / n0),
                    ("L +1σ・滞空 0.76 s", ath[f"n_tuck_L{L_NOM + L_SD}_t{T_AIR_HI}"] / n0),
                    ("最大の L 37.5・0.76 s", ath[f"n_tuck_L{L_MAX}_t{T_AIR_HI}"] / n0)):
        ax.axhline(val, color="C1", ls=":", lw=1); ax.text(55, val + 0.01, nm, fontsize=8, color="C1")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel("腕の角度（真上から外へ）[°]"); ax.set_ylabel("K（胸で組むのと同じ回転に要る L·t の倍率）")
    ax.set_title("(d) 腕を真上から 15〜20° より開くと能力で補えない"); ax.legend(fontsize=8, loc="upper left")
    # (e) 肩のトルク（接触なし）と腕を変える時間
    ax = axs[1, 1]
    st = fac["strength"]
    for p, c in (("tuck", "C0"), ("up", "C3"), ("uppar", "C4"), ("half", "C7")):
        xs = [0.10, 0.15, 0.20, 0.30]
        ax.plot(xs, [st[f"{p}_{x}"]["close"] for x in xs], "o-", color=c, label=f"{jp[p]}: 締めるとき")
        ax.plot(xs, [st[f"{p}_{x}"]["hold"] for x in xs], ":", color=c)
    ax.axhspan(40, 75, color="C2", alpha=0.15)
    ax.text(0.12, 120, "緑の帯: 等尺性の外転の力 2.18 N/kg（Bradley & Pierpoint 2023）\n× 腕の長さ 0.3〜0.55 m の目安", fontsize=7)
    ax.set_xlabel("腕の姿勢を変える時間 [s]"); ax.set_ylabel("片腕の肩のトルクの最大 [N·m]（点線は保持）")
    ax.set_title("(e) 力: 回転中に腕を頭上へ上げるのは締めの 2〜3 倍"); ax.legend(fontsize=7)
    # (f) 焼きなまし
    ax = axs[1, 2]
    if ann:
        for meth, c, lab in (("Simulated annealing", "C3", "焼きなまし"), ("Random search", "C7", "一様乱択")):
            for j, b in enumerate(ann["results"][meth]):
                ax.plot(np.maximum.accumulate(b["curve"]), color=c, lw=1, alpha=0.8, label=lab if j == 0 else None)
        ax.set_xlabel("評価の回数（MuJoCo の飛行）"); ax.set_ylabel("それまでの最良の点")
        ax.legend(fontsize=8)
        ax.set_title(f"(f) UC-AutoMaiL {ann['uc_version']} で腕の時間割を探す")
    fig.tight_layout(); fig.savefig(OUT / "bangai03_axel_arms.png", dpi=130)
    print("図 → figures/field/bangai03_axel_arms.png")
    # 姿勢の絵
    fig, axs = plt.subplots(1, 5, figsize=(14, 4))
    import mujoco
    for ax, (nm, pl, pr) in zip(axs, (("胸で組む", "tuck", "tuck"), ("頭上で合わせる", "up", "up"), ("片腕だけ上", "up", "tuck"),
                                      ("真横", "side", "side"), ("踏み切り", "open", "open"))):
        sk.inertia(pl, pr)
        d = sk.d
        S = B["seg"]
        o = d.xpos[sk.trunk]
        zc7 = B["H"] - S["head"]["len"] - B["hipH"]
        zsh = zc7 - B["sh_below"]
        ax.plot([o[1], o[1]], [o[2], o[2] + zc7], "k-", lw=4)                       # 背骨
        ax.plot([o[1] - B["wsh"], o[1] + B["wsh"]], [o[2] + zsh] * 2, "k-", lw=4)    # 肩
        ax.add_patch(plt.Circle((o[1], o[2] + zc7 + 0.5 * S["head"]["len"]), 0.5 * S["head"]["len"] * 0.8, fill=False, lw=2))
        for b in range(2, sk.m.nbody):
            nmb = sk.m.body(b).name
            if nmb.startswith(("hd", "boot", "ua")):
                continue
            par = sk.m.body_parentid[b]
            p0, p1 = d.xpos[par], d.xpos[b]
            ax.plot([p0[1], p1[1]], [p0[2], p1[2]], "k-", lw=3)
        for side in "LR":   # 足先（足首から下）
            ft = sk.m.body(f"ft{side}").id
            Rm = d.xmat[ft].reshape(3, 3)
            toe = d.xpos[ft] + Rm @ np.array([0, 0, -0.5 * S["foot"]["len"]])
            ax.plot([d.xpos[ft][1], toe[1]], [d.xpos[ft][2], toe[2]], "k-", lw=3)
        for side in "LR":
            fa = sk.m.body(f"fa{side}").id
            Rm = d.xmat[fa].reshape(3, 3)
            hand = d.xpos[fa] + Rm @ np.array([0, 0, -B["seg"]["forearm"]["len"]])
            ax.plot([d.xpos[fa][1], hand[1]], [d.xpos[fa][2], hand[2]], "k-", lw=3)
            for nmb in ("ua", "fa"):
                bb = sk.m.body(f"{nmb}{side}").id
                ax.plot(d.xpos[bb][1], d.xpos[bb][2], "o", color="C3", ms=4)
        I, _ = sk.inertia(pl, pr)
        ax.set_title(f"{nm}\nIzz {I[2, 2]:.2f} kg m²", fontsize=10)
        ax.set_aspect("equal"); ax.set_xlim(-0.95, 0.95); ax.set_ylim(-0.95, 1.3); ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "bangai03_poses.png", dpi=110)
    print("図 → figures/field/bangai03_poses.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="poses")
    ap.add_argument("--body"); ap.add_argument("--height", type=float); ap.add_argument("--mass", type=float)
    ap.add_argument("--boot", type=float)
    a = ap.parse_args()
    if a.mode == "poses":
        run_poses(a.body, a.height, a.mass, a.boot)
    elif a.mode == "flight":
        run_flight()
    elif a.mode == "factors":
        run_factors()
    elif a.mode == "errors":
        run_errors()
    elif a.mode == "anneal":
        run_anneal()
    elif a.mode == "figure":
        run_figure()
    elif a.mode == "all":
        run_poses(); run_flight(); run_factors(); run_errors(); run_anneal(); run_figure()
