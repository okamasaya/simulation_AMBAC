"""肩（上肢帯＋肩甲上腕関節）の簡易運動学モデル.

座標（胸郭固定）: x 前方, y 左, z 上。原点は胸の中心。右腕を扱う（左腕は鏡像）。
上腕の姿勢 R_h（胸郭に対する回転）。基準姿勢: 腕を下ろし（上腕軸 −z）、肘の屈曲軸 −y（前方に曲がる）。

肩の機構モデル:
  "gimbal" : 2 軸ジンバル（Sim07 の球面座標の肩）。R_h = Rz(ψ)·Ry(−el)。ひねりを独立に選べない
  "ball"   : 3 自由度の球関節。R_h を回転ベクトルで表し、スイング（腕の向き）とツイスト（ひねり）に分解して可動域を課す
  "ball+sg": 球関節＋上肢帯（鎖骨・肩甲骨）の前方突出（protraction）。肩関節の中心そのものが前方・内側へ動く
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as Rt

DOWN = np.array([0.0, 0.0, -1.0])
ELBOW_AXIS = np.array([0.0, -1.0, 0.0])


def swing_twist(R: Rt, axis=DOWN):
    """R = R_swing · R_twist（R_twist は局所軸 axis まわり）。ツイスト角 [rad] を返す."""
    x, y, z, w = R.as_quat()
    v = np.array([x, y, z])
    return 2 * np.arctan2(v @ axis, w)


def no_spin_transport(path):
    """腕の向きの列に沿って、ひねりを加えずに（角速度が常に腕に垂直）上腕を動かしたときの回転."""
    R = Rt.identity()
    for a, b in zip(path[:-1], path[1:]):
        ax = np.cross(a, b); s = np.linalg.norm(ax)
        if s > 1e-12:
            R = Rt.from_rotvec(ax / s * np.arctan2(s, a @ b)) * R
    return R


def slerp_path(a, b, n=200):
    t = np.linspace(0, 1, n)
    ang = np.arccos(np.clip(a @ b, -1, 1))
    return np.array([(np.sin((1 - x) * ang) * a + np.sin(x * ang) * b) / np.sin(ang) for x in t])


@dataclass
class UpperBody:
    torso_half: tuple = (0.11, 0.14)               # 胸郭（楕円柱）の半奥行・半幅 [m]
    torso_ztop: float = 0.20
    sc_joint: tuple = (0.06, -0.02, 0.17)          # 胸鎖関節（鎖骨の付け根）
    gh_neutral: tuple = (0.0, -0.19, 0.15)         # 肩関節中心（基準位置）
    l_upper: float = 0.30; r_upper: float = 0.05
    l_fore: float = 0.27; r_fore: float = 0.04     # 前腕＋手の中心まで
    # 人間の可動域の目安（右腕）
    hadd_max: float = np.radians(45)               # 水平内転: 体の正中線を越えて腕を内側へ向けられる角度
    twist_int: float = np.radians(70)              # 内旋
    twist_ext: float = np.radians(90)              # 外旋
    elbow_max: float = np.radians(145)
    prot_max: float = np.radians(25)               # 上肢帯の前方突出（鎖骨の回転）

    def shoulder_center(self, prot=0.0):
        sc = np.array(self.sc_joint); cl = np.array(self.gh_neutral) - sc
        return sc + Rt.from_rotvec([0, 0, prot]).apply(cl)

    def chain(self, R_h: Rt, elbow: float, prot: float = 0.0):
        S = self.shoulder_center(prot)
        u = R_h.apply(DOWN)
        E = S + self.l_upper * u
        f = (R_h * Rt.from_rotvec(ELBOW_AXIS * elbow)).apply(DOWN)
        H = E + self.l_fore * f
        return S, E, H, u, f

    def collision(self, S, E, H, n=12):
        """腕の各点の、胸郭（楕円柱）からの離れ具合（>0 で接触なし）"""
        out = []
        a0, b0 = self.torso_half
        for P0, P1, r, skip in ((S, E, self.r_upper, 0.30), (E, H, self.r_fore, 0.0)):
            for t in np.linspace(skip, 1, n):
                p = P0 + t * (P1 - P0)
                if p[2] > self.torso_ztop + r:
                    out.append(1.0); continue
                out.append((p[0] / (a0 + r))**2 + (p[1] / (b0 + r))**2 - 1)
        return np.array(out)
