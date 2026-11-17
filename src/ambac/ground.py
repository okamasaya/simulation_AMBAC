"""地上を動く機体（戦車型台車 + 腕）の矢状面モデル.

台車は地面に置かれて静止しているとし（滑り・転倒の「開始条件」を調べる）、
腕の運動に対して地面が台車に返すべき反力を、系全体のニュートン・オイラー式から求める。

座標: x 前方, z 上方。角度は x 軸から z 軸へ向かう向き（反時計回り）が正。

  垂直反力     N  = Σ m_i (z̈_i + g)
  水平反力     Fx = Σ m_i ẍ_i                （摩擦で賄う必要がある）
  ZMP          p  = [Σ m_i (x_i (z̈_i + g) − z_i ẍ_i) + Σ I_i φ̈_i] / N
  転倒しない条件 : −a ≤ p ≤ a    （a: 履帯接地長の半分）
  滑らない条件   : |Fx| ≤ μ N

腕の宇宙での AMBAC と同じく、反力のうち動的な部分は「角運動量の時間変化」そのものである。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TankArm:
    base_mass: float = 80.0           # 台車＋胴体 [kg]
    base_com: tuple = (0.0, 0.35)     # 台車重心 [m]
    half_track: float = 0.4           # 履帯接地長の半分 a [m]
    shoulder: tuple = (0.0, 0.75)     # 肩の位置 [m]
    l1: float = 0.5; m1: float = 4.0  # 上腕
    l2: float = 0.5; m2: float = 3.0  # 前腕
    payload: float = 10.0             # 先端の荷物（質点） [kg]
    g: float = 9.80665

    def scaled(self, s: float, g: float | None = None) -> "TankArm":
        """長さ s 倍・同密度（質量 s³ 倍）に相似拡大."""
        m = s**3
        return TankArm(self.base_mass * m, (self.base_com[0] * s, self.base_com[1] * s),
                       self.half_track * s, (self.shoulder[0] * s, self.shoulder[1] * s),
                       self.l1 * s, self.m1 * m, self.l2 * s, self.m2 * m, self.payload * m,
                       self.g if g is None else g)

    @property
    def masses(self):
        return np.array([self.base_mass, self.m1, self.m2, self.payload])

    @property
    def inertias(self):
        return np.array([0.0, self.m1 * self.l1**2 / 12, self.m2 * self.l2**2 / 12, 0.0])

    def bodies(self, q):
        """q: (..., 2) 関節角（q1: 上腕の絶対角, q2: 肘の相対角）
        → 各剛体の重心位置 (..., 4, 2) と絶対角 (..., 4)。台車の絶対角は 0."""
        q = np.asarray(q, float)
        a1 = q[..., 0]; a2 = q[..., 0] + q[..., 1]
        S = np.array(self.shoulder)
        e1 = np.stack([np.cos(a1), np.sin(a1)], -1); e2 = np.stack([np.cos(a2), np.sin(a2)], -1)
        elbow = S + self.l1 * e1
        tip = elbow + self.l2 * e2
        base = np.broadcast_to(np.array(self.base_com, float), elbow.shape)
        pos = np.stack([base, S + 0.5 * self.l1 * e1, elbow + 0.5 * self.l2 * e2, tip], -2)
        ang = np.stack([np.zeros_like(a1), a1, a2, a2], -1)
        return pos, ang

    def tip(self, q):
        return self.bodies(q)[0][..., 3, :]

    def reactions(self, q_path, T):
        """関節経路 q(τ), τ∈[0,1] の等間隔サンプル (K,2) を時間 T で実行したときの
        ZMP p, 垂直反力 N, 水平反力 Fx を返す（各 (K,)）。"""
        K = len(q_path)
        pos, ang = self.bodies(q_path)
        h = 1.0 / (K - 1)
        acc = np.gradient(np.gradient(pos, h, axis=0), h, axis=0) / T**2
        alp = np.gradient(np.gradient(ang, h, axis=0), h, axis=0) / T**2
        m = self.masses; I = self.inertias
        x, z = pos[..., 0], pos[..., 1]
        ax, az = acc[..., 0], acc[..., 1]
        N = (m * (az + self.g)).sum(-1)
        Fx = (m * ax).sum(-1)
        p = ((m * (x * (az + self.g) - z * ax)).sum(-1) + (I * alp).sum(-1)) / N
        return p, N, Fx


def min_jerk(t):
    """0→1 の最小躍度プロファイル."""
    return 10 * t**3 - 15 * t**4 + 6 * t**5
