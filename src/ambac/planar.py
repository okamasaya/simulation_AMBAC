"""平面（2次元）自由浮遊マルチボディモデル.

胴体（base）に1本以上の剛体リンク（腕・脚）が回転関節で取り付けられた系を扱う。
外力・外トルクが無い宇宙空間を想定し、全運動量・全角運動量は保存される。

記号:
    theta : 胴体の姿勢角 [rad]（慣性系基準）
    q     : 関節角ベクトル [rad]（胴体基準の相対角）
    L     : 系重心まわりの全角運動量

全角運動量は回転対称性により theta に依存せず
    L = A(q) * theta_dot + B(q) @ q_dot
と書ける（A: 系全体の慣性モーメント, B: 関節運動と姿勢の結合項）。
L = 0（静止状態から開始）のとき
    theta_dot = -(B(q) / A(q)) @ q_dot          ... (非ホロノミック拘束)
となり、これが AMBAC の基本式である。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _rot(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


_J = np.array([[0.0, -1.0], [1.0, 0.0]])  # 90度回転 (z x r)


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


@dataclass
class Limb:
    """胴体に回転関節で取り付けられた剛体リンク.

    既定では一様な棒（重心は関節から length/2、慣性 m l^2/12）。
    com_dist / inertia_c を与えると任意の剛体を表せる。
    例: リアクションホイール = 重心が回転軸上 (com_dist=0) の円盤
        → 回転しても質量配置が変わらない「連続回転できる関節」。
    """

    attach: tuple[float, float]       # 胴体座標系での関節位置 [m]
    mass: float                       # [kg]
    length: float                     # [m]
    com_dist: float | None = None     # 関節から重心までの距離 [m]
    inertia_c: float | None = None    # 重心まわり慣性モーメント [kg m^2]

    @property
    def r_com(self) -> float:
        return self.length / 2.0 if self.com_dist is None else self.com_dist

    @property
    def inertia(self) -> float:
        if self.inertia_c is not None:
            return self.inertia_c
        return self.mass * self.length**2 / 12.0


def reaction_wheel(mass: float, radius: float, attach=(0.0, 0.0)) -> Limb:
    """一様円盤のリアクションホイール."""
    return Limb(attach=attach, mass=mass, length=0.0, com_dist=0.0,
                inertia_c=0.5 * mass * radius**2)


@dataclass
class PlanarRobot:
    base_mass: float
    base_inertia: float
    limbs: list[Limb] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.limbs)

    @property
    def total_mass(self) -> float:
        return self.base_mass + sum(l.mass for l in self.limbs)

    # ---- 運動学 -----------------------------------------------------------
    def positions(self, theta: float, q: np.ndarray) -> np.ndarray:
        """各剛体（0: 胴体, 1..n: リンク）の重心位置 (胴体重心原点)."""
        p = [np.zeros(2)]
        for k, lb in enumerate(self.limbs):
            a = np.asarray(lb.attach, float)
            d = np.array([lb.r_com, 0.0])
            p.append(_rot(theta) @ a + _rot(theta + q[k]) @ d)
        return np.array(p)

    def velocities(self, theta, q, theta_dot, q_dot) -> np.ndarray:
        v = [np.zeros(2)]
        for k, lb in enumerate(self.limbs):
            a = np.asarray(lb.attach, float)
            d = np.array([lb.r_com, 0.0])
            v.append(theta_dot * (_J @ _rot(theta) @ a)
                     + (theta_dot + q_dot[k]) * (_J @ _rot(theta + q[k]) @ d))
        return np.array(v)

    def com(self, theta, q) -> np.ndarray:
        m = np.array([self.base_mass] + [l.mass for l in self.limbs])
        return (m[:, None] * self.positions(theta, q)).sum(0) / m.sum()

    # ---- 角運動量 ---------------------------------------------------------
    def angular_momentum(self, theta, q, theta_dot, q_dot) -> float:
        """系重心まわりの全角運動量."""
        q = np.asarray(q, float)
        q_dot = np.asarray(q_dot, float)
        m = np.array([self.base_mass] + [l.mass for l in self.limbs])
        I = np.array([self.base_inertia] + [l.inertia for l in self.limbs])
        w = np.concatenate([[theta_dot], theta_dot + q_dot])
        p = self.positions(theta, q)
        v = self.velocities(theta, q, theta_dot, q_dot)
        c = (m[:, None] * p).sum(0) / m.sum()
        vc = (m[:, None] * v).sum(0) / m.sum()
        L = float((I * w).sum())
        for i in range(len(m)):
            L += m[i] * _cross2(p[i] - c, v[i] - vc)
        return L

    def coupling(self, q) -> tuple[float, np.ndarray]:
        """A(q), B(q) を返す（L が速度に線形であることを利用）."""
        q = np.asarray(q, float)
        z = np.zeros(self.n)
        A = self.angular_momentum(0.0, q, 1.0, z)
        B = np.array([self.angular_momentum(0.0, q, 0.0, np.eye(self.n)[k])
                      for k in range(self.n)])
        return A, B

    def connection(self, q) -> np.ndarray:
        """機械的接続 (mechanical connection)  Γ(q) = B(q)/A(q).

        theta_dot = -Γ(q) @ q_dot  (L=0 のとき)
        """
        A, B = self.coupling(q)
        return B / A

    def curvature(self, q, h: float = 1e-5) -> float:
        """2関節系の曲率 F = ∂Γ2/∂q1 - ∂Γ1/∂q2.

        閉ループ C で囲まれた領域 S について、1周あたりの正味姿勢変化は
            Δθ = -∮_C Γ·dq = -∬_S F dq1 dq2     (ストークスの定理)
        """
        assert self.n == 2
        q = np.asarray(q, float)
        e1, e2 = np.array([h, 0.0]), np.array([0.0, h])
        dG2_dq1 = (self.connection(q + e1)[1] - self.connection(q - e1)[1]) / (2 * h)
        dG1_dq2 = (self.connection(q + e2)[0] - self.connection(q - e2)[0]) / (2 * h)
        return dG2_dq1 - dG1_dq2
