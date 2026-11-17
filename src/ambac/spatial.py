"""3 次元モデル: 胴体（剛体）＋ 球面座標の肩を持つ腕.

腕の向き（胴体座標系、肩が原点）:
    u(ψ, α) = (cos α cos ψ, cos α sin ψ, sin α)
    ψ: 方位角（x 前方 → y 左）, α: 仰角（水平面から上向きが正）
2 自由度の肩（旋回＋俯仰）は球面座標そのもの。α = ±90°（真上・真下）で ψ が定まらない
（座標の特異点 = 機械のジンバルロック）。

腕（一様な棒）と荷物は点質量の列で表す（棒を n 分割した各区間の中点に質量を置く）。
これにより、回転慣性の項を別に扱わずに角運動量・反力を計算できる。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


def sph(psi, alpha):
    psi, alpha = np.asarray(psi, float), np.asarray(alpha, float)
    return np.stack([np.cos(alpha) * np.cos(psi), np.cos(alpha) * np.sin(psi), np.sin(alpha)], -1)


def sph_jac(psi, alpha):
    """∂u/∂ψ, ∂u/∂α  (..., 3, 2)"""
    d_psi = np.stack([-np.cos(alpha) * np.sin(psi), np.cos(alpha) * np.cos(psi), np.zeros_like(psi)], -1)
    d_alp = np.stack([-np.sin(alpha) * np.cos(psi), -np.sin(alpha) * np.sin(psi), np.cos(alpha)], -1)
    return np.stack([d_psi, d_alp], -1)


def to_sph(u):
    u = np.asarray(u, float)
    return np.arctan2(u[..., 1], u[..., 0]), np.arcsin(np.clip(u[..., 2], -1, 1))


@dataclass
class Arm3D:
    shoulder: tuple            # 胴体座標系での肩の位置 [m]
    mass: float                # 腕（棒）の質量 [kg]
    length: float              # [m]
    payload: float = 0.0       # 先端の荷物（質点） [kg]
    n_pts: int = 40

    def points(self, q):
        """q = (ψ, α) → 点の位置 (P,3), 質量 (P,), 腕方向ヤコビアン (P,3,2)"""
        d = (np.arange(self.n_pts) + 0.5) / self.n_pts * self.length
        m = np.full(self.n_pts, self.mass / self.n_pts)
        if self.payload > 0:
            d = np.r_[d, self.length]; m = np.r_[m, self.payload]
        u = sph(q[0], q[1]); J = sph_jac(q[0], q[1])
        return np.array(self.shoulder) + d[:, None] * u, m, d[:, None, None] * J


@dataclass
class FloatingBody3D:
    """宇宙空間の自由浮遊系: 胴体（直方体）＋腕 1 本."""
    base_mass: float
    base_dims: tuple           # 直方体の寸法 (x, y, z) [m]
    arm: Arm3D

    @property
    def I_base(self):
        a, b, c = self.base_dims
        return self.base_mass / 12 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])

    def coupling(self, q):
        """胴体座標系の角運動量 L = A(q) ω + B(q) q'（系重心まわり, ω: 胴体角速度・胴体座標）"""
        p, m, J = self.arm.points(q)
        P = np.vstack([np.zeros(3), p]); M = np.r_[self.base_mass, m]
        Jp = np.concatenate([np.zeros((1, 3, 2)), J])
        c = (M[:, None] * P).sum(0) / M.sum()
        Jc = (M[:, None, None] * Jp).sum(0) / M.sum()
        r = P - c; Jr = Jp - Jc
        A = self.I_base + np.einsum("i,ij->", M, r * r) * np.eye(3) - np.einsum("i,ij,ik->jk", M, r, r)
        B = np.einsum("i,ijk->jk", M, np.cross(r[:, :, None], Jr, axisa=1, axisb=1, axisc=1))
        return A, B

    def omega(self, q, qd, L_body=None):
        A, B = self.coupling(q)
        rhs = -B @ qd if L_body is None else L_body - B @ qd
        return np.linalg.solve(A, rhs)

    def integrate(self, q_fun, qd_fun, T, n=4000, R0=None):
        """L = 0 の下で関節軌道を与え、胴体姿勢（回転）を積分する（中点則）."""
        R = Rotation.identity() if R0 is None else R0
        dt = T / n; out = [R]
        for k in range(n):
            tm = (k + 0.5) * dt
            w = self.omega(q_fun(tm), qd_fun(tm))
            R = R * Rotation.from_rotvec(w * dt)       # 胴体座標の角速度 → 右から掛ける
            out.append(R)
        return out


@dataclass
class TankArm3D:
    """地上の戦車型台車＋球面座標の肩を持つ腕（台車は静止として反力を求める）."""
    base_mass: float = 80.0
    base_com: tuple = (0.0, 0.0, 0.35)
    track_half: tuple = (0.4, 0.3)       # 接地面の半分の長さ (前後 a, 左右 b) [m]
    arm: Arm3D = None
    g: float = 9.80665

    def __post_init__(self):
        if self.arm is None:
            self.arm = Arm3D(shoulder=(0.0, 0.0, 0.75), mass=7.0, length=1.0, payload=10.0)

    def zmp_path(self, q_path, T):
        """q(τ) の等間隔サンプル (K,2) を時間 T で実行したときの ZMP (K,2), 垂直反力 N, 水平反力 (K,2)"""
        P = np.array([self.arm.points(q)[0] for q in q_path])          # (K,P,3)
        _, m, _ = self.arm.points(q_path[0])
        K = len(q_path); h = 1 / (K - 1)
        acc = np.gradient(np.gradient(P, h, axis=0), h, axis=0) / T**2
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        ax, ay, az = acc[..., 0], acc[..., 1], acc[..., 2]
        mb = self.base_mass; xb, yb, zb = self.base_com
        N = mb * self.g + (m * (az + self.g)).sum(-1)
        px = (mb * self.g * xb + (m * (x * (az + self.g) - z * ax)).sum(-1)) / N
        py = (mb * self.g * yb + (m * (y * (az + self.g) - z * ay)).sum(-1)) / N
        F = np.stack([(m * ax).sum(-1), (m * ay).sum(-1)], -1)
        return np.stack([px, py], -1), N, F

    def margin(self, zmp):
        """接地長方形の縁までの最短距離（負なら外 = 転倒）"""
        a, b = self.track_half
        return np.minimum(a - np.abs(zmp[..., 0]), b - np.abs(zmp[..., 1]))
