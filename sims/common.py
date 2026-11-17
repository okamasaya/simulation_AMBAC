"""シミュレーション共通設定・ヘルパ."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data"
FIG = ROOT / "figures"
DATA.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

from ambac.planar import Limb, PlanarRobot  # noqa: E402


def human_like(arm_mass=4.0, leg_mass=None) -> PlanarRobot:
    """宇宙服なしの成人を矢状面に投影した粗いモデル（胴体＋頭＋脚を胴体に集約）.

    胴体: 60 kg, 幅 0.35 m x 高さ 1.0 m の一様長方形
    腕  : 4 kg, 0.7 m, 肩 (0, +0.35) に取付
    脚  : leg_mass を与えると独立リンク化（股関節 (0, -0.15), 長さ 0.9 m）
    """
    m0 = 60.0
    I0 = m0 * (0.35**2 + 1.0**2) / 12.0
    limbs = [Limb(attach=(0.0, 0.35), mass=arm_mass, length=0.7)]
    if leg_mass is not None:
        limbs.append(Limb(attach=(0.0, -0.15), mass=leg_mass, length=0.9))
    return PlanarRobot(base_mass=m0, base_inertia=I0, limbs=limbs)


def integrate_attitude(robot: PlanarRobot, q_fun, qd_fun, T, theta0=0.0, n=2001):
    """L=0 拘束 theta_dot = -Γ(q)·q_dot を積分する."""
    def rhs(t, y):
        return [-float(robot.connection(q_fun(t)) @ qd_fun(t))]

    t = np.linspace(0, T, n)
    sol = solve_ivp(rhs, (0, T), [theta0], t_eval=t, rtol=1e-10, atol=1e-12,
                    max_step=T / 400)
    return sol.t, sol.y[0]


def smooth_step(s):
    """0→1 の滑らかなステップ（速度・加速度が端点で0）と、その微分."""
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5, 30 * s**2 - 60 * s**3 + 30 * s**4
