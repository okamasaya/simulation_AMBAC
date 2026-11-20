"""スキッドステア（左右の履帯の速度差による旋回）の基準モデル（Python、準静的）.

履帯の接地面（長さ L、幅 b、中心の間隔 B）に一様な圧力を仮定し、各小片の滑りの速度と逆向きにクーロン摩擦（μ）が働くとする。
履帯の送りの速さ u_L, u_R（車体に対するベルトの速さ。前進が正）を与え、
車体の速度 (vx, vy) と旋回の角速度 ω を、力と y 軸まわりのモーメントのつり合い（慣性を無視）から求める。
戻り値の効率 = 実際の ω ÷ 滑りがない場合の ω（(u_R − u_L) / B）。
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import fsolve


def contact_points(L, B, b, nx=40, ny=4):
    xs = (np.arange(nx) + 0.5) / nx * L - L / 2
    ys = (np.arange(ny) + 0.5) / ny * b - b / 2
    pts = []
    for side, y0 in (("L", B / 2), ("R", -B / 2)):
        for x in xs:
            for y in ys:
                pts.append((x, y0 + y, side))
    return pts


def forces(state, uL, uR, pts, mu, W, eps=1e-4):
    vx, vy, w = state
    n = len(pts); p = W / n
    F = np.zeros(3)
    for x, y, side in pts:
        u = uL if side == "L" else uR
        sx = vx - w * y - u          # 接地点の地面に対する滑り（車体の速度 − ベルトの送り）
        sy = vy + w * x
        sp = np.hypot(sx, sy) + eps
        fx, fy = -mu * p * sx / sp, -mu * p * sy / sp
        F += (fx, fy, x * fy - y * fx)
    return F


def steady_turn(uL, uR, L=0.8, B=0.45, b=0.15, mu=0.6, W=1500.0):
    """定常旋回の (vx, vy, ω) と、効率（ω ÷ 滑りなしの ω）"""
    pts = contact_points(L, B, b)
    w0 = (uR - uL) / B
    sol = fsolve(lambda s: forces(s, uL, uR, pts, mu, W) / W, [(uL + uR) / 2, 0.0, 0.5 * w0], xtol=1e-10)
    eff = sol[2] / w0 if abs(w0) > 1e-12 else np.nan
    return sol, eff


def pivot_resistance(L=0.8, B=0.45, mu=0.6, W=1500.0):
    """その場旋回で、片側の履帯が出す必要のある前後の力（一様な圧力、滑り始めの近似: μ W L / (4 B)）"""
    return mu * W * L / (4 * B)


def steady_turn_dist(uL, uR, xs, share, B=0.45, b=0.15, mu=0.6, W=1500.0, ny=4):
    """前後方向の荷重分布（位置 xs と割合 share、左右で同じ）を与えた場合の定常旋回の (vx, vy, ω)"""
    share = np.asarray(share, float) / np.sum(share)
    ys = (np.arange(ny) + 0.5) / ny * b - b / 2
    pts = [(x, s * B / 2 + y, "L" if s > 0 else "R", w * W / 2 / ny) for x, w in zip(xs, share) for s in (1, -1) for y in ys]

    def F(st):
        vx, vy, w = st; out = np.zeros(3)
        for x, y, side, p in pts:
            u = uL if side == "L" else uR
            sx = vx - w * y - u; sy = vy + w * x; sp = np.hypot(sx, sy) + 1e-4
            fx, fy = -mu * p * sx / sp, -mu * p * sy / sp
            out += (fx, fy, x * fy - y * fx)
        return out / W
    w0 = (uR - uL) / B
    return fsolve(F, [(uL + uR) / 2, 0.0, 0.5 * w0], xtol=1e-10)
