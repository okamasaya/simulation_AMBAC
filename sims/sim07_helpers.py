"""Sim07 の円錐運動（球面座標の肩）を他のシミュレーションから使うための補助関数."""
import numpy as np
from scipy.spatial.transform import Rotation

from ambac.spatial import sph_jac, to_sph


def cone_holonomy(body, axis, beta, T=2.0, n=3000, direction=+1):
    """肢の先端方向が axis まわりに半頂角 beta の円錐を 1 周描いたときの、胴体の正味の回転ベクトル（胴体座標）"""
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    t_ = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(t_, a); e1 /= np.linalg.norm(e1); e2 = np.cross(a, e1)
    w = direction * 2 * np.pi / T

    def u(t):
        return np.cos(beta) * a + np.sin(beta) * (np.cos(w * t) * e1 + np.sin(w * t) * e2)

    def q(t):
        return np.array(to_sph(u(t)))

    def qd(t):
        ud = np.sin(beta) * w * (-np.sin(w * t) * e1 + np.cos(w * t) * e2)
        return np.linalg.lstsq(sph_jac(*q(t)), ud, rcond=None)[0]

    return body.integrate(q, qd, T, n=n)[-1].as_rotvec()
