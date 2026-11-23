"""Field04: 「答えを隠した仮想の実機」によるキャリブレーション（シミュレーション場の段階 4）.

1. 宇宙の人型（space_arm_leg）の腕の質量・脚の質量・胴体の慣性の倍率を乱数でずらした「仮想の実機」を MuJoCo で作る（正解は推定側に隠す）。
   総質量は既知（76 kg）とし、胴体の質量 = 76 − 腕 − 脚。
2. 曲率マップの異なる場所に置いた 6 種類の円ループ（詳細版 D14 と同じ）を 1 周ずつ回す。
3. センサー: 胴体の角速度（ジャイロ、白色雑音 0.2°/s ＋ 一定のバイアス最大 ±0.2°/s）、関節角度（エンコーダ、雑音 0.05°）を 100 Hz で記録。
4. 推定: (a) 1 周ごとの正味の回転（ジャイロを積分）だけを使う
         (b) 周回中の角速度の波形を使う（θ' = −Γ(q; p)·q' の当てはめ、ジャイロのバイアスも同時に推定）
         (b') 微分を使わない形: 短い区間（0.1 s）ごとに、ジャイロの積分（回転量）と −∫Γ(q; p)·dq（関節角の変化から計算）を比べる
         (c) (b) に関節トルク（雑音 0.2 N m）を加える（逆動力学 τ = τ(q, q', q''; p) の当てはめ）
   検証の結果、(b)・(c) の偏りは「関節角を微分して速度・加速度を求める前処理」によるものだった
   （MuJoCo の正確な速度・加速度を使うと誤差 0.05% 以下）。(b') はこの前処理を避ける。
   姿勢の波形は質量・慣性の「比」しか決めないため、(b) では全体を同じ割合で増減させる方向が推定しにくい。
   (c) ではトルク（力の情報）で絶対値を決める。
5. 乱数を変えて 5 回繰り返し、推定誤差を正解と比べる。
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import copy
import json
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from common import DATA, FIG, smooth_step
from ambac.planar import Limb, PlanarRobot
from ambac.field.assembly import build, load
from ambac.field import space as fs

D = np.degrees
RAD = np.radians
OUT = FIG / "field"
M_TOT = 76.0
NOMINAL = np.array([4.0, 12.0, 1.0])            # 設計値: 腕 4 kg、脚 12 kg、胴体の慣性の倍率 1
LOOPS = [(-20, -40, 70), (-60, -60, 40), (40, -60, 40), (0, -90, 25), (-90, -20, 30), (60, -20, 30)]
LOOPS = [tuple(RAD(v) for v in L) for L in LOOPS]
T_LOOP, FS = 4.0, 100.0
SIG_GYRO, BIAS_MAX, SIG_ENC, SIG_TAU = RAD(0.2), RAD(0.2), RAD(0.05), 0.2


def robot_dict(p):
    ma, ml, kI = p
    r = copy.deepcopy(load("robots", "space_arm_leg"))
    r["base"]["mass"] = M_TOT - ma - ml; r["base"]["inertia_scale"] = kI
    r["arms"][0]["links"][0]["mass"] = ma; r["arms"][1]["links"][0]["mass"] = ml
    return r


def planar(p):
    ma, ml, kI = p; mb = M_TOT - ma - ml
    return PlanarRobot(mb, kI * mb * (0.35**2 + 1) / 12, [Limb((0, 0.35), ma, 0.7), Limb((0, -0.15), ml, 0.9)])


def experiment(p_true, rng):
    """仮想の実機で 6 ループを実行し、ノイズ付きのセンサーデータを返す"""
    xml, _ = build(robot_dict(p_true), load("stages", "space"))
    bias = rng.uniform(-BIAS_MAX, BIAS_MAX)
    data = []
    for c1, c2, R in LOOPS:
        def ref(t, c1=c1, c2=c2, R=R):
            s, _ = smooth_step(min(t / T_LOOP, 1.0)); ph = 2 * np.pi * s
            return [c1 + R * np.cos(ph), c2 + R * np.sin(ph)]
        o = fs.run(xml, ref(0.0), ref, T_LOOP + 0.3, record_every=int(1 / (FS * 5e-4)))
        gyro = o["gyro"] + bias + rng.normal(0, SIG_GYRO, len(o["t"]))
        enc = o["q"] + rng.normal(0, SIG_ENC, o["q"].shape)
        tau = o["tau"] + rng.normal(0, SIG_TAU, o["tau"].shape)
        data.append(dict(t=o["t"], gyro=gyro, enc=enc, tau=tau, theta_true=o["theta"]))
    return data, bias


def predicted_rate(p, enc, dt):
    r = planar(p)
    qd = np.gradient(enc, dt, axis=0)
    G = np.array([r.connection(q) for q in enc])
    return -np.sum(G * qd, 1)


def fit_waveform(data):
    dt = 1 / FS
    def resid(x):
        p, bias = x[:3], x[3]
        return np.concatenate([d["gyro"] - bias - predicted_rate(p, d["enc"], dt) for d in data])
    sol = least_squares(resid, np.r_[NOMINAL, 0.0], x_scale=[1, 1, 0.1, 0.01],
                        bounds=([0.5, 1.0, 0.3, -0.05], [15, 40, 3.0, 0.05]))
    return sol.x


def smooth_derivs(enc, dt, win=15, order=3):
    """エンコーダの角度を平滑化し、角度・角速度・角加速度を求める（Savitzky–Golay）"""
    return [savgol_filter(enc, win, order, deriv=k, delta=dt, axis=0) for k in (0, 1, 2)]


def fit_waveform_torque(data, sub=4):
    dt = 1 / FS
    pre = [smooth_derivs(d["enc"], dt) for d in data]

    def resid(x):
        p, bias = x[:3], x[3]
        r = planar(p); out = []
        for d, (q, qd, qdd) in zip(data, pre):
            rate = -np.sum(np.array([r.connection(qq) for qq in q]) * qd, 1)
            out.append((d["gyro"] - bias - rate) / SIG_GYRO)
            tau_m = np.array([r.inverse_dynamics(q[i], qd[i], qdd[i])[2] for i in range(5, len(q) - 5, sub)])
            out.append(((d["tau"][5:len(q) - 5:sub] - tau_m) / SIG_TAU).ravel())
        return np.concatenate(out)
    sol = least_squares(resid, np.r_[NOMINAL, 0.0], x_scale=[1, 1, 0.1, 0.01],
                        bounds=([0.5, 1.0, 0.3, -0.05], [15, 40, 3.0, 0.05]))
    return sol.x


def fit_pathintegral(data, win=10):
    """微分を使わない推定: 区間ごとの回転量（ジャイロの積分）と、関節の経路に沿った −∫Γ·dq を比べる"""
    dt = 1 / FS

    def resid(x):
        p, bias = x[:3], x[3]
        r = planar(p); out = []
        for d in data:
            q = d["enc"]; n = (len(q) - 1) // win
            G = np.array([r.connection(qq) for qq in (q[:-1] + q[1:]) / 2])
            dth_model = -np.sum(G * np.diff(q, axis=0), 1)              # 各サンプル間の回転量
            dth_gyro = (d["gyro"][:-1] + d["gyro"][1:]) / 2 * dt - bias * dt
            dm = dth_model[:n * win].reshape(n, win).sum(1); dg = dth_gyro[:n * win].reshape(n, win).sum(1)
            out.append((dg - dm) / (SIG_GYRO * dt * np.sqrt(win)))
        return np.concatenate(out)
    sol = least_squares(resid, np.r_[NOMINAL, 0.0], x_scale=[1, 1, 0.1, 0.01],
                        bounds=([0.5, 1.0, 0.3, -0.05], [15, 40, 3.0, 0.05]))
    return sol.x


def fit_netrotation(data):
    """1 周ごとの正味の回転（ジャイロの積分）だけで推定。バイアスは静止区間がないので推定しない"""
    dt = 1 / FS
    meas = np.array([np.sum(d["gyro"]) * dt for d in data])
    def model(p):
        r = planar(p); out = []
        for c1, c2, R in LOOPS:
            ph = np.linspace(0, 2 * np.pi, 361)
            q = np.c_[c1 + R * np.cos(ph), c2 + R * np.sin(ph)]
            dq = np.diff(q, axis=0); mid = (q[1:] + q[:-1]) / 2
            out.append(-np.sum(np.sum(np.array([r.connection(m) for m in mid]) * dq, 1)))
        return np.array(out)
    sol = least_squares(lambda p: model(p) - meas, NOMINAL, x_scale=[1, 1, 0.1], bounds=([0.5, 1.0, 0.3], [15, 40, 3.0]))
    return sol.x


def main():
    rows = []
    rng = np.random.default_rng(2026)
    print("=== 仮想の実機によるキャリブレーション（5 回）===")
    print("  試行  正解（腕, 脚, 慣性倍率）      | (a) 正味の回転だけ              | (b) 波形（微分）                 | (b') 波形（微分なし）            | (c) 波形（微分）＋トルク")
    t0 = time.time()
    for trial in range(5):
        p_true = NOMINAL * np.array([rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15), rng.uniform(0.9, 1.1)])
        data, bias = experiment(p_true, rng)
        pa = fit_netrotation(data); pb = fit_waveform(data); pc = fit_waveform_torque(data); pp = fit_pathintegral(data)
        ea = (pa - p_true) / p_true * 100; eb = (pb[:3] - p_true) / p_true * 100; ec = (pc[:3] - p_true) / p_true * 100
        ep = (pp[:3] - p_true) / p_true * 100
        rows.append(dict(trial=trial, true=p_true.tolist(), bias_true=float(D(bias)), est_a=pa.tolist(), est_b=pb[:3].tolist(),
                         bias_est=float(D(pb[3])), est_c=pc[:3].tolist(), est_p=pp[:3].tolist(), bias_est_p=float(D(pp[3])),
                         err_a=ea.tolist(), err_b=eb.tolist(), err_c=ec.tolist(), err_p=ep.tolist()))
        print(f"  {trial}  ({p_true[0]:.2f}, {p_true[1]:.2f}, {p_true[2]:.3f})  | "
              f"誤差 {ea[0]:+6.1f}%, {ea[1]:+6.1f}%, {ea[2]:+5.1f}%  | 誤差 {eb[0]:+5.2f}%, {eb[1]:+5.2f}%, {eb[2]:+5.2f}%"
              f"  | 誤差 {ep[0]:+5.2f}%, {ep[1]:+5.2f}%, {ep[2]:+5.2f}%  | 誤差 {ec[0]:+5.2f}%, {ec[1]:+5.2f}%, {ec[2]:+5.2f}%", flush=True)
    ea = np.array([r["err_a"] for r in rows]); eb = np.array([r["err_b"] for r in rows]); ec = np.array([r["err_c"] for r in rows])
    ep = np.array([r["err_p"] for r in rows])
    rms_a = np.sqrt((ea**2).mean(0)); rms_b = np.sqrt((eb**2).mean(0)); rms_c = np.sqrt((ec**2).mean(0)); rms_p = np.sqrt((ep**2).mean(0))
    print(f"  誤差の二乗平均: (a) 腕 {rms_a[0]:.1f}%, 脚 {rms_a[1]:.1f}%, 慣性 {rms_a[2]:.1f}% / "
          f"(b) 腕 {rms_b[0]:.2f}%, 脚 {rms_b[1]:.2f}%, 慣性 {rms_b[2]:.2f}% / (b') 腕 {rms_p[0]:.2f}%, 脚 {rms_p[1]:.2f}%, 慣性 {rms_p[2]:.2f}% / "
          f"(c) 腕 {rms_c[0]:.2f}%, 脚 {rms_c[1]:.2f}%, 慣性 {rms_c[2]:.2f}%"
          f"（{time.time()-t0:.0f} s）")
    json.dump(dict(trials=rows, rms_a=rms_a.tolist(), rms_b=rms_b.tolist(), rms_p=rms_p.tolist(), rms_c=rms_c.tolist(),
                   settings=dict(gyro_noise_deg_s=0.2, bias_max_deg_s=0.2, encoder_noise_deg=0.05, torque_noise_Nm=SIG_TAU, fs_hz=FS, loops=len(LOOPS))),
              open(DATA / "field04_calibration.json", "w"), indent=2, ensure_ascii=False)

    # ---- 図 ------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    names = ["arm mass", "leg mass", "torso inertia"]
    x = np.arange(3)
    for k in range(len(rows)):
        ax[0].plot(x - 0.12, ea[k], "o", color="C0", alpha=.6, label="(a) net rotation only" if k == 0 else None)
        ax[0].plot(x, eb[k], "s", color="C1", alpha=.8, label="(b) waveform (differentiated encoder)" if k == 0 else None)
        ax[0].plot(x + 0.12, ec[k], "D", color="C4", alpha=.8, label="(c) waveform (differentiated) + torque" if k == 0 else None)
        ax[0].plot(x + 0.24, ep[k], "*", color="C3", ms=10, label="(b') waveform, no differentiation" if k == 0 else None)
    ax[0].axhline(0, color="k", lw=.6); ax[0].set_xticks(x, names); ax[0].set_ylabel("estimation error [%]"); ax[0].set_ylim(-15, 15)
    ax[0].text(-0.3, 13, "(a) is off the scale (up to ~280%)", fontsize=7)
    ax[0].set_title("calibration with a hidden-truth virtual robot (5 trials)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    d0 = data[0]; dt = 1 / FS
    ax[1].plot(d0["t"], D(d0["gyro"]), color="gray", lw=.6, label="gyro (noisy, biased)")
    ax[1].plot(d0["t"], D(predicted_rate(pb[:3], d0["enc"], dt) + pb[3]), "C3", lw=1.2, label="model with estimated params")
    ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("base angular rate [deg/s]"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[1].set_title("loop L1: measured vs fitted (last trial)")
    fig.tight_layout(); fig.savefig(OUT / "field04_calibration.png", dpi=150)


if __name__ == "__main__":
    main()
