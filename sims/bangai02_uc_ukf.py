"""番外編 02：振り子の周期を、UC-AutoMaiL（TestPyPI 0.4.0a9）の UKF / EKF で逐次に推定し、番外編 01 の一括当てはめと誤差を比べる.

番外編 01（bangai01_bar_imu.py）は、2 台のジャイロに振り子の運動方程式を一括で当てはめ（GLS、AR(1) の白色化）、
合成データ 60 回の pull =（推定 − 正解）/ σ の標準偏差が 0.88（白色化なしの OLS は 1.50）だった。
ここでは同じ合成データの 1 台目のジャイロだけを、状態 [θ, θ', ω0, γ, β, b1]（＋ AR(1) の雑音 n）として
uc_automail.unscented_kalman_filter / extended_kalman_filter に一括で渡し、最後の時刻の ω0 と σ から T0 と pull を出す。

  white: 観測の雑音を白色とみなす（静止区間の σ をそのまま R に使う）
  ar1  : 雑音 n を状態に足す（n_k = ρ n_{k−1} + e、Var e = σ²(1 − ρ²)）。観測 = θ' + b1 + n、R はごく小さい値
比べること: T0 の pull の平均と標準偏差（1 なら σ が正直）、ω0 の σ の大きさ、計算時間、
            フィルタ自身の整合性の目安（NIS = 正規化したイノベーションの 2 乗の平均、1 が理想）と、イノベーションの 1 次の自己相関
実行: python sims/bangai02_uc_ukf.py [--n 40]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from multiprocessing import Pool

import numpy as np
import uc_automail as UC

import bangai01_bar_imu as B
from common import DATA

NAMES = ["th", "om", "w0", "gam", "beta", "b1"]


def rk4(x, dt, nsub=2):
    th, om, w0, gam, beta = x[:5]
    h = dt / nsub
    f = lambda a, b: (b, -w0 ** 2 * math.sin(a) - 2 * gam * b - beta * b * abs(b))  # noqa: E731
    for _ in range(nsub):
        k1 = f(th, om); k2 = f(th + h / 2 * k1[0], om + h / 2 * k1[1])
        k3 = f(th + h / 2 * k2[0], om + h / 2 * k2[1]); k4 = f(th + h * k3[0], om + h * k3[1])
        th += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); om += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return th, om


def num_jac(fun, x, eps=1e-7):
    f0 = fun(x); J = np.empty((len(f0), len(x)))
    for j in range(len(x)):
        dx = np.zeros(len(x)); dx[j] = eps * max(1.0, abs(x[j])); J[:, j] = (fun(x + dx) - f0) / dx[j]
    return J


def run_filter(t, y, P, method, noise):
    st = P["static"]; x0 = P["x0"]; rho = st["rho1"]; sig = st["sig1"]
    ar = noise == "ar1"
    ns = 7 if ar else 6
    dts = np.diff(t)

    def f(x, k):
        dt = dts[k - 1]
        th, om = rk4(x, dt)
        out = np.r_[th, om, x[2:6]]
        if ar:
            out = np.r_[out, rho ** (dt / 0.01) * x[6]]
        return out

    h = (lambda x, _k: np.array([x[1] + x[5] + x[6]])) if ar else (lambda x, _k: np.array([x[1] + x[5]]))  # noqa: E731
    m0 = [x0["th_a"], x0["om_a"], x0["w0"], max(x0["gam"], 1e-4), 0.0, st["b1"]] + ([0.0] if ar else [])
    sd0 = [0.02, 0.02, 0.02 * x0["w0"], 0.005, 0.02, 0.003] + ([sig] if ar else [])
    Q = np.zeros((len(t) - 1, ns, ns))
    Q[:, 0, 0] = Q[:, 1, 1] = 1e-12             # 状態の雑音はほぼ 0（モデルが正しい前提）
    if ar:
        Q[:, 6, 6] = sig ** 2 * (1 - rho ** (2 * dts / 0.01))
    R = np.array([[1e-10]]) if ar else np.array([[sig ** 2]])
    kw = dict(initial_mean=np.array(m0), initial_covariance=np.diag(np.square(sd0)), transition=f, measurement=h,
              process_covariance=Q, observation_covariance=R)
    t0 = time.perf_counter()
    if method == "ukf":
        r = UC.unscented_kalman_filter(y[1:, None], **kw)
    else:
        Hm = np.zeros((1, ns)); Hm[0, 1] = Hm[0, 5] = 1
        if ar:
            Hm[0, 6] = 1
        r = UC.extended_kalman_filter(y[1:, None], transition_jacobian=lambda x, k: num_jac(lambda z: f(z, k), x),
                                      measurement_jacobian=lambda x, _k: Hm, **kw)
    sec = time.perf_counter() - t0
    xf, Pf = r["filtered"][-1], r["filtered_covariance"][-1]
    res = r["predicted_observation"][:, 0] - y[1:]
    S = r["predicted_observation_covariance"][:, 0, 0]
    u = res / np.sqrt(S)
    return dict(w0=float(xf[2]), sd_w0=float(math.sqrt(Pf[2, 2])), sec=sec, nis=float(np.mean(u ** 2)),
                ac1=float(np.corrcoef(u[:-1], u[1:])[0, 1]))     # 正規化したイノベーションの 1 次の自己相関（白色なら 0）


def toy(i):
    rng = np.random.default_rng([20261126, i])      # 番外編 01 の _toy_pend と同じ乱数（同じ合成データ）
    phones, tr = B.synth_pendulum(rng, theta0_deg=rng.uniform(7, 13), tau2=rng.uniform(0.1, 0.6), gam=rng.uniform(0.001, 0.006),
                                  beta=rng.uniform(0.0, 0.03), source="ode")
    cfg = B.default_config("pendulum")
    out = dict(i=i, T0_true=tr["T0"])
    try:
        D, P = B.prepare_pendulum(*phones, cfg)
        sel = (D["t1"] >= P["t_a"]) & (D["t1"] <= P["t_b"])
        t, y = D["t1"][sel], D["w1"][sel]
        for method in ("ukf", "ekf"):
            for noise in ("white", "ar1"):
                r = run_filter(t, y, P, method, noise)
                T0 = 2 * math.pi / r["w0"]; sd = 2 * math.pi / r["w0"] ** 2 * r["sd_w0"]
                out[f"{method}_{noise}"] = dict(T0=T0, sd=sd, pull=(T0 - tr["T0"]) / sd, sec=r["sec"], nis=r["nis"], ac1=r["ac1"], n=len(t))
    except Exception as e:  # noqa: BLE001  失敗も記録する
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=40); a = ap.parse_args()
    T = time.time()
    with Pool(4) as pool:
        res = pool.map(toy, range(a.n))
    ref = json.load(open(DATA / "bangai01" / "toys.json"))["summary"]["pend"]
    summ = {}
    print(f"=== 合成データ {a.n} 回、T0 の pull（平均 ± 標準偏差）。参照: 番外編 01 の GLS {ref['pull']['mean']:+.2f} ± {ref['pull']['sd']:.2f}、"
          f"OLS（白色化なし） {ref['pull_ols']['mean']:+.2f} ± {ref['pull_ols']['sd']:.2f} ===")
    for key in ("ukf_white", "ukf_ar1", "ekf_white", "ekf_ar1"):
        rr = [r[key] for r in res if key in r]
        p = np.array([r["pull"] for r in rr]); sd = np.array([r["sd"] for r in rr]) * 1e6
        summ[key] = dict(n=len(p), pull_mean=float(p.mean()), pull_sd=float(p.std(ddof=1)), sd_T0_us_median=float(np.median(sd)),
                         sec_mean=float(np.mean([r["sec"] for r in rr])), nis_mean=float(np.mean([r["nis"] for r in rr])), ac1_mean=float(np.mean([r["ac1"] for r in rr])),
                         n_samples=int(np.median([r["n"] for r in rr])))
        s = summ[key]
        print(f"| {key} | pull {s['pull_mean']:+.2f} ± {s['pull_sd']:.2f} (±{s['pull_sd'] / math.sqrt(2 * (len(p) - 1)):.2f}) | "
              f"σ(T0) 中央値 {s['sd_T0_us_median']:.0f} μs | NIS 平均 {s['nis_mean']:.2f} | 自己相関 {s['ac1_mean']:+.2f} | {s['sec_mean']:.1f} s / {s['n_samples']} 点 |", flush=True)
    errs = [r for r in res if "error" in r]
    print("失敗:", [(r["i"], r["error"]) for r in errs])
    json.dump(dict(summary=summ, ref=dict(gls=ref["pull"], ols=ref["pull_ols"]), toys=res, failures=errs, seconds=time.time() - T),
              open(DATA / "bangai02_uc_ukf.json", "w"), indent=1, ensure_ascii=False, default=float)
    print(f"[{time.time() - T:.0f} s]")


if __name__ == "__main__":
    main()
