"""Field30: 自作パッケージ UC-AutoMaiL（TestPyPI 0.4.0a9）の EKF / UKF を、重心の状態推定に差し替えて試す（下半身の段 13）.

段階 25・26 の推定器（field21_walk.Estimator、type "lipm"）は、軸ごとに 2 状態（c, v）の線形カルマンフィルタを手で書いている。
ここでは同じ測定（関節角の運動学の c_kin、足圧中心 p、雑音は段階 25 の基準）を UC-AutoMaiL の
  uc_automail.extended_kalman_filter / uc_automail.unscented_kalman_filter
に渡して、1 ms ごとに 1 行ずつ（前の事後分布を次の初期値にして）呼ぶ。

  uc_lin: 状態 4（c_x, v_x, c_y, v_y）、LIPM（c̈ = ω²(c − p)）の線形モデル。手書きの KF と数式は同じ → 結果が一致するかの検算
  uc_aug: 状態 5（… ＋ w = ω²）。w を未知数にした非線形モデル（重心の高さが分からない場合）。w の初期値は真の ω² の om_scale² 倍
  手書き KF でも om_scale を使えるようにし、「ω を 1 割まちがえた固定のモデル」と「ω² も推定する EKF/UKF」を比べる

比べること: 歩ける範囲（段階 23 と同じ 24 通り、足の場所はオドメトリ、K 0.3）、押されたときの回復、推定の誤差の RMS、
            フィルタ自身が出す σ と実際の誤差の比（pull の RMS。1 なら共分散が正直）、1 回の呼び出しにかかる時間
実行: python field30_uc_automail.py [main|figure|timing]   （UC-AutoMaiL は pip install -i https://test.pypi.org/simple/ UC-AutoMaiL）
"""
import json
import sys
import time

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import uc_automail as UC

import field21_walk as W
import field23_stabilizer as F23
import field25_estimation as F25
from common import DATA

T0 = time.time()
OUT = W.OUT
NOISE, NOISE3 = F25.NOISE, F25.NOISE3
BASE = {"sw": 10, "anchor": "odo", "_K": 0.3, **NOISE}
OM_OFF = 0.9          # ω を 1 割小さく思い込む（ω² は 0.81 倍）
CASES = [
    ("手書き KF（段階 26 と同じ）", {"type": "lipm", **BASE}),
    ("UC-EKF、線形 LIPM", {"type": "uc_ekf_lin", **BASE}),
    ("UC-UKF、線形 LIPM", {"type": "uc_ukf_lin", **BASE}),
    ("手書き KF、ω を 0.9 倍", {"type": "lipm", "om_scale": OM_OFF, **BASE}),
    ("UC-EKF、ω² も推定（初期 0.81 倍）", {"type": "uc_ekf_aug", "om_scale": OM_OFF, **BASE}),
    ("UC-UKF、ω² も推定（初期 0.81 倍）", {"type": "uc_ukf_aug", "om_scale": OM_OFF, **BASE}),
    ("UC-EKF、ω² も推定（初期は正しい）", {"type": "uc_ekf_aug", **BASE}),
    ("UC-UKF、ω² も推定、雑音 3 倍", {"type": "uc_ukf_aug", "om_scale": OM_OFF, **{**BASE, **NOISE3}}),
]
PCASES = [0, 1, 2, 3, 4, 5]
LS, TS, FS = F23.LS, F23.TS, F25.FS


class UCEstimator(W.Estimator):
    """field21_walk.Estimator の "lipm" と同じ測定を、UC-AutoMaiL のフィルタで処理する"""

    def __init__(self, cfg, rb, m, om, rng, c):
        super().__init__(cfg, rb, m, om * cfg.get("om_scale", 1.0), rng, c)
        self.om_true = om
        self.pull = []; self.wlog = []; self.calls = 0; self.call_s = 0.0

    def step(self, d, feet, support, dt):
        typ = self.cfg.get("type", "lipm")
        if not typ.startswith("uc_"):
            return super().step(d, feet, support, dt)
        cf = self.cfg
        ck, _ = self.c_kin(d, feet, support)
        if hasattr(self, "anc"):
            self.anc_err = getattr(self, "anc_err", [])
            self.anc_err.append(np.r_[d.time, np.ravel([self.anc[sd] - d.site_xpos[self.rb.site[sd]][:2] for sd in "LR"])])
        mujoco.mj_subtreeVel(self.m, d)
        ct, vt = d.subtree_com[1][:2].copy(), d.subtree_linvel[1][:2].copy()
        p = self.cop(d)
        p = np.zeros(2) if p is None else p           # 手書き KF と同じ扱い（足圧中心がないときは p = 0）
        aug = typ.endswith("_aug")
        ns = 5 if aug else 4
        om2 = self.om ** 2
        sw = cf.get("sw", 1.0); Rk = cf.get("sr", 0.005) ** 2
        Gm = np.array([dt * dt / 2, dt]); q2 = np.outer(Gm, Gm) * sw ** 2
        Q = np.zeros((ns, ns)); Q[0:2, 0:2] = q2; Q[2:4, 2:4] = q2
        if aug:
            Q[4, 4] = (cf.get("sw_w", 0.05) * om2) ** 2 * dt      # w は 1 √s あたり 5 % のランダムウォーク
        if self.x is None:
            x0 = [ck[0], 0.0, ck[1], 0.0] + ([om2] if aug else [])
            P0 = np.diag([1e-4, 1e-2, 1e-4, 1e-2] + ([(cf.get("w_sd", 0.3) * om2) ** 2] if aug else []))
            self.x, self.Pk = np.array(x0), P0

        def f(x, _k):
            w = x[4] if aug else om2
            return np.r_[x[0] + dt * x[1], x[1] + dt * w * (x[0] - p[0]),
                         x[2] + dt * x[3], x[3] + dt * w * (x[2] - p[1]), x[4:]]

        def F(x, _k):
            w = x[4] if aug else om2
            J = np.eye(ns); J[0, 1] = J[2, 3] = dt; J[1, 0] = J[3, 2] = dt * w
            if aug:
                J[1, 4] = dt * (x[0] - p[0]); J[3, 4] = dt * (x[2] - p[1])
            return J

        Hm = np.zeros((2, ns)); Hm[0, 0] = Hm[1, 2] = 1.0
        h = lambda x, _k: Hm @ x       # noqa: E731
        Hj = lambda x, _k: Hm          # noqa: E731
        t0 = time.perf_counter()
        if typ.startswith("uc_ekf"):
            r = UC.extended_kalman_filter(ck[None, :], initial_mean=self.x, initial_covariance=self.Pk, transition=f,
                                          transition_jacobian=F, measurement=h, measurement_jacobian=Hj,
                                          process_covariance=Q, observation_covariance=np.eye(2) * Rk)
        else:
            r = UC.unscented_kalman_filter(ck[None, :], initial_mean=self.x, initial_covariance=self.Pk, transition=f,
                                           measurement=h, process_covariance=Q, observation_covariance=np.eye(2) * Rk)
        self.call_s += time.perf_counter() - t0; self.calls += 1
        self.x, self.Pk = r["filtered"][0], r["filtered_covariance"][0]
        out_c, out_v = self.x[[0, 2]], self.x[[1, 3]]
        e = np.r_[out_c - ct, out_v - vt]
        sd = np.sqrt(np.diag(self.Pk)[[0, 2, 1, 3]])
        self.pull.append(e / sd)
        if aug:
            self.wlog.append(self.x[4] / self.om_true ** 2)
        self.err.append(e)
        return out_c, out_v


W.Estimator = UCEstimator      # field21_walk.execute が使う推定器を差し替える（このスクリプトの中だけ）


def run(args):
    over, est, push = args
    est = dict(est); K = est.pop("_K", 1.0)
    ex = {"arms": {"A": F23.A_STAR}, "stab": {"K": K, "est": est}}
    if push is not None:
        ex["push"] = push
    holder = {}
    orig = UCEstimator.__init__

    def init(self, *a, **k):
        orig(self, *a, **k); holder["e"] = self
    UCEstimator.__init__ = init
    try:
        r = W.execute(W.Cfg(extra=ex, **over))
    except Exception as e:  # noqa: BLE001
        r = dict(ok=False, fell=True, err=f"{type(e).__name__}: {e}")
    finally:
        UCEstimator.__init__ = orig
    es = holder.get("e")
    if es is not None and es.pull:
        P = np.array(es.pull); r["pull_rms"] = np.sqrt(np.mean(P ** 2, 0)).tolist()
        r["call_us"] = 1e6 * es.call_s / max(es.calls, 1)
        if es.wlog:
            r["w_end"] = float(es.wlog[-1]); r["w_mean2"] = float(np.mean(es.wlog[len(es.wlog) // 2:]))
    r.pop("anc_err", None)
    return r


def seq(rs):
    return "".join("○" if r["ok"] else "×" for r in rs)


def main():
    from multiprocessing import Pool
    tp = F23.push_time(0.8)
    jobs = []
    for _, est in CASES:
        jobs += [({"L": L, "T_step": T}, est, None) for L in LS for T in TS]
    for ci in PCASES:
        jobs += [({"T_step": 0.8}, CASES[ci][1], {"t": tp, "F": [dx * f, dy * f], "dur": 0.1}) for dx, dy in ((1, 0), (0, 1)) for f in FS]
    with Pool(4) as pool:
        res = pool.map(run, jobs, chunksize=2)
    ng = len(LS) * len(TS); out = {"grid": [], "push": [], "LS": LS, "TS": TS, "FS": FS}
    print("=== 歩ける範囲（オドメトリ、K 0.3、腕の振りあり）。推定の誤差の RMS（c_x, c_y [mm]、v_x, v_y [cm/s]）、pull の RMS（c_x, c_y, v_x, v_y）、1 回の呼び出し [μs]、ω² の推定 / 真（後半の平均） ===")
    for a, (name, est) in enumerate(CASES):
        rs = res[a * ng:(a + 1) * ng]
        ok = [r for r in rs if r.get("ok") and "est_rms" in r]
        er = np.array([r["est_rms"] for r in ok]); pr = np.array([r["pull_rms"] for r in ok if "pull_rms" in r])
        cu = [r["call_us"] for r in rs if "call_us" in r]; wm = [r["w_mean2"] for r in ok if "w_mean2" in r]
        es = f"{er[:, 0].mean() * 1e3:.1f}, {er[:, 1].mean() * 1e3:.1f} mm、{er[:, 2].mean() * 100:.1f}, {er[:, 3].mean() * 100:.1f} cm/s" if len(er) else "−"
        ps = ", ".join(f"{v:.1f}" for v in pr.mean(0)) if len(pr) else "−"
        fr = np.mean([r["ok"] for r in rs])
        errs = sorted({r["err"] for r in rs if "err" in r})
        print(f"| {name} | {' '.join(seq(rs[b * len(TS):(b + 1) * len(TS)]) for b in range(len(LS)))} | {fr:.2f} | {es} | pull {ps} | "
              f"{np.mean(cu) if cu else float('nan'):.0f} μs | w {np.mean(wm) if wm else float('nan'):.3f} | {errs[:2]}", flush=True)
        out["grid"].append(dict(name=name, est=est, frac=float(fr), ok=[r["ok"] for r in rs],
                                est_rms=er.mean(0).tolist() if len(er) else None, pull_rms=pr.mean(0).tolist() if len(pr) else None,
                                call_us=float(np.mean(cu)) if cu else None, w_ratio=float(np.mean(wm)) if wm else None,
                                w_ratio_all=wm, errors=errs))
    base = len(CASES) * ng; n = len(FS)
    print(f"=== 押されたときの回復（T 0.8 s、0.1 s、力 {FS} N）: 前 / 横 ===")
    for j, ci in enumerate(PCASES):
        rr = res[base + j * 2 * n: base + (j + 1) * 2 * n]
        lims = []
        for b in range(2):
            lim = 0
            for f_, r in zip(FS, rr[b * n:(b + 1) * n]):
                if not r["ok"]:
                    break
                lim = f_
            lims.append(lim)
        print(f"| {CASES[ci][0]} | 前 {lims[0] / 10:.0f} N·s {seq(rr[:n])} | 横 {lims[1] / 10:.0f} N·s {seq(rr[n:])} |", flush=True)
        out["push"].append(dict(name=CASES[ci][0], fwd=lims[0] / 10, lat=lims[1] / 10))
    # 検算: 手書き KF と UC-EKF / UC-UKF（線形）で、歩けたかどうかが歩き方ごとに一致するか
    agree = {CASES[k][0]: int(np.sum(np.array(out["grid"][k]["ok"]) == np.array(out["grid"][0]["ok"]))) for k in (1, 2)}
    out["agree_with_handwritten"] = agree
    print("手書き KF と一致した歩き方の数（24 中）:", agree)
    json.dump(out, open(DATA / "field30_uc_automail.json", "w"), indent=1, ensure_ascii=False, default=float)
    print(f"[{time.time() - T0:.0f} s]")


def timing():
    """1 ステップ分の呼び出し時間と、線形モデルでの手書き KF との数値の一致（同じ入力列）"""
    rng = np.random.default_rng(1); dt = 1e-3; om2 = 9.81 / 0.25; n = 2000
    c = np.cumsum(rng.normal(0, 1e-3, (n, 2)), 0); p = c + rng.normal(0, 0.01, (n, 2)); z = c + rng.normal(0, 0.005, (n, 2))
    A = np.array([[1, dt], [om2 * dt, 1]]); Bv = np.array([0, -om2 * dt]); Gm = np.array([dt * dt / 2, dt]); q2 = np.outer(Gm, Gm) * 100
    xs = [np.array([z[0, a], 0.0]) for a in range(2)]; Ps = [np.diag([1e-4, 1e-2])] * 2
    x4 = np.array([z[0, 0], 0, z[0, 1], 0.0]); P4 = np.diag([1e-4, 1e-2, 1e-4, 1e-2]); Q = np.zeros((4, 4)); Q[:2, :2] = Q[2:, 2:] = q2
    Hm = np.zeros((2, 4)); Hm[0, 0] = Hm[1, 2] = 1
    diff = 0.0; t_hand = t_uc = t_ukf = 0.0; x4u, P4u = x4.copy(), P4.copy()
    for k in range(n):
        t0 = time.perf_counter()
        for a in range(2):
            x, P_ = xs[a], Ps[a]
            x = A @ x + Bv * p[k, a]; P_ = A @ P_ @ A.T + q2; S = P_[0, 0] + 0.005 ** 2; K = P_[:, 0] / S
            x = x + K * (z[k, a] - x[0]); P_ = P_ - np.outer(K, P_[0, :]); xs[a], Ps[a] = x, P_
        t_hand += time.perf_counter() - t0
        f = lambda x, _k, pk=p[k]: np.r_[x[0] + dt * x[1], x[1] + dt * om2 * (x[0] - pk[0]), x[2] + dt * x[3], x[3] + dt * om2 * (x[2] - pk[1])]  # noqa: E731
        J = np.eye(4); J[0, 1] = J[2, 3] = dt; J[1, 0] = J[3, 2] = dt * om2
        t0 = time.perf_counter()
        r = UC.extended_kalman_filter(z[k][None], initial_mean=x4, initial_covariance=P4, transition=f, transition_jacobian=lambda x, _k: J,
                                      measurement=lambda x, _k: Hm @ x, measurement_jacobian=lambda x, _k: Hm,
                                      process_covariance=Q, observation_covariance=np.eye(2) * 0.005 ** 2)
        t_uc += time.perf_counter() - t0
        t0 = time.perf_counter()
        ru = UC.unscented_kalman_filter(z[k][None], initial_mean=x4u, initial_covariance=P4u, transition=f,
                                        measurement=lambda x, _k: Hm @ x, process_covariance=Q, observation_covariance=np.eye(2) * 0.005 ** 2)
        t_ukf += time.perf_counter() - t0
        x4, P4 = r["filtered"][0], r["filtered_covariance"][0]; x4u, P4u = ru["filtered"][0], ru["filtered_covariance"][0]
        diff = max(diff, np.max(np.abs(x4[[1, 3]] - np.r_[xs[0][1], xs[1][1]])), np.max(np.abs(x4u[[1, 3]] - x4[[1, 3]])))
    # 一括（オフライン）で 1 回呼んだ場合（p が時刻ごとに違うので step から引く）
    fb = lambda x, k: np.r_[x[0] + dt * x[1], x[1] + dt * om2 * (x[0] - p[k - 1, 0]), x[2] + dt * x[3], x[3] + dt * om2 * (x[2] - p[k - 1, 1])]  # noqa: E731
    J = np.eye(4); J[0, 1] = J[2, 3] = dt; J[1, 0] = J[3, 2] = dt * om2
    t0 = time.perf_counter()
    UC.extended_kalman_filter(z, initial_mean=np.array([z[0, 0], 0, z[0, 1], 0.0]), initial_covariance=np.diag([1e-4, 1e-2, 1e-4, 1e-2]),
                              transition=fb, transition_jacobian=lambda x, _k: J, measurement=lambda x, _k: Hm @ x,
                              measurement_jacobian=lambda x, _k: Hm, process_covariance=Q, observation_covariance=np.eye(2) * 0.005 ** 2)
    t_batch = time.perf_counter() - t0
    res = dict(n=n, hand_us=1e6 * t_hand / n, uc_ekf_us=1e6 * t_uc / n, uc_ukf_us=1e6 * t_ukf / n, uc_ekf_batch_us=1e6 * t_batch / n,
               max_abs_diff_velocity=float(diff))
    print(json.dumps(res, indent=1))
    json.dump(res, open(DATA / "field30_uc_timing.json", "w"), indent=1)


def figure():
    res = json.load(open(DATA / "field30_uc_automail.json"))
    en = ["hand-written KF (stage 26)", "UC-EKF, linear LIPM", "UC-UKF, linear LIPM", "hand-written KF, omega x0.9",
          "UC-EKF, estimate omega^2 (init x0.81)", "UC-UKF, estimate omega^2 (init x0.81)", "UC-EKF, estimate omega^2 (init true)",
          "UC-UKF, estimate omega^2, noise x3"]
    col = ["k", "tab:blue", "tab:cyan", "tab:gray", "tab:red", "tab:orange", "tab:purple", "tab:brown"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    fr = [g["frac"] for g in res["grid"]]; yy = np.arange(len(fr))
    ax[0].barh(yy, fr, color=col); ax[0].set_yticks(yy); ax[0].set_yticklabels(en, fontsize=8); ax[0].invert_yaxis()
    ax[0].set_xlabel("fraction of 24 gaits walked (odometry, K 0.3)"); ax[0].set_xlim(0, 1)
    ax[0].set_title("UC-AutoMaiL 0.4.0a9 filters in the walking loop", fontsize=9)
    vv = [(g["est_rms"][2] * 100 if g["est_rms"] else np.nan) for g in res["grid"]]
    pv = [(g["pull_rms"][2] if g.get("pull_rms") else np.nan) for g in res["grid"]]
    ax[1].scatter(vv, fr, c=col)
    for j, (v_, f_, n_) in enumerate(zip(vv, fr, en)):
        if j in (1, 2):      # 線形の UC-EKF / UC-UKF は手書き KF と同じ点に重なる
            continue
        if j == 0:
            n_ = "hand-written KF = UC-EKF = UC-UKF (linear)"
        if np.isfinite(v_):
            ax[1].annotate(n_, (v_, f_), fontsize=6, xytext=(3, -9 if j == 0 else 3), textcoords="offset points")
    ax[1].set_xlabel("forward CoM velocity error RMS [cm/s]"); ax[1].set_ylabel("fraction walked"); ax[1].set_title("success vs velocity error", fontsize=9)
    ax[2].barh(yy, pv, color=col); ax[2].axvline(1, color="gray", ls=":"); ax[2].set_xscale("log")
    ax[2].set_yticks(yy); ax[2].set_yticklabels([]); ax[2].invert_yaxis()
    for j, v_ in enumerate(pv):
        if not np.isfinite(v_):
            ax[2].text(0.62, j, "not recorded", va="center", fontsize=7, color="gray")
    ax[2].set_ylim(ax[0].get_ylim())
    ax[2].set_xlabel("RMS(error / filter sigma), v_x  (1 = honest)"); ax[2].set_title("does the filter's own sigma cover the error?", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "field30_uc_automail.png", dpi=130)
    print("図 → figures/field/field30_uc_automail.png")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "main"
    {"main": main, "figure": figure, "timing": timing}[arg]()
