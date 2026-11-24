"""Field04b: 「答えを隠した仮想の実機」によるキャリブレーション（段階 4）の誤差評価.

段階 4（field04_calibration.py）の推定誤差を、次の 3 種類に分けて評価する。
  統計誤差: センサーの雑音と、隠した正解の値のばらつきによる、試行ごとのばらつき
  計算誤差: 仮想の実機（MuJoCo）の時間刻み・積分法、最適化の収束の許容値
  系統誤差: 雑音が推定式に入ることによる偏り（errors-in-variables）、推定側の仮定（総質量・一定のバイアス・雑音の大きさ・サンプリング）

主な発見: 部品表から組み立てた機体（assembly.build）は積分法が implicitfast（1 次）で、
時間刻み 0.5 ms では角運動量がわずかに保存されず、雑音がなくても推定値が +1.3〜1.8% 偏る。
姿勢の波形は質量・慣性の「比」しか決めないため、この小さなずれが推定しにくい方向（全体の大きさ）に拡大される。
仮想の実機を RK4 で計算すると、雑音なしの偏りは 0.1% 以下になる。以下の評価は RK4 を基準にする。

実行: cd sims && MUJOCO_GL=disable python field04b_error_budget.py（4 並列で約 30 分）
出力: data/field04b_error_budget.json, data/field04b_summary.txt, figures/field/field04b_error_budget.png
"""
import os
os.environ.setdefault("MUJOCO_GL", "disable")
import json
import time
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import field04_calibration as F
from common import DATA, FIG, smooth_step
from ambac.field.assembly import build, load
from ambac.field import space as fs

D = np.degrees
RAD = np.radians
M_TOT0, FS0 = F.M_TOT, F.FS
N_STAT = 50          # 統計誤差の試行回数
N_SEED = 20          # 系統誤差・感度の各条件の雑音の乱数の数
FIT = {"a": F.fit_netrotation, "b": F.fit_waveform, "bp": F.fit_pathintegral}
P_FIX = F.NOMINAL * np.array([1.05, 0.95, 1.03])   # 系統誤差・感度で固定する正解（設計値から少しずらす）


def simulate(p, ts=5e-4, integ="RK4", fs_hz=FS0):
    """仮想の実機で 6 ループを実行し、雑音のない生データを返す"""
    xml, _ = build(F.robot_dict(p), load("stages", "space"), timestep=ts)
    xml = xml.replace('integrator="implicitfast"', f'integrator="{integ}"')
    out = []
    for c1, c2, R in F.LOOPS:
        def ref(t, c1=c1, c2=c2, R=R):
            s, _ = smooth_step(min(t / F.T_LOOP, 1.0)); ph = 2 * np.pi * s
            return [c1 + R * np.cos(ph), c2 + R * np.sin(ph)]
        out.append(fs.run(xml, ref(0.0), ref, F.T_LOOP + 0.3, record_every=int(round(1 / (fs_hz * ts)))))
    return out


def sensors(raw, rng, k_gyro=1.0, k_enc=1.0, bias=None, drift=0.0):
    """生データにセンサーの雑音を加える. bias=None なら一様乱数（±BIAS_MAX）. drift: ループ中のバイアスの変化 [rad/s]"""
    if bias is None:
        bias = rng.uniform(-F.BIAS_MAX, F.BIAS_MAX)
    data = []
    for o in raw:
        n = len(o["t"])
        b = bias + drift * (o["t"] / o["t"][-1] - 0.5)
        gyro = o["gyro"] + b + (rng.normal(0, k_gyro * F.SIG_GYRO, n) if k_gyro > 0 else 0)
        enc = o["q"] + (rng.normal(0, k_enc * F.SIG_ENC, o["q"].shape) if k_enc > 0 else 0)
        data.append(dict(t=o["t"], gyro=gyro, enc=enc, tau=o["tau"], theta_true=o["theta"]))
    return data


def fit(data, methods, fs_hz=FS0, m_tot=M_TOT0, tight=False):
    F.FS, F.M_TOT = fs_hz, m_tot
    out = {}
    for k in methods:
        if tight:   # 収束の許容値を厳しくした最適化（計算誤差の確認用）
            import scipy.optimize as so
            orig = so.least_squares
            F.least_squares = lambda *a, **kw: orig(*a, **kw, xtol=1e-14, ftol=1e-14, gtol=1e-14)
        out[k] = FIT[k](data)[:3]
        if tight:
            F.least_squares = orig
    F.FS, F.M_TOT = FS0, M_TOT0
    return out


def err(est, p):
    return {k: ((v - p) / p * 100).tolist() for k, v in est.items()}


# ---- 各ジョブ -------------------------------------------------------------------
def job_stat(i):
    """統計誤差: 正解も雑音も乱数（元の段階 4 と同じ条件、RK4）"""
    rng = np.random.default_rng([2026, i])
    p = F.NOMINAL * np.array([rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15), rng.uniform(0.9, 1.1)])
    raw = simulate(p)
    return dict(kind="stat", i=i, p=p.tolist(), err=err(fit(sensors(raw, rng), ["a", "b", "bp"]), p))


RAW_FIX = {}


def raw_fix(ts=5e-4, integ="RK4", fs_hz=FS0):
    key = (ts, integ, fs_hz)
    if key not in RAW_FIX:
        RAW_FIX[key] = simulate(P_FIX, ts, integ, fs_hz)
    return RAW_FIX[key]


def job_cond(args):
    """固定した正解で、条件 cond・乱数 seed の推定を 1 回行う"""
    cond, seed = args
    rng = np.random.default_rng([7, seed])
    c = CONDS[cond]
    raw = raw_fix(c.get("ts", 5e-4), c.get("integ", "RK4"), c.get("fs", FS0))
    data = sensors(raw, rng, **c.get("sens", {}))
    est = fit(data, c.get("methods", ["b", "bp"]), fs_hz=c.get("fs", FS0), m_tot=c.get("m_tot", M_TOT0), tight=c.get("tight", False))
    return dict(kind="cond", cond=cond, seed=seed, err=err(est, P_FIX))


NOISELESS = dict(k_gyro=0, k_enc=0, bias=0.0)
CONDS = {
    # 計算誤差（雑音なし）
    **{f"num_{integ}_{ts:g}": dict(ts=ts, integ=integ, sens=NOISELESS, n=1)
       for integ in ("RK4", "implicitfast") for ts in (2e-3, 1e-3, 5e-4, 2.5e-4, 1e-4)},
    "num_RK4_5e-04_tight": dict(sens=NOISELESS, tight=True, n=1),
    # 系統誤差の切り分け（RK4、0.5 ms）
    "sys_gyro": dict(sens=dict(k_enc=0, bias=0.0), n=N_SEED),
    "sys_enc": dict(sens=dict(k_gyro=0, bias=0.0), n=N_SEED),
    "sys_bias": dict(sens=dict(k_gyro=0, k_enc=0), n=N_SEED),
    "sys_all": dict(sens={}, n=N_SEED),
    # 推定側の仮定への感度
    "sens_noise_x0.5": dict(sens=dict(k_gyro=0.5, k_enc=0.5), n=N_SEED),
    "sens_noise_x2": dict(sens=dict(k_gyro=2, k_enc=2), n=N_SEED),
    "sens_fs50": dict(fs=50.0, sens={}, n=N_SEED),
    "sens_fs200": dict(fs=200.0, sens={}, n=N_SEED),
    "sens_mtot-2%": dict(m_tot=M_TOT0 * 0.98, sens={}, n=N_SEED),
    "sens_mtot+2%": dict(m_tot=M_TOT0 * 1.02, sens={}, n=N_SEED),
    "sens_drift0.2": dict(sens=dict(drift=RAD(0.2)), n=N_SEED),
    "orig_implicitfast_all": dict(integ="implicitfast", sens={}, n=N_SEED),
}


def summarize(vals):
    v = np.array(vals)
    n = len(v)
    mean = v.mean(0)
    sd = v.std(0, ddof=1) if n > 1 else np.zeros(v.shape[1])
    return dict(n=n, mean=mean.tolist(), sd=sd.tolist(), se=(sd / np.sqrt(max(n, 1))).tolist(),
                ci95=(1.96 * sd / np.sqrt(max(n, 1))).tolist(), rms=np.sqrt((v**2).mean(0)).tolist())


def main():
    t0 = time.time()
    jobs = [(c, s) for c, v in CONDS.items() for s in range(v["n"])]
    with Pool(4) as pool:
        res_cond = pool.map(job_cond, jobs, chunksize=1)
        print(f"条件別の評価 {len(jobs)} 件（{time.time()-t0:.0f} s）", flush=True)
        res_stat = pool.map(job_stat, range(N_STAT), chunksize=1)
        print(f"統計誤差の評価 {N_STAT} 件（{time.time()-t0:.0f} s）", flush=True)

    summ = {}
    for c in CONDS:
        rs = [r for r in res_cond if r["cond"] == c]
        summ[c] = {m: summarize([r["err"][m] for r in rs]) for m in rs[0]["err"]}
    summ["stat"] = {m: summarize([r["err"][m] for r in res_stat]) for m in ("a", "b", "bp")}
    json.dump(dict(p_fix=P_FIX.tolist(), n_stat=N_STAT, n_seed=N_SEED, summary=summ, stat=res_stat, cond=res_cond,
                   settings=dict(gyro_noise_deg_s=0.2, bias_max_deg_s=0.2, encoder_noise_deg=0.05, fs_hz=FS0,
                                 timestep=5e-4, integrator_reference="RK4")),
              open(DATA / "field04b_error_budget.json", "w"), indent=1, ensure_ascii=False)

    lines = ["=== 段階 4 の誤差評価（推定誤差 [%]：腕の質量, 脚の質量, 胴体の慣性）===",
             f"固定した正解 P_FIX = {np.round(P_FIX, 3).tolist()}、雑音の乱数 {N_SEED} 通り、統計誤差の試行 {N_STAT} 回",
             "平均 ± 標準偏差（n>1 のとき）。b = 波形（微分）、bp = 波形（微分なし）、a = 正味の回転だけ"]
    def fmt(s):
        return ", ".join(f"{m:+.3f}±{sd:.3f}" if s["n"] > 1 else f"{m:+.3f}" for m, sd in zip(s["mean"], s["sd"]))
    for c, v in summ.items():
        for m, s in v.items():
            lines.append(f"  {c:26s} {m:3s}: {fmt(s)}" + (f"  (95% CI 半幅 {', '.join(f'{x:.3f}' for x in s['ci95'])})" if c == "stat" else ""))
    lines.append(f"計算時間 {time.time()-t0:.0f} s")
    txt = "\n".join(lines); print(txt)
    (DATA / "field04b_summary.txt").write_text(txt + "\n")
    plot(summ, res_stat)


def plot(summ, res_stat):
    names = ["arm mass", "leg mass", "torso inertia"]
    x = np.arange(3)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    # (1) 統計誤差: 50 回の分布
    for j, (m, col, lab) in enumerate([("b", "C1", "(b) waveform, differentiated"), ("bp", "C3", "(b') waveform, no differentiation")]):
        e = np.array([r["err"][m] for r in res_stat]); s = summ["stat"][m]
        ax[0].plot(x[:, None] + (j - 0.5) * 0.3 + np.random.default_rng(0).uniform(-.06, .06, (3, len(e))), e.T, ".", color=col, alpha=.35)
        ax[0].errorbar(x + (j - 0.5) * 0.3, s["mean"], yerr=s["sd"], fmt="o", color=col, capsize=4, label=lab + " (mean ± SD)")
    ax[0].axhline(0, color="k", lw=.6); ax[0].set_xticks(x, names); ax[0].set_ylabel("estimation error [%]")
    ax[0].set_title(f"statistical: {N_STAT} hidden-truth trials (RK4)"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    # (2) 計算誤差: 時間刻みと積分法（雑音なし、b'）
    tss = [2e-3, 1e-3, 5e-4, 2.5e-4, 1e-4]
    for integ, mk in (("RK4", "o-"), ("implicitfast", "s--")):
        e = np.array([summ[f"num_{integ}_{ts:g}"]["bp"]["mean"] for ts in tss])
        for k in range(3):
            ax[1].plot(np.array(tss) * 1e3, e[:, k], mk, color=f"C{k}", label=f"{integ}, {names[k]}")
    ax[1].axvline(0.5, color="gray", ls=":", lw=1); ax[1].axhline(0, color="k", lw=.6)
    ax[1].set_xscale("log"); ax[1].set_xlabel("MuJoCo timestep [ms]"); ax[1].set_ylabel("error without noise [%]")
    ax[1].set_title("numerical: integrator of the virtual robot (b')"); ax[1].legend(fontsize=6, ncol=2); ax[1].grid(alpha=.3)
    # (3) 系統誤差・仮定への感度（b'、平均 ± 95% CI）
    conds = [("sys_gyro", "gyro noise only"), ("sys_enc", "encoder noise only"), ("sys_bias", "gyro bias only"), ("sys_all", "all noise"),
             ("sens_noise_x2", "noise x2"), ("sens_fs50", "50 Hz"), ("sens_mtot+2%", "total mass +2%"),
             ("sens_mtot-2%", "total mass -2%"), ("sens_drift0.2", "bias drift 0.2 deg/s"), ("orig_implicitfast_all", "implicitfast (original)")]
    y = np.arange(len(conds))[::-1]
    for k in range(3):
        mm = [summ[c]["bp"]["mean"][k] for c, _ in conds]; ci = [summ[c]["bp"]["ci95"][k] for c, _ in conds]
        ax[2].errorbar(mm, y + (k - 1) * 0.22, xerr=ci, fmt="o", ms=4, color=f"C{k}", capsize=2, label=names[k])
    ax[2].axvline(0, color="k", lw=.6); ax[2].set_yticks(y, [l for _, l in conds], fontsize=8)
    ax[2].set_xlabel("mean error [%] (95% CI)"); ax[2].set_title("systematic: noise and assumptions (b')")
    ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "field" / "field04b_error_budget.png", dpi=150)


if __name__ == "__main__":
    main()
