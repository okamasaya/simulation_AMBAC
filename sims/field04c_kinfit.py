"""Field04c: 拘束条件つきの当てはめ（素粒子実験の kinematic fit の考え方）による校正と、その誤差の自己評価.

素粒子実験の kinematic fit は、測った量（運動量など）をその誤差の範囲で動かし、保存則（拘束条件）を厳密に満たすように当てはめる。
得られるのは、(1) 推定値、(2) 推定値ごとの誤差（共分散）、(3) 当てはまりの良さ（χ²）で、
pull（(推定 − 正解) / 推定した誤差）が平均 0・標準偏差 1 になることで、誤差の見積もりが正しいことを確かめる。

ここでは、校正の問題を次のように書き直す。
  測った量: エンコーダの関節角 q_m（誤差 σ_q）、ジャイロの角速度 ω_m（誤差 σ_ω）
  当てはめる量: 真の関節角 Q（各サンプル）、パラメータ p（腕・脚の質量、胴体の慣性の倍率）、ジャイロのバイアス b
  拘束条件: 角運動量 L = 0、すなわち ω = −Γ(Q; p)·Q'（Q' は当てはめた Q の中心差分）
  χ² = Σ (Q − q_m)²/σ_q² + Σ (ω_m − b + Γ(Q; p)·Q')²/σ_ω²
関節角も「誤差のある測定値」として動かすので、雑音の入った関節角を微分して偏る問題（errors-in-variables）が起きない。

逐次更新: ループを 1 本ずつ加え、前の推定値と共分散を事前情報として当てはめ直す（カルマンフィルタと同じ形）。

実行: cd sims && MUJOCO_GL=disable python field04c_kinfit.py（4 並列で約 20 分）
出力: data/field04c_kinfit.json, data/field04c_summary.txt, figures/field/field04c_kinfit.png
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
from scipy.optimize import least_squares
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.stats import chi2 as chi2dist

import field04_calibration as F
import field04b_error_budget as E
from common import DATA, FIG

SQ, SW = F.SIG_ENC, F.SIG_GYRO
N_TRIAL = 50
N_IMPL = 20


# ---- 機械的接続 Γ(q; p) をまとめて計算する（PlanarRobot.connection と同じ式） -------------
def gamma_vec(Q, p, m_tot=None):
    """Q: (N, 2)。戻り値 (N, 2)"""
    m_tot = F.M_TOT if m_tot is None else m_tot
    r = F.planar(p)
    mb, Ib = r.base_mass, r.base_inertia
    m = np.array([l.mass for l in r.limbs]); I = np.array([l.inertia for l in r.limbs])
    a = np.array([l.attach for l in r.limbs]); rc = np.array([l.r_com for l in r.limbs])
    u = np.stack([np.cos(Q), np.sin(Q)], -1) * rc[None, :, None]        # (N, 2, 2) リンクの重心への向き
    pos = a[None] + u                                                   # 胴体の重心を原点とした位置
    M = mb + m.sum()
    c = (m[None, :, None] * pos).sum(1) / M                              # 系の重心
    A = Ib + I.sum() + mb * (c**2).sum(-1) + (m[None] * ((pos - c[:, None])**2).sum(-1)).sum(1)
    B = I[None] + m[None] * ((pos - c[:, None]) * u).sum(-1)
    return B / A[:, None]


def model_rate(Q, p, dt):
    """内側のサンプルでの ω_model = −Γ(Q)·Q'（Q' は中心差分）"""
    G = gamma_vec(Q[1:-1], p)
    Qd = (Q[2:] - Q[:-2]) / (2 * dt)
    return -(G * Qd).sum(1)


# ---- kinematic fit ------------------------------------------------------------------
def kinfit(data, x0=None, prior=None, dt=None):
    """data: ループのリスト（gyro, enc）。prior: (mean(4), cov(4x4)) で p と b に事前情報を与える（逐次更新用）
    戻り値: dict(p, b, cov（p, b の 4x4）, chi2, ndf, pulls_q, pulls_w)"""
    dt = 1 / F.FS if dt is None else dt
    ns = [len(d["enc"]) for d in data]
    nQ = sum(ns) * 2
    offs = np.cumsum([0] + [n * 2 for n in ns])
    x0 = np.r_[F.NOMINAL, 0.0] if x0 is None else x0
    X0 = np.r_[x0, np.concatenate([d["enc"].ravel() for d in data])]
    if prior is not None:
        Lp = np.linalg.cholesky(np.linalg.inv(prior[1]))   # (p−μ)ᵀ C⁻¹ (p−μ) = |Lpᵀ (p−μ)|²

    def resid(X):
        p, b = X[:3], X[3]
        out = []
        for k, d in enumerate(data):
            Q = X[4 + offs[k]: 4 + offs[k + 1]].reshape(-1, 2)
            out.append(((Q - d["enc"]) / SQ).ravel())
            out.append((d["gyro"][1:-1] - b - model_rate(Q, p, dt)) / SW)
        if prior is not None:
            out.append(Lp.T @ (X[:4] - prior[0]))
        return np.concatenate(out)

    # ヤコビアンの疎な構造: 関節角の残差は自分の Q だけ、ω の残差は前後の Q と p, b
    rows, cols = [], []
    r0 = 0
    for k, n in enumerate(ns):
        base = 4 + offs[k]
        for j in range(2 * n):
            rows.append(r0 + j); cols.append(base + j)
        r0 += 2 * n
        for i in range(1, n - 1):
            for jj in range(4):
                rows.append(r0 + i - 1); cols.append(jj)
            for s in (-1, 0, 1):
                for c2 in range(2):
                    rows.append(r0 + i - 1); cols.append(base + 2 * (i + s) + c2)
        r0 += n - 2
    if prior is not None:
        for i in range(4):
            for jj in range(4):
                rows.append(r0 + i); cols.append(jj)
        r0 += 4
    S = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(r0, 4 + nQ)).tocsr()
    lo = np.r_[[0.5, 1.0, 0.3, -0.05], np.full(nQ, -np.inf)]
    hi = np.r_[[15, 40, 3.0, 0.05], np.full(nQ, np.inf)]
    sol = least_squares(resid, X0, jac_sparsity=S, bounds=(lo, hi), x_scale=np.r_[[1, 1, 0.1, 0.01], np.full(nQ, SQ)],
                        method="trf", tr_solver="lsmr", xtol=1e-12, ftol=1e-12, max_nfev=2000)
    J = sparse.csc_matrix(sol.jac)
    H = (J.T @ J).tocsc()
    lu = splu(H)
    Einv = lu.solve(np.eye(H.shape[0])[:, :4])
    cov = Einv[:4]
    cov = (cov + cov.T) / 2
    chi2 = float(2 * sol.cost)
    ndf = r0 - (4 + nQ)
    return dict(p=sol.x[:3], b=float(sol.x[3]), x=sol.x[:4], cov=cov, chi2=chi2, ndf=int(ndf), nfev=sol.nfev)


def sequential(data):
    """ループを 1 本ずつ加えて逐次更新する。最初のループは事前情報なし"""
    out = []; prior = None; x0 = None
    for k in range(len(data)):
        f = kinfit([data[k]], x0=x0, prior=prior)
        prior = (f["x"], f["cov"]); x0 = f["x"]
        out.append(f)
    return out


# ---- 試行 ---------------------------------------------------------------------------
def job(args):
    kind, i = args
    rng = np.random.default_rng([4242 if kind == "rk4" else 4343, i])
    p = F.NOMINAL * np.array([rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15), rng.uniform(0.9, 1.1)])
    raw = E.simulate(p, integ="RK4" if kind == "rk4" else "implicitfast")
    bias = rng.uniform(-F.BIAS_MAX, F.BIAS_MAX)
    data = E.sensors(raw, rng, bias=bias)
    t0 = time.time()
    f = kinfit(data)
    t_fit = time.time() - t0
    bp = F.fit_pathintegral(data)[:3]
    res = dict(kind=kind, i=i, p_true=p.tolist(), bias_true=bias, p_kf=f["p"].tolist(), b_kf=f["b"], cov=f["cov"].tolist(),
               chi2=f["chi2"], ndf=f["ndf"], t_fit=t_fit, p_bp=bp.tolist())
    if kind == "rk4" and i < 10:
        seq = sequential(data)
        res["seq"] = [dict(p=s["p"].tolist(), sd=np.sqrt(np.diag(s["cov"])[:3]).tolist(), chi2=s["chi2"], ndf=s["ndf"]) for s in seq]
    return res


def main():
    t0 = time.time()
    jobs = [("rk4", i) for i in range(N_TRIAL)] + [("impl", i) for i in range(N_IMPL)]
    with Pool(4) as pool:
        res = pool.map(job, jobs, chunksize=1)
    print(f"{len(jobs)} 件（{time.time()-t0:.0f} s）", flush=True)
    names = ["腕", "脚", "慣性"]
    lines = ["=== kinematic fit（拘束条件つきの当てはめ）による校正の自己評価 ==="]
    summ = {}
    for kind, lab in (("rk4", "仮想の実機 RK4"), ("impl", "仮想の実機 implicitfast（積分法の系統誤差あり）")):
        rs = [r for r in res if r["kind"] == kind]
        pt = np.array([r["p_true"] for r in rs]); pk = np.array([r["p_kf"] for r in rs]); pb = np.array([r["p_bp"] for r in rs])
        sd = np.array([np.sqrt(np.diag(r["cov"])[:3]) for r in rs])
        pull = (pk - pt) / sd
        ek = (pk - pt) / pt * 100; eb = (pb - pt) / pt * 100
        chi = np.array([r["chi2"] / r["ndf"] for r in rs]); pv = np.array([chi2dist.sf(r["chi2"], r["ndf"]) for r in rs])
        cover = (np.abs(pull) < 1).mean(0)
        rel_sd = sd / pt * 100
        s = dict(n=len(rs), err_kf_mean=ek.mean(0).tolist(), err_kf_sd=ek.std(0, ddof=1).tolist(), err_bp_mean=eb.mean(0).tolist(),
                 err_bp_sd=eb.std(0, ddof=1).tolist(), pull_mean=pull.mean(0).tolist(), pull_sd=pull.std(0, ddof=1).tolist(),
                 cover68=cover.tolist(), sigma_rel_mean=rel_sd.mean(0).tolist(), chi2ndf_mean=float(chi.mean()), chi2ndf_sd=float(chi.std(ddof=1)),
                 pvalue_mean=float(pv.mean()), t_fit_mean=float(np.mean([r["t_fit"] for r in rs])),
                 corr=np.mean([np.array(r["cov"])[:3, :3] / np.outer(np.sqrt(np.diag(r["cov"])[:3]), np.sqrt(np.diag(r["cov"])[:3])) for r in rs], 0).tolist())
        summ[kind] = s
        f2 = lambda v, d=2: " / ".join(f"{x:+.{d}f}" for x in v)
        lines += [f"[{lab}] {len(rs)} 回",
                  f"  誤差 [%]（腕 / 脚 / 慣性） kinematic fit: 平均 {f2(s['err_kf_mean'])}, 標準偏差 {f2(s['err_kf_sd'])}",
                  f"                             (b') 微分なし: 平均 {f2(s['err_bp_mean'])}, 標準偏差 {f2(s['err_bp_sd'])}",
                  f"  推定した誤差 σ の平均 [%]: {f2(s['sigma_rel_mean'])}",
                  f"  pull 平均 {f2(s['pull_mean'])}, pull 標準偏差 {f2(s['pull_sd'])}, |pull|<1 の割合 {f2(s['cover68'])}",
                  f"  χ²/ndf 平均 {s['chi2ndf_mean']:.4f} ± {s['chi2ndf_sd']:.4f}, p 値の平均 {s['pvalue_mean']:.3f}, 当てはめ 1 回 {s['t_fit_mean']:.0f} s",
                  f"  相関係数（腕-脚, 腕-慣性, 脚-慣性）: {s['corr'][0][1]:+.3f}, {s['corr'][0][2]:+.3f}, {s['corr'][1][2]:+.3f}"]
    seqs = [r for r in res if "seq" in r]
    sd_seq = np.array([[np.array(s["sd"]) / np.array(r["p_true"]) * 100 for s in r["seq"]] for r in seqs])   # (試行, ループ, 3)
    err_seq = np.array([[(np.array(s["p"]) - np.array(r["p_true"])) / np.array(r["p_true"]) * 100 for s in r["seq"]] for r in seqs])
    diff_batch = np.array([(np.array(r["seq"][-1]["p"]) - np.array(r["p_kf"])) / np.array(r["p_true"]) * 100 for r in seqs])
    lines += ["[逐次更新] ループを 1 本ずつ加えたときの推定した誤差 σ の平均 [%]（腕 / 脚 / 慣性）"]
    for k in range(sd_seq.shape[1]):
        lines.append(f"  ループ {k+1}: σ {' / '.join(f'{x:.2f}' for x in sd_seq[:, k].mean(0))}, 実際の誤差の二乗平均 {' / '.join(f'{x:.2f}' for x in np.sqrt((err_seq[:, k]**2).mean(0)))}")
    lines.append(f"  逐次更新の最終値と一括の当てはめの差の最大 [%]: {np.abs(diff_batch).max():.3f}")
    lines.append(f"計算時間 {time.time()-t0:.0f} s")
    txt = "\n".join(lines); print(txt)
    (DATA / "field04c_summary.txt").write_text(txt + "\n")
    json.dump(dict(summary=summ, trials=res), open(DATA / "field04c_kinfit.json", "w"), indent=1, ensure_ascii=False)
    plot(res, summ, sd_seq, err_seq)


def plot(res, summ, sd_seq, err_seq):
    names = ["arm mass", "leg mass", "torso inertia"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    bins = np.linspace(-5, 5, 26); xx = np.linspace(-5, 5, 200)
    for kind, col, lab in (("rk4", "C0", "RK4"), ("impl", "C3", "implicitfast")):
        rs = [r for r in res if r["kind"] == kind]
        pull = np.array([(np.array(r["p_kf"]) - np.array(r["p_true"])) / np.sqrt(np.diag(r["cov"])[:3]) for r in rs])
        ax[0].hist(pull.ravel(), bins=bins, density=True, histtype="step", lw=1.6, color=col,
                   label=f"{lab}: mean {pull.mean():+.2f}, SD {pull.std(ddof=1):.2f}")
    ax[0].plot(xx, np.exp(-xx**2 / 2) / np.sqrt(2 * np.pi), "k:", label="N(0, 1)")
    ax[0].set_xlabel("pull = (estimate − truth) / estimated σ"); ax[0].set_ylabel("density")
    ax[0].set_title("pulls of the kinematic fit (3 parameters pooled)"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    for kind, col, lab in (("rk4", "C0", "RK4"), ("impl", "C3", "implicitfast")):
        rs = [r for r in res if r["kind"] == kind]
        pv = [chi2dist.sf(r["chi2"], r["ndf"]) for r in rs]
        ax[1].hist(pv, bins=np.linspace(0, 1, 11), histtype="step", lw=1.6, color=col, density=True, label=lab)
    ax[1].axhline(1, color="k", ls=":", lw=1)
    ax[1].set_xlabel("χ² p-value"); ax[1].set_ylabel("density"); ax[1].set_title("goodness of fit"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    loops = np.arange(1, sd_seq.shape[1] + 1)
    for k in range(3):
        ax[2].plot(loops, sd_seq[:, :, k].mean(0), "o-", color=f"C{k}", label=f"{names[k]}: estimated σ")
        ax[2].plot(loops, np.sqrt((err_seq[:, :, k]**2).mean(0)), "x--", color=f"C{k}", label=f"{names[k]}: actual RMS error")
    ax[2].set_yscale("log"); ax[2].set_xlabel("number of loops (sequential update)"); ax[2].set_ylabel("[%]")
    ax[2].set_title(f"sequential update ({sd_seq.shape[0]} trials): estimated σ vs actual error"); ax[2].legend(fontsize=6, ncol=2); ax[2].grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(FIG / "field" / "field04c_kinfit.png", dpi=150)


if __name__ == "__main__":
    main()
