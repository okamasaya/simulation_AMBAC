"""Sim05B: スラスタ・RW・AMBAC の統合姿勢制御 — 最適制御 vs 学習（18 m 級）.

Part B: 最適制御（線形計画 LP）による配分。90° 静止→静止旋回の 時間–推進剤 パレート曲線
        構成: (1) スラスタのみ (2) +RW (3) +AMBAC (4) +AMBAC+RW
Part C: モデル誤差・外乱がある場合の比較（スラスタ + AMBAC, T = 60 s）
        (a) 最適制御・開ループ（公称モデルで一度だけ計画）
        (b) MPC（2 s ごとに LP を再計画）
        (b') MPC + 揺れ補償（計画から予測した終端位相の揺れを目標から差し引く、終盤 10 s は AMBAC 停止）
        (c) 学習方策（ES、ドメインランダム化で学習）
        評価用プラントは「周回内の揺れ」を含む位相モデル（制御器側は平均化モデル）

モデル（L: 全角運動量, h: RW の角運動量, Ā: 周回平均慣性, Δθc: 1 周あたり回転）
  平均化モデル : θ̄' = (L − h)/Ā + f Δθc,  L' = τ_th + τ_d,  h' = τ_rw
  位相モデル   : θ'  = (L − h)/A(φ) − g(φ) φ',  φ' = 2π f     （g = Γ·dq/dφ, ループ上で厳密）
"""
import json
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

from common import DATA, FIG, human_like
from ambac.es import MLP, OpenAIES

D = np.degrees
G0 = 9.80665
S = 10.0                                         # 18 m 級
TARGET = np.pi / 2
TAU_D_MAX = 500.0                                # Part C の一定外乱の範囲 ±[N m]

# ---- パラメータ（Sim03/04 から） ----------------------------------------------
lj = json.load(open(DATA / "sim03_optimal_loop.json"))
C1, C2, R, DTH = (lj[k] for k in ("center_arm_rad", "center_leg_rad", "radius_rad", "dtheta_per_cycle_rad"))
scal = np.genfromtxt(DATA / "sim04_scaling.csv", delimiter=",", names=True)
F_MAX = float(scal["f_maxHz"][scal["scale"] == S][0])           # 0.102 Hz
A_BAR = S**5 / lj["inv_A_mean"]                                  # 周回平均慣性 [kg m²]
TAU_TH = 2.5e5                                                   # スラスタ最大トルク [N m]
R_TH, ISP = 5.0, 300.0                                           # モーメントアーム [m], 比推力 [s]
KG_PER_NMS = 1.0 / (R_TH * ISP * G0)                             # 角力積あたりの推進剤 [kg/(N m s)]
H_RW = 8.64 * S**4                                               # RW 容量（Sim03 の 2 kg RW を相似拡大）
TAU_RW = 0.2 * S**3                                              # RW トルク（同上, 200 N m）

print(f"18 m 級: Ā = {A_BAR:.3g} kg m², Δθc = {D(DTH):.2f}°/周, f_max = {F_MAX:.3f} Hz")
print(f"  AMBAC 単独の 90° 旋回時間 ≈ {TARGET / DTH / F_MAX:.0f} s, RW: h_max {H_RW:.3g}, τ_max {TAU_RW:.0f} N m")
print(f"  スラスタ: τ_max {TAU_TH:.2g} N m, 推進剤 {KG_PER_NMS * 1e3:.3f} g/(N m s)")


# =============================================================================
# 最適制御: 線形計画
# =============================================================================
def plan_lp(T, th0=0.0, L0=0.0, h0=0.0, use_rw=False, use_amb=False, N=None, dth=DTH, abar=A_BAR,
            tau_d=0.0, soft=True, target=TARGET, amb_until=None):
    """推進剤最小の計画. 戻り値: dict(dt, tau, f, w, prop, ok)."""
    N = N or max(10, min(200, int(np.ceil(T / 0.5))))
    dt = T / N
    # 変数: [tp(N), tm(N), fp(N), fm(N), wp(N), wm(N), s+(1), s-(1), l+(1), l-(1)]
    nv = 6 * N + 4
    c = np.zeros(nv)
    c[0:2 * N] = KG_PER_NMS * dt                 # 推進剤 [kg]
    c[2 * N:4 * N] = 1e-6                        # 不要な AMBAC 動作の抑制
    c[4 * N:6 * N] = 1e-9
    c[6 * N:6 * N + 2] = 1e4                     # 終端姿勢スラック [kg/rad]
    c[6 * N + 2:6 * N + 4] = 1e-1                # 終端角運動量スラック
    ub = np.zeros(nv)
    ub[0:2 * N] = TAU_TH
    ub[2 * N:4 * N] = F_MAX if use_amb else 0.0
    if use_amb and amb_until is not None:          # 指定時刻以降は AMBAC を停止
        tk_ = np.arange(N) * dt
        ub[2 * N:3 * N][tk_ >= amb_until] = 0.0; ub[3 * N:4 * N][tk_ >= amb_until] = 0.0
    ub[4 * N:6 * N] = TAU_RW if use_rw else 0.0
    ub[6 * N:] = np.inf if soft else 0.0
    # L_k = L0 + τ_d k dt + Σ_{j<k} (tp - tm)_j dt,  h_k = h0 + Σ_{j<k} (wp - wm)_j dt
    low = np.tril(np.ones((N, N)), -1) * dt      # k 行: j<k
    # θ_N = th0 + Σ_k [ (L_k - h_k) dt/Ā + (τ_k - w_k) dt²/(2Ā) + f_k Δθ dt ]
    coefL = (low.sum(0) * dt + dt**2 / 2) / abar  # 各 j の (tp-tm)_j が θ_N に与える係数
    A_eq = np.zeros((3, nv)); b_eq = np.zeros(3)
    A_eq[0, 0:N] = coefL; A_eq[0, N:2 * N] = -coefL
    A_eq[0, 4 * N:5 * N] = -coefL; A_eq[0, 5 * N:6 * N] = coefL
    A_eq[0, 2 * N:3 * N] = dth * dt; A_eq[0, 3 * N:4 * N] = -dth * dt
    A_eq[0, 6 * N] = -1; A_eq[0, 6 * N + 1] = 1
    tk = np.arange(N) * dt
    b_eq[0] = target - th0 - (L0 - h0) * T / abar - tau_d * np.sum(tk * dt + dt**2 / 2) / abar
    A_eq[1, 0:N] = dt; A_eq[1, N:2 * N] = -dt; A_eq[1, 6 * N + 2] = -1; A_eq[1, 6 * N + 3] = 1
    b_eq[1] = -L0 - tau_d * T                    # L_N = 0
    A_eq[2, 4 * N:5 * N] = dt; A_eq[2, 5 * N:6 * N] = -dt
    b_eq[2] = -h0                                # h_N = 0
    # |h_k| ≤ H_RW
    up = np.tril(np.ones((N, N))) * dt
    A_ub = np.zeros((2 * N, nv)); b_ub = np.zeros(2 * N)
    A_ub[:N, 4 * N:5 * N] = up; A_ub[:N, 5 * N:6 * N] = -up; b_ub[:N] = H_RW - h0
    A_ub[N:, 4 * N:5 * N] = -up; A_ub[N:, 5 * N:6 * N] = up; b_ub[N:] = H_RW + h0
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=list(zip(np.zeros(nv), ub)),
                method="highs")
    if r.status != 0:
        return dict(ok=False)
    x = r.x
    tau = x[0:N] - x[N:2 * N]; f = x[2 * N:3 * N] - x[3 * N:4 * N]; w = x[4 * N:5 * N] - x[5 * N:6 * N]
    prop = KG_PER_NMS * dt * np.abs(tau).sum()
    ok = x[6 * N:].max() < 1e-6
    return dict(ok=ok, dt=dt, tau=tau, f=f, w=w, prop=prop, slack=x[6 * N:])


# =============================================================================
# Part B: パレート曲線
# =============================================================================
T_LIST = np.geomspace(5.2, 400, 90)
CFGS = {"thrusters": (False, False), "thr+RW": (True, False), "thr+AMBAC": (False, True),
        "thr+AMBAC+RW": (True, True)}
pareto = {k: [] for k in CFGS}
for T in T_LIST:
    for k, (rw, amb) in CFGS.items():
        p = plan_lp(T, use_rw=rw, use_amb=amb, soft=False)
        pareto[k].append(p["prop"] if p.get("ok") else np.nan)
pareto = {k: np.array(v) for k, v in pareto.items()}
np.savetxt(DATA / "sim05b_pareto.csv", np.c_[T_LIST, *[pareto[k] for k in CFGS]], delimiter=",",
           comments="", header="T[s]," + ",".join(f"propellant_{k}[kg]" for k in CFGS))
print("\n=== Part B: 90° 旋回の最小推進剤 [kg] ===")
print("   T[s] " + "".join(f"{k:>15s}" for k in CFGS))
for i in range(0, len(T_LIST), 9):
    print(f"{T_LIST[i]:7.1f} " + "".join(f"{pareto[k][i]:15.2f}" for k in CFGS))
for k in CFGS:
    z = T_LIST[pareto[k] < 1e-3]
    print(f"  {k:14s}: 推進剤ゼロで旋回できる最短時間 ≈ {z.min():.0f} s（格子上）" if len(z) else f"  {k:14s}: 推進剤ゼロ不可")

# 代表例 T=60 s の計画
T_EX = 60.0
ex = {k: plan_lp(T_EX, use_rw=rw, use_amb=amb, soft=False) for k, (rw, amb) in CFGS.items()}

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for k in CFGS:
    ax[0].loglog(T_LIST, np.maximum(pareto[k], 1e-2), "-", label=k)
ax[0].axvspan(5.2, 3 / F_MAX, color="gray", alpha=.12)
ax[0].text(6, 2e-2, "AMBAC averaged model\nless reliable (<3 cycles)", fontsize=7)
ax[0].set_xlabel("maneuver time T [s]"); ax[0].set_ylabel("propellant per 90° turn [kg]  (floor 0.01)")
ax[0].set_title("Part B: time–propellant Pareto front (18 m class)")
ax[0].grid(alpha=.3, which="both"); ax[0].legend(fontsize=8)
p = ex["thr+AMBAC+RW"]; t = np.arange(len(p["tau"])) * p["dt"]
ax[1].step(t, p["tau"] / TAU_TH, where="post", label="thruster τ / τmax")
ax[1].step(t, p["f"] / F_MAX, where="post", label="AMBAC f / fmax")
ax[1].step(t, p["w"] / TAU_RW, where="post", label="RW τ / τmax")
ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("normalized command")
ax[1].set_title(f"optimal allocation, thr+AMBAC+RW, T = {T_EX:.0f} s ({p['prop']:.2f} kg)")
ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "sim05b_pareto.png", dpi=150)

# =============================================================================
# Part C: モデル誤差・外乱下での比較（スラスタ + AMBAC）
# =============================================================================
robot = human_like(arm_mass=4.0, leg_mass=12.0)
PH = np.linspace(0, 2 * np.pi, 721)
LQ = np.c_[C1 + R * np.cos(PH), C2 + R * np.sin(PH)]
LD = np.c_[-R * np.sin(PH), R * np.cos(PH)]
G_PH = np.array([robot.connection(q) @ d for q, d in zip(LQ, LD)])       # Γ·dq/dφ（スケール不変）
INVA_PH = np.array([1.0 / robot.coupling(q)[0] for q in LQ]) / S**5     # 1/A(φ)
dth_true_nom = -np.trapezoid(G_PH, PH)
geo = np.concatenate([[0], np.cumsum(-(G_PH[1:] + G_PH[:-1]) / 2 * np.diff(PH))])
OSC = geo - PH / (2 * np.pi) * dth_true_nom; OSC -= OSC.mean()

DT_SIM = 0.1
T_C = 60.0
NS = int(T_C / DT_SIM)


def interp_ph(tab, phi):
    return np.interp(np.mod(phi, 2 * np.pi), PH, tab)


def sample_plants(n, rng, nominal=False):
    """真のプラント: Δθc の倍率 κ（ループ形状の誤差に相当）, 慣性倍率 α, 一定外乱 τ_d."""
    if nominal:
        return dict(kap=np.ones(n), alp=np.ones(n), tau_d=np.zeros(n))
    return dict(kap=rng.uniform(0.7, 1.3, n), alp=rng.uniform(0.9, 1.1, n), tau_d=rng.uniform(-TAU_D_MAX, TAU_D_MAX, n))


def plant_step(st, tau, f, pl):
    """位相モデル（揺れを含む）を 1 ステップ進める. st: dict of arrays."""
    phid = 2 * np.pi * f
    thd = st["L"] * interp_ph(INVA_PH, st["phi"]) / pl["alp"] - pl["kap"] * interp_ph(G_PH, st["phi"]) * phid
    st["th"] = st["th"] + DT_SIM * thd
    st["phi"] = st["phi"] + DT_SIM * phid
    st["L"] = st["L"] + DT_SIM * (tau + pl["tau_d"])
    st["prop"] = st["prop"] + KG_PER_NMS * DT_SIM * np.abs(tau)
    st["cyc"] = st["cyc"] + DT_SIM * np.abs(f)          # AMBAC の周回数
    return st


def init_state(n):
    z = np.zeros(n)
    return dict(th=z.copy() + interp_ph(OSC, 0.0), phi=z.copy(), L=z.copy(), prop=z.copy(), cyc=z.copy())


def theta_bar(st):
    return st["th"] - interp_ph(OSC, st["phi"])


def final_metrics(st, pl):
    th_err = D(np.abs(st["th"] - TARGET))
    rate = D(np.abs(st["L"] * interp_ph(INVA_PH, st["phi"]) / pl["alp"]))
    return th_err, rate, st["prop"]


# 制御器: reset(n) → act(st, t) が (τ_th, f) を返す ---------------------------------
REPLAN = 2.0
T_FREEZE = 10.0


class OpenLoopOC:
    name = "open-loop OC"

    def reset(self, n):
        self.p = plan_lp(T_C, use_amb=True, soft=True)

    def act(self, st, t):
        j = min(int(t / self.p["dt"]), len(self.p["tau"]) - 1)
        return np.full(len(st["th"]), self.p["tau"][j]), np.full(len(st["th"]), self.p["f"][j])


class MPC:
    """各プラント個別に LP を再計画（状態観測: θ̄, L, φ）."""

    def __init__(self, wobble_comp=False):
        self.comp = wobble_comp
        self.name = "MPC + wobble comp." if wobble_comp else "MPC"

    def reset(self, n):
        self.p = [None] * n; self.t0 = 0.0

    def act(self, st, t):
        n = len(st["th"])
        if round(t / DT_SIM) % round(REPLAN / DT_SIM) == 0:
            Trem = T_C - t; self.t0 = t
            for m in range(n):
                kw = dict(th0=float(theta_bar(st)[m]), L0=float(st["L"][m]), use_amb=True, soft=True,
                          N=max(4, int(np.ceil(Trem / 0.5))))
                if self.comp:
                    # 旋回終了時の位相 φ_end を計画から予測し、その位相での揺れを目標から差し引く（反復）
                    kw["amb_until"] = Trem - T_FREEZE
                    phi_end = float(st["phi"][m])
                    for _ in range(3):
                        kw["target"] = TARGET - float(interp_ph(OSC, phi_end))
                        p = plan_lp(Trem, **kw)
                        phi_end = float(st["phi"][m]) + 2 * np.pi * p["dt"] * p["f"].sum()
                    self.p[m] = p
                else:
                    self.p[m] = plan_lp(Trem, **kw)
        j = [min(int((t - self.t0) / p["dt"]), len(p["tau"]) - 1) for p in self.p]
        return (np.array([p["tau"][k] for p, k in zip(self.p, j)]),
                np.array([p["f"][k] for p, k in zip(self.p, j)]))


def simulate(ctrl, pl, record=False):
    n = len(pl["kap"]); st = init_state(n); ctrl.reset(n); hist = []
    for i in range(NS):
        tau, f = ctrl.act(st, i * DT_SIM)
        st = plant_step(st, tau, f, pl)
        if record:
            hist.append((st["th"].copy(), tau.copy(), f.copy()))
    return st, hist


# (c) 学習方策（ES + ドメインランダム化）--------------------------------------------
NET = MLP([6, 32, 32, 2])
L_REF = TAU_TH * 5.0


def policy_obs(st, t):
    e = TARGET - theta_bar(st)
    return np.stack([e / (np.pi / 2), st["L"] / L_REF, np.full_like(e, 1 - t / T_C),
                     np.sin(st["phi"]), np.cos(st["phi"]), np.full_like(e, t / T_C)], -1)


def run_policy(P, pl, n_pop=1):
    """P: (pop, n_params), pl: プラント (batch,) → 各 pop×batch の状態"""
    nb = len(pl["kap"])
    st = init_state(n_pop * nb)
    plb = {k: np.tile(v, n_pop) for k, v in pl.items()}
    for i in range(NS):
        o = policy_obs(st, i * DT_SIM).reshape(n_pop, nb, -1)
        a = NET(P, o).reshape(n_pop * nb, 2)
        st = plant_step(st, a[:, 0] * TAU_TH, a[:, 1] * F_MAX, plb)
    return st, plb


class Learned:
    name = "learned (ES)"

    def __init__(self, x):
        self.x = x

    def reset(self, n):
        pass

    def act(self, st, t):
        a = NET(self.x[None], policy_obs(st, t)[None])[0]
        return a[:, 0] * TAU_TH, a[:, 1] * F_MAX


def reward(st, pl):
    th_err, rate, prop = final_metrics(st, pl)
    return -(prop / 2.0 + th_err / 1.0 + rate / 0.05)        # 2 kg ≈ 1° ≈ 0.05°/s


rng = np.random.default_rng(0)
es = OpenAIES(NET.init(rng, scale=0.3), sigma=0.05, lr=0.02, n_pairs=32, seed=2)
N_IT, NB = 800, 24
log = []; t0 = time.time()
POLICY_FILE = DATA / "sim05b_policy.npy"
if POLICY_FILE.exists() and not os.environ.get("RETRAIN"):
    es.x = np.load(POLICY_FILE)                  # 学習済み方策を再利用（再学習は RETRAIN=1）
    N_IT = 0
    print("\n=== Part C: 学習済み方策を読み込み（再学習するには RETRAIN=1）===")
else:
    print("\n=== Part C: 学習（約 12 分）===")
for it in range(N_IT):
    pl = sample_plants(NB, rng)
    P = es.ask()
    st, plb = run_policy(P, pl, len(P))
    fit = reward(st, plb).reshape(len(P), NB).mean(1)
    es.tell(fit)
    if it % 100 == 0 or it == N_IT - 1:
        st1, plb1 = run_policy(es.x[None], pl)
        e, r_, pr = final_metrics(st1, plb1)
        log.append((it, reward(st1, plb1).mean(), e.mean(), r_.mean(), pr.mean()))
        print(f"iter {it:4d}  reward {log[-1][1]:+8.2f}  姿勢誤差 {e.mean():6.2f}°  残留角速度 {r_.mean():.3f}°/s"
              f"  推進剤 {pr.mean():6.2f} kg  ({time.time() - t0:.0f} s)", flush=True)
if N_IT:
    np.savetxt(DATA / "sim05b_training_log.csv", np.array(log), delimiter=",", comments="",
               header="iter,reward,err_deg,rate_deg_s,propellant_kg")
if N_IT:
    np.save(POLICY_FILE, es.x)

# ---- 評価 -------------------------------------------------------------------
rng_test = np.random.default_rng(123)
tests = {"nominal": sample_plants(1, rng_test, nominal=True), "randomized(100)": sample_plants(100, rng_test)}
CTRLS = [OpenLoopOC(), MPC(False), MPC(True), Learned(es.x)]
p_thr = plan_lp(T_C, soft=False)["prop"]; p_amb = plan_lp(T_C, use_amb=True, soft=False)["prop"]
print(f"\n参考（公称・平均化モデルの最適値, T = {T_C:.0f} s）: スラスタのみ {p_thr:.2f} kg, スラスタ+AMBAC {p_amb:.2f} kg")
rows = []
print("\n=== Part C: 評価（T = 60 s, 指標: 終端姿勢誤差 / 残留角速度 / 推進剤）===")
for tname, pl in tests.items():
    for ctrl in CTRLS:
        t0 = time.time()
        st, _ = simulate(ctrl, pl)
        e, r_, pr = final_metrics(st, pl)
        rows.append((tname, ctrl.name, e.mean(), np.percentile(e, 95), r_.mean(), pr.mean(), st["cyc"].mean()))
        print(f"[{tname:15s}] {ctrl.name:19s}: 誤差 平均 {e.mean():6.2f}° (95% {np.percentile(e, 95):6.2f}°)"
              f"  残留 {r_.mean():.4f}°/s  推進剤 {pr.mean():6.2f} kg  AMBAC {st['cyc'].mean():4.1f} 周"
              f"  ({time.time() - t0:.1f} s)", flush=True)
with open(DATA / "sim05b_eval.csv", "w") as fh:
    fh.write("test,controller,err_mean_deg,err_p95_deg,rate_mean_deg_s,propellant_kg,ambac_cycles\n")
    for r in rows:
        fh.write(",".join([r[0], r[1]] + [f"{x:.4f}" for x in r[2:]]) + "\n")

# 代表例の時系列（ランダムプラントの 1 つ）
pl1 = sample_plants(1, np.random.default_rng(7))
tt = (np.arange(NS) + 1) * DT_SIM
fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
for k, ctrl in enumerate(CTRLS):
    _, h = simulate(ctrl, pl1, record=True)
    h = np.array([[x[0][0], x[1][0], x[2][0]] for x in h])
    ax[0].plot(tt, D(h[:, 0]), f"C{k}", label=ctrl.name, lw=1)
    ax[1].plot(tt, h[:, 1] / TAU_TH, f"C{k}", lw=1)
    ax[2].plot(tt, h[:, 2] / F_MAX, f"C{k}", lw=1)
ax[0].axhline(90, color="k", ls=":"); ax[0].set_ylabel("θ [deg]"); ax[0].legend(fontsize=8)
ax[0].set_title(f"Part C example plant: κ={pl1['kap'][0]:.2f}, α={pl1['alp'][0]:.2f}, τd={pl1['tau_d'][0]:.0f} N m")
ax[1].set_ylabel("thruster τ / τmax"); ax[2].set_ylabel("AMBAC f / fmax"); ax[2].set_xlabel("time [s]")
for a in ax: a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "sim05b_robustness.png", dpi=150)
