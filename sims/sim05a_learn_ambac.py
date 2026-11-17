"""Sim05A: 学習エージェントは AMBAC（幾何学的位相）を自力で発見できるか.

設定（人間大、L = 0、運動学）:
  観測  : 目標との姿勢誤差 e（sin, cos）, 関節角 q（可動域で正規化）
  行動  : 関節角速度 q' ∈ [-QD_MAX, QD_MAX]²（関節可動域で停止）
  報酬  : −(姿勢誤差 |e| の時間平均) − 終端誤差 − 小さな行動ペナルティ
  物理の知識（接続・曲率・ループ）は一切与えない。

  3 条件を比較する（すべて進化戦略 ES で学習）:
    plain : 観測 → MLP → 関節角速度。構造なし
    cpg   : plain の観測に周期信号 (sin, cos)(2π f t) を追加
    tg    : 軌道生成器（Trajectory Generator）を学習で調整する構造（PMTG, Iscen+ 2018 と同系統）
            内部位相 ψ を持つ振動子が目標関節角 q* = c + A⊙(cos ψ, sin(ψ+δ)) を生成し、
            学習するのは ループ中心 c・振幅 A・形 δ と、
            姿勢誤差 → 位相速度 ψ' = 2π f u(e) のフィードバック則 u (MLP)
            ※ ループの位置・大きさ・回す向きは与えない（初期値は可動域中央の小さな円）
"""
import json
import time

import numpy as np
import matplotlib.pyplot as plt

from common import DATA, FIG, human_like
from ambac.es import MLP, OpenAIES

D = np.degrees
robot = human_like(arm_mass=4.0, leg_mass=12.0)
loopj = json.load(open(DATA / "sim03_optimal_loop.json"))
C1, C2, R, DTH_C = (loopj[k] for k in ("center_arm_rad", "center_leg_rad", "radius_rad", "dtheta_per_cycle_rad"))
F_MAX = 1.02                                # Sim04: 人間の持続トルクでの周回周波数上限
QD_MAX = 2 * np.pi * F_MAX * R              # 最適ループを f_max で回すときの関節速度 [rad/s]
Q_LO = np.radians([-150.0, -120.0]); Q_HI = np.radians([150.0, 30.0])
Q0 = np.array([C1 - R, C2])                 # 初期姿勢: 最適ループ上の点（腕 -90°, 脚 -40°）
DT, T_EP = 0.04, 25.0
NSTEP = int(T_EP / DT)

# ---- Γ(q) を格子上に事前計算（双一次補間で高速化） ------------------------------
NG1, NG2 = 121, 61
g1 = np.linspace(Q_LO[0], Q_HI[0], NG1); g2 = np.linspace(Q_LO[1], Q_HI[1], NG2)
GAM = np.array([[robot.connection([a, b]) for b in g2] for a in g1])   # (NG1, NG2, 2)


def gamma(q):
    """q: (..., 2) → Γ: (..., 2)"""
    x = (q[..., 0] - g1[0]) / (g1[1] - g1[0]); y = (q[..., 1] - g2[0]) / (g2[1] - g2[0])
    i = np.clip(x.astype(int), 0, NG1 - 2); j = np.clip(y.astype(int), 0, NG2 - 2)
    fx = (x - i)[..., None]; fy = (y - j)[..., None]
    return (GAM[i, j] * (1 - fx) * (1 - fy) + GAM[i + 1, j] * fx * (1 - fy)
            + GAM[i, j + 1] * (1 - fx) * fy + GAM[i + 1, j + 1] * fx * fy)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def step(q, th, qd):
    """RK2（中点法）で 1 ステップ. 関節可動域で停止."""
    qm = np.clip(q + 0.5 * DT * qd, Q_LO, Q_HI)
    qn = np.clip(q + DT * qd, Q_LO, Q_HI)
    qd_eff = (qn - q) / DT
    th = th - DT * np.sum(gamma(qm) * qd_eff, -1)
    return qn, th


F_CPG = F_MAX                                # CPG の周波数 [Hz]


def obs(q, th, target, k, cpg):
    e = wrap(target - th)
    qn = 2 * (q - Q_LO) / (Q_HI - Q_LO) - 1
    o = [np.sin(e)[..., None], np.cos(e)[..., None], qn]
    if cpg:
        w = 2 * np.pi * F_CPG * k * DT
        o += [np.full(e.shape + (1,), np.sin(w)), np.full(e.shape + (1,), np.cos(w))]
    return np.concatenate(o, -1)


def rollout(policy_fn, targets, cpg, record=False):
    """targets: (pop, batch). 戻り値: 報酬 (pop,)"""
    shp = targets.shape
    q = np.broadcast_to(Q0, (*shp, 2)).copy(); th = np.zeros(shp)
    cost = np.zeros(shp); act = np.zeros(shp)
    hist = []
    for k in range(NSTEP):
        a = policy_fn(obs(q, th, targets, k, cpg), k)
        q, th = step(q, th, a * QD_MAX)
        cost += np.abs(wrap(targets - th)) * DT / T_EP
        act += np.sum(a**2, -1) * DT / T_EP
        if record:
            hist.append((q.copy(), th.copy()))
    J = cost + 2.0 * np.abs(wrap(targets - th)) + 0.01 * act
    return (-J.mean(-1)), hist


# ---- 軌道生成器型方策（tg）-------------------------------------------------------
TG_NET = MLP([2, 16, 1])
KP_POS = 10.0                               # 目標関節角への追従ゲイン [1/s]


def tg_unpack(P):
    c = Q_LO + (np.tanh(P[:, 0:2]) + 1) / 2 * (Q_HI - Q_LO)
    return c, np.abs(P[:, 2:4]), P[:, 4], P[:, 5:]


def tg_target(c, A, dlt, psi):
    qt = c + A * np.stack([np.cos(psi), np.sin(psi + dlt)], -1)
    return np.clip(qt, Q_LO, Q_HI)


def rollout_tg(P, targets, record=False):
    pop, nb = targets.shape
    c, A, dlt, W = tg_unpack(P)
    q = np.broadcast_to(Q0, (pop, nb, 2)).copy(); th = np.zeros((pop, nb)); psi = np.zeros((pop, nb))
    cost = np.zeros((pop, nb)); hist = []
    for k in range(NSTEP):
        e = wrap(targets - th)
        u = TG_NET(W, np.stack([np.sin(e), np.cos(e)], -1))[..., 0]
        psi = psi + DT * 2 * np.pi * F_MAX * u
        qt = tg_target(c[:, None], A[:, None], dlt[:, None], psi)
        qd = np.clip(KP_POS * (qt - q), -QD_MAX, QD_MAX)
        q, th = step(q, th, qd)
        cost += np.abs(wrap(targets - th)) * DT / T_EP
        if record:
            hist.append((q.copy(), th.copy(), psi.copy()))
    J = cost + 2.0 * np.abs(wrap(targets - th))
    return -J.mean(-1), hist


# ---- 学習 --------------------------------------------------------------------
TRAIN_TGT = np.radians([-150, -110, -70, -30, 30, 70, 110, 150])
N_ITER = {"plain": 250, "cpg": 250, "tg": 250}
params, logs, rollouts = {}, {}, {}
for variant in ("plain", "cpg", "tg"):
    if variant == "tg":
        x0 = np.concatenate([[0, 0, 0.3, 0.3, 0.0], TG_NET.init(np.random.default_rng(0))])
        es = OpenAIES(x0, sigma=0.1, lr=0.03, n_pairs=40, seed=1)
        ro = lambda P, tg, record=False: rollout_tg(P, tg, record)
    else:
        cpg = variant == "cpg"
        net = MLP([6 if cpg else 4, 24, 24, 2])
        es = OpenAIES(net.init(np.random.default_rng(0)), sigma=0.08, lr=0.02, n_pairs=40, seed=1)
        ro = (lambda net, cpg: lambda P, tg, record=False:
              rollout(lambda o, k: net(P, o), tg, cpg, record))(net, cpg)
    log = []; t0 = time.time()
    for it in range(N_ITER[variant]):
        P = es.ask()
        fit, _ = ro(P, np.broadcast_to(TRAIN_TGT, (len(P), len(TRAIN_TGT))))
        es.tell(fit)
        if it % 10 == 0 or it == N_ITER[variant] - 1:
            f_mean, _ = ro(es.x[None], TRAIN_TGT[None])
            log.append((it, f_mean[0]))
            if it % 50 == 0 or it == N_ITER[variant] - 1:
                print(f"[{variant}] iter {it:4d}  reward {f_mean[0]:+.4f}  ({time.time() - t0:.0f} s)", flush=True)
    params[variant], logs[variant], rollouts[variant] = es.x.copy(), np.array(log), ro
    np.save(DATA / f"sim05a_policy_{variant}.npy", es.x)
    np.savetxt(DATA / f"sim05a_training_{variant}.csv", logs[variant], delimiter=",",
               header="iter,reward", comments="")

# ---- 専門家制御（モデルベース）---------------------------------------------------
ph = np.linspace(0, 2 * np.pi, 2001)
lq = np.c_[C1 + R * np.cos(ph), C2 + R * np.sin(ph)]
ld = np.c_[-R * np.sin(ph), R * np.cos(ph)]
inc = -np.sum(np.array([robot.connection(x) for x in lq]) * ld, 1)
geo = np.concatenate([[0], np.cumsum((inc[1:] + inc[:-1]) / 2 * np.diff(ph))])
osc = geo - ph / (2 * np.pi) * DTH_C
osc -= np.interp(np.pi, ph, osc)          # 初期位相 φ=π（Q0）で 0


def expert_run(target, kp=1.0):
    phi, th = np.pi, 0.0
    out = []
    for k in range(NSTEP):
        tb = th - np.interp(phi % (2 * np.pi), ph, osc)
        f = np.clip(kp * wrap(target - tb) / DTH_C, -F_MAX, F_MAX)
        qd = np.array([-R * np.sin(phi), R * np.cos(phi)]) * 2 * np.pi * f
        q = np.array([C1 + R * np.cos(phi), C2 + R * np.sin(phi)])
        _, th = step(q, th, qd)
        phi += DT * 2 * np.pi * f
        out.append((q, th))
    return out


# ---- 評価: 学習方策の軌道を厳密モデル（格子補間なし・細かい刻み）で再生して検証 ------
EVAL_TGT = np.radians([90.0, -90.0, 45.0, 150.0])
tt = np.arange(1, NSTEP + 1) * DT
LAST = tt > T_EP - 5.0


def exact_replay(qtraj):
    """関節軌道を 10 分割して厳密な Γ で積分し直す."""
    th = 0.0; qprev = Q0; out = []
    for qn in qtraj:
        for s in range(10):
            a = qprev + (qn - qprev) * s / 10; b = qprev + (qn - qprev) * (s + 1) / 10
            th -= robot.connection((a + b) / 2) @ (b - a)
        qprev = qn; out.append(th)
    return np.array(out)


res = {}
for variant in ("plain", "cpg", "tg"):
    _, hist = rollouts[variant](params[variant][None], EVAL_TGT[None], True)
    Qh = np.array([h[0][0] for h in hist]); Th = np.array([h[1][0] for h in hist])
    TH = np.array([exact_replay(Qh[:, j]) for j in range(len(EVAL_TGT))]).T
    res[variant] = (Qh, TH, np.abs(TH - Th).max())
TH_EXP = np.array([[e[1] for e in expert_run(tg)] for tg in EVAL_TGT]).T


def metrics(th, tg):
    e = np.abs(wrap(tg - th))
    return D(e[-1]), D(e[LAST].mean())


print("\n=== 評価（25 s, 指標: 終端誤差 / 最後 5 s の平均誤差）===")
summary = []
for j, tg in enumerate(EVAL_TGT):
    line = f"目標 {D(tg):+5.0f}°:"
    for name, th in [(v, res[v][1][:, j]) for v in ("plain", "cpg", "tg")] + [("expert", TH_EXP[:, j])]:
        a, b = metrics(th, tg)
        line += f"  {name} {a:5.1f}° / {b:5.1f}°"
        summary.append((D(tg), name, a, b))
    print(line)
for v in ("plain", "cpg", "tg"):
    print(f"[{v}] 格子モデルと厳密モデルの差 max {D(res[v][2]):.3f}°")
# 専門家の報酬（学習と同じ評価関数）
th_e = np.array([[e[1] for e in expert_run(tg)] for tg in TRAIN_TGT]).T
J_e = np.mean(np.abs(wrap(TRAIN_TGT - th_e)).mean(0) + 2.0 * np.abs(wrap(TRAIN_TGT - th_e[-1])))
print(f"学習と同じ評価関数での報酬: " + ", ".join(f"{v} {logs[v][-1, 1]:+.3f}" for v in logs) + f", expert {-J_e:+.3f}")
# tg が学習したループ（目標軌道）1 周あたりの正味回転を厳密モデルで計算
c, A, dlt, _ = tg_unpack(params["tg"][None])
psi = np.linspace(0, 2 * np.pi, 4001)
lt = tg_target(c[0], A[0], dlt[0], psi)
dq = np.diff(lt, axis=0); mid = (lt[1:] + lt[:-1]) / 2
dth_tg = -np.sum(np.sum(np.array([robot.connection(m) for m in mid]) * dq, 1))
print(f"[tg] 学習したループ: 中心 (腕 {D(c[0,0]):.0f}°, 脚 {D(c[0,1]):.0f}°), 振幅 ({D(A[0,0]):.0f}°, {D(A[0,1]):.0f}°), δ {D(dlt[0]):.0f}°"
      f" → 1 周あたり {D(dth_tg):+.2f}°（Sim03 最適円 {D(DTH_C):+.2f}°, 可動域内上界 +9.57°）")
LEARNED_LOOP = lt

# 学習した関節軌道の掃引面積（後半 = 定常周回部分）と曲率の対応
Qc = res["tg"][0]
for j in range(2):
    q = Qc[NSTEP // 10: NSTEP // 3, j]
    area = 0.5 * np.sum(q[:-1, 0] * q[1:, 1] - q[1:, 0] * q[:-1, 1])
    print(f"[tg] 目標 {D(EVAL_TGT[j]):+.0f}°: 関節軌道の符号付き掃引面積 {area:+.1f} rad²（正=反時計回り）")
with open(DATA / "sim05a_summary.csv", "w") as fh:
    fh.write("target_deg,controller,final_err_deg,mean_err_last5s_deg\n")
    for r in summary:
        fh.write(f"{r[0]:.0f},{r[1]},{r[2]:.3f},{r[3]:.3f}\n")
np.savez(DATA / "sim05a_eval.npz", targets=EVAL_TGT, t=tt,
         q_plain=res["plain"][0], th_plain=res["plain"][1],
         q_cpg=res["cpg"][0], th_cpg=res["cpg"][1], q_tg=res["tg"][0], th_tg=res["tg"][1],
         th_expert=TH_EXP, learned_loop=LEARNED_LOOP)

# ---- 図 --------------------------------------------------------------------------
F = np.gradient(GAM[..., 1], g1, axis=0) - np.gradient(GAM[..., 0], g2, axis=1)
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
vmax = np.abs(F).max()
for a, v in zip(ax[0], ("plain", "cpg", "tg")):
    im = a.pcolormesh(D(g1), D(g2), F.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    a.contour(D(g1), D(g2), F.T, levels=[0], colors="k", linewidths=.6, linestyles="--")
    a.plot(D(lq[:, 0]), D(lq[:, 1]), "k:", lw=1.2, label="expert loop (Sim03)")
    for j, c_ in zip(range(2), ["C1", "C2"]):
        qs = res[v][0][NSTEP // 10: NSTEP // 3, j]
        a.plot(D(qs[:, 0]), D(qs[:, 1]), c_, lw=.9, label=f"learned, target {D(EVAL_TGT[j]):+.0f}°")
    a.plot(D(Q0[0]), D(Q0[1]), "ko", ms=4)
    for j, c_ in zip(range(2), ["C1", "C2"]):          # 終端姿勢（静止した場合はここに留まる）
        a.plot(*D(res[v][0][-1, j]), "x", color=c_, ms=10, mew=2.5)
    a.set_xlabel("arm q1 [deg]"); a.set_ylabel("leg q2 [deg]")
    a.set_title(f"[{v}] learned joint trajectory"); a.legend(fontsize=7, loc="lower right")
fig.colorbar(im, ax=ax[0, 2])
for a, v in zip(ax[1, :2], ("plain", "tg")):
    for j, c_ in zip(range(len(EVAL_TGT)), ["C1", "C2", "C3", "C4"]):
        a.plot(tt, D(res[v][1][:, j]), c_, label=f"{v} → {D(EVAL_TGT[j]):+.0f}°")
        a.plot(tt, D(TH_EXP[:, j]), c_, ls=":", lw=1)
        a.axhline(D(EVAL_TGT[j]), color=c_, lw=.5)
    a.set_xlabel("time [s]"); a.set_ylabel("θ [deg]"); a.grid(alpha=.3)
    a.set_title(f"attitude (solid: {v}, dotted: expert)"); a.legend(fontsize=7)
for v in ("plain", "cpg", "tg"):
    ax[1, 2].plot(logs[v][:, 0], logs[v][:, 1], label=v)
ax[1, 2].axhline(-J_e, color="k", ls=":", label="expert")
ax[1, 2].set_xlabel("ES iteration"); ax[1, 2].set_ylabel("reward"); ax[1, 2].legend()
ax[1, 2].set_title("training curve"); ax[1, 2].grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "sim05a_learned_ambac.png", dpi=150)
