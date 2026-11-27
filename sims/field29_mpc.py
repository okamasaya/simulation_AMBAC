"""Field29: 足の置き場所の MPC（下半身の段 12）.

段階 24 の踏み出しは、1 歩ずつ足をずらして、歩行の計画（ZMP プレビュー制御＋10 ms ごとの逆運動学）を丸ごと作り直していた。
その結果、踏み出しすぎと逆向きの踏み出し（F43）、測った状態からの計画し直しで悪化（F44）、横は単調でない（F55）、1 本数分の計算、が残った。
ここでは、計画を前もって作らず、実行中に次の 2 つを 10 ms ごとに解く（Herdt ら 2010 の考え方を簡単にしたもの）。

  1. MPC（線形倒立振子、前後・左右で別々の 2 次計画、osqp）: 先 1.6 s（0.05 s × 32）の ZMP と、その間に着く足の置き場所を同時に決める
       制約: ZMP は支持する足の中（両脚支持では 2 つの足の間を時間で内分した場所の周り）、足は予定から ±0.3 m 以内、脚を交差させない
       評価: ZMP を足の中央に近づける、ZMP の変化を小さく、足を予定に近づける、先の終わりの DCM を足の中央に
       初期の状態: 内部の LIPM の状態に、測った重心の位置・速度を κ の割合で混ぜる（κ = 0 は開ループ、1 は測った値そのもの）
  2. 逆運動学: MPC の重心の目標になる骨盤の位置と、足首の位置（振り出す足は、いまの MPC の着地点へ向かう）から関節角

着地の t_fix 前（0.15 s）に、その足の置き場所を固定する。歩く時刻（いつ足を上げ、いつ着くか）は予定のまま変えない。
実行: python field29_mpc.py [walk|push|figure]
"""
import json
import sys
import time

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import osqp
from scipy import sparse

import field21_walk as W
import field23_stabilizer as F23
import field24_stepping as F24
import field25_estimation as F25
from common import DATA, smooth_step

T0 = time.time()
OUT = W.OUT
G = W.G


class MPC:
    """前後・左右で別々の 2 次計画。変数は ZMP z_0..z_{N-1} と、先で着く足の置き場所 f_j"""

    def __init__(self, zc, Tm=0.05, N=32, hx=0.09, hy=0.035, w=None):
        self.om = np.sqrt(G / zc); self.Tm = Tm; self.N = N; self.h = (hx, hy)
        o = self.om; ch, sh = np.cosh(o * Tm), np.sinh(o * Tm)
        self.Ad = np.array([[ch, sh / o], [o * sh, ch]]); self.Bd = np.array([1 - ch, -o * sh])
        # x_k = A^k x0 + Σ A^{k-1-i} B z_i
        self.Ak = [np.linalg.matrix_power(self.Ad, k) for k in range(N + 1)]
        self.w = {**dict(z=1.0, dz=10.0, f=0.05, xi=20.0), **(w or {})}

    def solve(self, x0, refs, ax, free, fnom, fbox, extra_ineq=()):
        """x0: (c, v)。refs: N 個の (λ, P, Q)。P・Q は ("k", 値) か ("f", 何番目の変数か)、ZMP の中央 = λ P + (1−λ) Q。
        free: 変数の足の数。fnom: 予定の位置。fbox: (下限, 上限)。extra_ineq: [(係数の辞書 {j: a}, 下限, 上限)]（足どうしの制約）"""
        N, nf = self.N, free; nv = N + nf
        # 中央 = a_k + Σ_j b_kj f_j
        a = np.zeros(N + 1); Bm = np.zeros((N + 1, nf))
        for k, (lam, P, Q) in enumerate(refs):
            for wgt, term in ((lam, P), (1 - lam, Q)):
                if term[0] == "k":
                    a[k] += wgt * term[1]
                else:
                    Bm[k, term[1]] += wgt
        # 先の終わりの DCM ξ_N = c_N + v_N / ω（z に線形）
        o = self.om; e = np.array([1.0, 1.0 / o])
        xiN0 = e @ self.Ak[N] @ x0; xiZ = np.array([e @ self.Ak[N - 1 - i] @ self.Bd for i in range(N)])
        H = np.zeros((nv, nv)); g = np.zeros(nv); wz, wdz, wf, wxi = (self.w[k] for k in ("z", "dz", "f", "xi"))
        for k in range(N):          # wz (z_k − a_k − B_k f)²
            r = np.zeros(nv); r[k] = 1; r[N:] = -Bm[k]
            H += wz * np.outer(r, r); g += -wz * r * a[k]
            if k > 0:
                r = np.zeros(nv); r[k] = 1; r[k - 1] = -1; H += wdz * np.outer(r, r)
        for j in range(nf):
            H[N + j, N + j] += wf; g[N + j] += -wf * fnom[j]
        r = np.zeros(nv); r[:N] = xiZ; r[N:] = -Bm[N]       # wxi (ξ_N − a_N − B_N f)²
        H += wxi * np.outer(r, r); g += wxi * r * (xiN0 - a[N])
        rows = []; lo = []; hi = []; h = self.h[ax]
        for k in range(N):          # |z_k − 中央_k| ≤ h
            r = np.zeros(nv); r[k] = 1; r[N:] = -Bm[k]; rows.append(r); lo.append(a[k] - h); hi.append(a[k] + h)
        for j in range(nf):
            r = np.zeros(nv); r[N + j] = 1; rows.append(r); lo.append(fbox[j][0]); hi.append(fbox[j][1])
        for coef, l_, u_ in extra_ineq:
            r = np.zeros(nv)
            for j, cval in coef.items():
                r[N + j] = cval
            rows.append(r); lo.append(l_); hi.append(u_)
        Acon = sparse.csc_matrix(np.array(rows)); Hs = sparse.csc_matrix(H + 1e-9 * np.eye(nv))
        pr = osqp.OSQP(); pr.setup(Hs, g, Acon, np.array(lo), np.array(hi), verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=4000, polishing=False)
        res = pr.solve()
        if res.x is None or res.info.status_val not in (1, 2):
            return None
        return res.x[:N], res.x[N:]


def schedule(c, steps, Tds, xc):
    """各歩の時刻と、ZMP の中央を決める 2 つの足（足の番号）。足の番号: -2 = 最初の左、-1 = 最初の右、k = k 歩目の着地"""
    s = c.s
    first = {"L": -2, "R": -1}; last_of = dict(first); stance_of = []
    for k, (t0, t1, sw, a_, b_) in enumerate(steps):
        st_side = "L" if sw == "R" else "R"
        stance_of.append(last_of[st_side]); last_of[sw] = k
    return stance_of


def center_terms(t, c, steps, Tds, stance_of):
    """時刻 t の ZMP の中央 = λ P + (1−λ) Q。P, Q は足の番号（"mid" は最初または最後の両足の中央）"""
    for k, (t0, t1, sw, a_, b_) in enumerate(steps):
        st = stance_of[k]
        if t < t0:
            return (1.0, "mid0", "mid0")
        if t < t0 + Tds / 2:
            if k == 0:
                u = smooth_step((t - t0) / (Tds / 2))[0]
                return (u, st, "mid0")
            return (1.0, st, st)
        if t < t1 - Tds / 2:
            return (1.0, st, st)
        if t < t1:
            u = smooth_step((t - (t1 - Tds / 2)) / (Tds / 2))[0]
            if k == len(steps) - 1:
                return (1 - u, st, "end")
            return (1 - u, st, k)
    return (1.0, "end", "end")


def execute_mpc(c: W.Cfg, kappa=0.3, est_cfg=None, K_stab=0.0, record=False, t_fix=0.15, step_max=0.30, w=None, lead=0.0, tau=0.0, mix_state=False, dead=0.0):
    s = c.s
    rb = W.Robot(c); rk = W.Robot(c)         # rb: MuJoCo の実行、rk: 逆運動学（実行中の MjData を書き換えないため）
    m, d = rb.m, rb.d
    steps, T_end, ah = W.footsteps(c)
    Tds = c.ds * c.T_step * np.sqrt(s); xc = (W.TOE - W.HEEL) / 2 * s
    stance_of = schedule(c, steps, Tds, xc)
    feet0, _ = W.feet_at(0.0, c, steps, ah)
    z_hip = ah + c.z_hip * s
    q00 = rk.d.qpos.copy()
    for sd, a_ in W.arm_angles(0.0, c, steps, ah).items():
        q00[rk.arm[sd]] = a_
    q0, _ = rk.ik((0, 0, z_hip), feet0, q00)
    zc = rk.com(q0)[2]
    mpc = MPC(zc, w=w); om = mpc.om
    # 足の置き場所: 最初の両足（既知）と、各歩の着地（予定 → MPC が更新、着地の t_fix 前に固定）
    pos = {-2: np.array([0.0, W.HIP_Y * s]), -1: np.array([0.0, -W.HIP_Y * s])}
    fnom = {k: st[4].copy() for k, st in enumerate(steps)}
    fcur = {k: st[4].copy() for k, st in enumerate(steps)}
    fixed = set()
    act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in W.JOINTS]
    adr = [rk.qa[j] for j in W.JOINTS]
    if rk.arm:
        act += [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"sh{sd}") for sd in "LR"]
        adr += [rk.arm[sd] for sd in "LR"]
    kp = c.kp * s**3
    mujoco.mj_resetData(m, d); d.qpos[:] = q0; d.qpos[2] += 0.0005 * s; mujoco.mj_forward(m, d)
    com0 = rk.com(q0)[:2]
    xm = [np.array([com0[ax], 0.0]) for ax in range(2)]      # 内部の LIPM の状態（前後・左右）
    hist = []          # 内部の状態の履歴（サーボの遅れ tau だけ前の状態と、測った値を比べる）
    ntau = int(round(tau / 0.01))
    nc = int(round(0.01 / m.opt.timestep)); n = int(round(T_end / m.opt.timestep))
    q = q0.copy(); xy = q[0:2].copy(); u_ = None; zmp_cmd = com0.copy()
    push = c.extra.get("push"); tilt_max = 0.0; fell = False; n_fail = 0
    est = W.Estimator(est_cfg, rb, m, om, np.random.default_rng(7), c) if est_cfg else None
    ne = int(round(0.001 / m.opt.timestep)); ce = ve = None
    rec = {k: [] for k in ("t", "com", "zmp", "ref", "feet")} if record else None
    Lr = 0.98 * (W.L_TH + W.L_SH) * s

    for kk in range(n):
        tt = d.time
        if est is not None and kk % ne == 0:
            fe, su = W.feet_at(tt, c, steps, ah)
            ce, ve = est.step(d, fe, su, ne * m.opt.timestep)
        if kk % nc == 0:
            mujoco.mj_subtreeVel(m, d)
            if est is not None:
                cm, vm = ce, ve
            else:
                cm, vm = d.subtree_com[1][:2].copy(), d.subtree_linvel[1][:2].copy()
            # 固定: 着地の t_fix 前を過ぎた足
            for k_, st_ in enumerate(steps):
                if k_ not in fixed and tt > st_[1] - Tds / 2 - t_fix * np.sqrt(s):
                    fixed.add(k_)
            # MPC の変数にする足: まだ固定していない、先 N Tm 以内に着く足
            Th = mpc.N * mpc.Tm
            free_idx = [k_ for k_, st_ in enumerate(steps) if k_ not in fixed and st_[1] - Tds / 2 < tt + Th + c.T_step]
            vmap = {k_: j for j, k_ in enumerate(free_idx)}

            def term(idx, ax):
                if idx == "mid0":
                    return ("k", 0.5 * (pos[-2][ax] + pos[-1][ax]) + (xc if ax == 0 else 0.0))
                if idx == "end":
                    a1 = fcur[len(steps) - 1]; a2 = fcur[len(steps) - 2]
                    return ("k", 0.5 * (a1[ax] + a2[ax]) + (xc if ax == 0 else 0.0))   # 最後は両足の中央（固定とみなす）
                if idx in vmap:
                    return ("f", vmap[idx])
                p_ = pos[idx] if idx < 0 else fcur[idx]
                return ("k", p_[ax] + (xc if ax == 0 else 0.0))

            hist.append([x.copy() for x in xm])
            xd = hist[max(0, len(hist) - 1 - ntau)]     # 遅れ tau だけ前の内部の状態（測った重心は、サーボの遅れでこれに近いはず）
            # 測った値と内部の状態の差。DCM のずれが dead 以下なら 0（押されていないときの小さなずれ・遅れで目標を動かさない）
            dlt = []; exi = np.zeros(2)
            for ax in range(2):
                e_ = np.array([cm[ax], vm[ax]]) - xd[ax]; exi[ax] = e_[0] + e_[1] / om
                g_ = max(0.0, 1.0 - dead * s / max(abs(exi[ax]), 1e-9)) if dead > 0 else 1.0
                dlt.append(kappa * g_ * e_)
            out_z = []; ok_all = True
            for ax in range(2):
                x0 = xm[ax] + dlt[ax]
                # 前後の変数は「足首 + xc」（足の中央）にする。既知の足の中央は term で xc を足してある
                sh_ = xc if ax == 0 else 0.0
                refs = []
                for i in range(mpc.N + 1):
                    lam, P_, Q_ = center_terms(tt + i * mpc.Tm, c, steps, Tds, stance_of)
                    refs.append((lam, term(P_, ax), term(Q_, ax)))
                fn = [fnom[k_][ax] + sh_ for k_ in free_idx]
                fb = [(fnom[k_][ax] + sh_ - step_max * s, fnom[k_][ax] + sh_ + step_max * s) for k_ in free_idx]
                ineq = []
                if ax == 1:          # 脚を交差させない: 左足 − 右足 ≥ 0.12
                    for k_ in free_idx:
                        sw = steps[k_][2]; other = stance_of[k_]
                        sign = 1.0 if sw == "L" else -1.0
                        coef = {vmap[k_]: sign}; lb = 0.12 * s
                        if other in vmap:
                            coef[vmap[other]] = -sign
                        else:
                            op = pos[other] if other < 0 else fcur[other]
                            lb = 0.12 * s + sign * op[1]
                        ineq.append((coef, lb, np.inf))
                sol = mpc.solve(x0, refs, ax, len(free_idx), fn, fb, ineq)
                if sol is None:
                    ok_all = False; out_z.append(None); continue
                zs, fs = sol
                for k_, fv in zip(free_idx, fs):
                    fcur[k_][ax] = fv - sh_
                out_z.append(zs[0])
            if not ok_all:
                n_fail += 1
            for ax in range(2):
                if out_z[ax] is not None:
                    zmp_cmd[ax] = out_z[ax]
                # 内部の状態を 10 ms 進める（測った値を κ で混ぜた状態から）
                # 内部の状態は、測った値で書き換えない（測った値は MPC の判断にだけ使う。書き換えると、サーボの遅れが積み重なって目標が後ろへずれる）
                x0 = xm[ax] + dlt[ax] if mix_state else xm[ax]
                dtc = nc * m.opt.timestep; o = om
                ch, sh = np.cosh(o * dtc), np.sinh(o * dtc)
                Ad = np.array([[ch, sh / o], [o * sh, ch]]); Bd = np.array([1 - ch, -o * sh])
                xm[ax] = Ad @ x0 + Bd * zmp_cmd[ax]
            com_ref = np.array([xm[0][0] + lead * xm[0][1], xm[1][0] + lead * xm[1][1]])
            if K_stab > 0:       # 段階 23 と同じ足首の戦略: DCM のずれと逆に重心の目標をずらす（最大 8 cm）
                com_ref = com_ref - np.clip(K_stab * exi, -0.08 * s, 0.08 * s)
            # 足首の目標: 着いた足は MPC の値、振り出す足は離れた場所から MPC のいまの着地点へ
            feet = {}; support = "LR"
            last = {"L": pos[-2].copy(), "R": pos[-1].copy()}
            for k_, (t0, t1, sw, a_, b_) in enumerate(steps):
                ta, tb = t0 + Tds / 2, t1 - Tds / 2
                if tt >= tb:
                    last[sw] = fcur[k_].copy()
                elif tt > ta:
                    u = (tt - ta) / (tb - ta); v = smooth_step(u)[0]
                    p0 = last[sw]; p1 = fcur[k_]
                    xy_ = p0 + (p1 - p0) * v
                    feet[sw] = np.array([xy_[0], xy_[1], ah + c.lift * s * np.sin(np.pi * u)])
                    support = "L" if sw == "R" else "R"
                    break
                else:
                    break
            for sd in "LR":
                if sd not in feet:
                    feet[sd] = np.array([last[sd][0], last[sd][1], ah])
            for sd, a_ in arm_angles_feet(c, feet).items():
                q[rk.arm[sd]] = a_
            z = z_hip
            for sd, hy in (("L", W.HIP_Y * s), ("R", -W.HIP_Y * s)):
                dx, dy = feet[sd][0] - xy[0], feet[sd][1] - (xy[1] + hy)
                z = min(z, feet[sd][2] + np.sqrt(max(Lr**2 - dx * dx - dy * dy, 0.0)))
            q, _ = rk.pelvis_for_com(com_ref, z, feet, q, xy)
            xy = q[0:2].copy()
            u_ = q[adr].copy()
            if c.ff:
                u_[:len(W.JOINTS)] += rk.static_torques(q, support) / kp
            if record:
                rec["t"].append(tt); rec["com"].append(d.subtree_com[1][:2].copy()); rec["zmp"].append(zmp_cmd.copy())
                rec["ref"].append(com_ref.copy()); rec["feet"].append({k_: fcur[k_].copy() for k_ in fcur})
        d.ctrl[act] = u_
        if push:
            d.xfrc_applied[1, :3] = np.r_[push["F"], 0.0] * s**3 if push["t"] <= tt < push["t"] + push.get("dur", 0.1) * np.sqrt(s) else 0.0
        mujoco.mj_step(m, d)
        Rm = d.xmat[1].reshape(3, 3); tilt_max = max(tilt_max, np.degrees(np.arccos(np.clip(Rm[2, 2], -1, 1))))
        if d.qpos[2] < 0.5 * rb.z0 or not np.isfinite(d.qpos).all():
            fell = True; break
    end = 0.5 * (fcur[len(steps) - 1] + fcur[len(steps) - 2])
    pos_err = float(np.linalg.norm(d.subtree_com[1][:2] - end)) if not fell else np.inf
    nom_end = 0.5 * (fnom[len(steps) - 1] + fnom[len(steps) - 2])
    out = dict(fell=fell, tilt_max=float(tilt_max), pos_err=pos_err, t_end=float(d.time), n_fail=n_fail,
               offs={k_: np.round(fcur[k_] - fnom[k_], 3).tolist() for k_ in fcur if np.linalg.norm(fcur[k_] - fnom[k_]) > 0.01},
               end_shift=float(np.linalg.norm(end - nom_end)))
    out["ok"] = bool(not fell and tilt_max < 15 and pos_err < 0.10 * s)
    if est is not None and est.err:
        out["est_rms"] = np.sqrt(np.mean(np.array(est.err)**2, 0)).tolist()
    if record:
        out["rec"] = rec
    return out


def arm_angles_feet(c, feet):
    ar_ = c.extra.get("arms")
    if not ar_:
        return {}
    u = (feet["L"][0] - feet["R"][0]) / (c.L * c.s)
    A = np.radians(ar_.get("A", 0.0))
    return {"R": A * u, "L": -A * u}


WW = {"f": 50.0, "z": 10.0, "dz": 0.1, "xi": 1.0}
CASES = {
    "MPC 開ループ（κ 0、K 0）": dict(kappa=0.0),
    "MPC＋足首の戦略 K 1（測った値は骨盤だけ）": dict(kappa=0.0, K_stab=1.0),
    "MPC＋足首 K 1＋足の置き場所（不感帯 4 cm、κ 1）": dict(kappa=1.0, mix_state=True, dead=0.04, K_stab=1.0),
    "MPC＋足首 K 1＋足の置き場所（不感帯 6 cm、κ 0.5）": dict(kappa=0.5, mix_state=True, dead=0.06, K_stab=1.0),
}
FS = [300, 400, 500, 600, 700]
DIRS = {"前 +x": (1, 0), "後ろ −x": (-1, 0), "横（振り出す足の側）+y": (0, 1), "横（立つ足の側）−y": (0, -1)}


def _run(a):
    name, over, push = a
    ex = {"arms": {"A": F23.A_STAR}}
    if push:
        ex["push"] = push
    t0 = time.time()
    try:
        o = execute_mpc(W.Cfg(extra=ex, **over), w=WW, **CASES[name])
    except Exception as e:  # noqa: BLE001
        o = dict(ok=False, fell=True, err=str(e), offs={})
    return dict(ok=o["ok"], fell=o["fell"], offs=o.get("offs"), sec=time.time() - t0)


def sym(r):
    return "○" if r["ok"] else ("△" if not r.get("fell") else "×")


def main():
    from multiprocessing import Pool
    tp = F23.push_time(0.8)
    jobs = [(n, {"L": L, "T_step": T}, None) for n in CASES for L in F23.LS for T in F23.TS]
    jobs += [(n, {"T_step": 0.8}, {"t": tp, "F": [dx * f, dy * f], "dur": 0.1}) for n in CASES for dx, dy in DIRS.values() for f in FS]
    jobs += [(list(CASES)[2], {"T_step": 0.8}, {"t": tp + dt0, "F": [0, f], "dur": 0.1}) for dt0 in (0.001, 0.002, 0.005, 0.01) for f in FS]
    with Pool(4) as pool:
        res = pool.map(_run, jobs, chunksize=1)
    ng = len(F23.LS) * len(F23.TS); nc = len(CASES); npush = len(DIRS) * len(FS)
    out = {"grid": [], "push": [], "repeat": [], "LS": F23.LS, "TS": F23.TS, "FS": FS, "dirs": list(DIRS)}
    print(f"=== 歩ける範囲（段階 23 と同じ 24 通り、腕の振りあり、真の値）。行 = 歩幅 {F23.LS}、並び = T {F23.TS}。1 本の計算時間の平均 ===")
    for a, n in enumerate(CASES):
        rs = res[a * ng:(a + 1) * ng]
        print(f"| {n} | {' '.join(''.join(sym(r) for r in rs[b * 8:(b + 1) * 8]) for b in range(3))} | {np.mean([r['ok'] for r in rs]):.2f} | {np.mean([r['sec'] for r in rs]):.1f} s |", flush=True)
        out["grid"].append(dict(name=n, frac=float(np.mean([r["ok"] for r in rs])), ok=[r["ok"] for r in rs]))
    base = nc * ng
    print(f"=== 押されたときの回復（T 0.8 s、L 0.25 m、4 歩目の片脚支持の中ごろ、0.1 s、{FS} N）。耐えた力積 [N·s] ===")
    for a, n in enumerate(CASES):
        rr = res[base + a * npush: base + (a + 1) * npush]; row = []; lims = []; seqs = []
        for b, dn in enumerate(DIRS):
            l, sq = F24.limit(rr[b * len(FS):(b + 1) * len(FS)], FS); lims.append(l / 10 if l else 0); seqs.append(sq)
            row.append(f"{dn} {sq} {lims[-1]:.0f}")
        print(f"| {n} | " + " | ".join(row) + " |", flush=True)
        for b, dn in enumerate(DIRS):
            for f, r in zip(FS, rr[b * len(FS):(b + 1) * len(FS)]):
                print(f"      {dn} {f} N: {sym(r)} 足のずらし {r['offs']}")
        out["push"].append(dict(name=n, lims=lims, seq=seqs))
    base2 = base + nc * npush
    print("=== 再現性: 横（振り出す足の側）を押す時刻を 1〜10 ms ずらす（不感帯 4 cm、κ 1） ===")
    for j, dt0 in enumerate((0.001, 0.002, 0.005, 0.01)):
        rr = res[base2 + j * len(FS): base2 + (j + 1) * len(FS)]
        print(f"  +{dt0 * 1000:.0f} ms: {''.join(sym(r) for r in rr)}"); out["repeat"].append(dict(dt=dt0, seq="".join(sym(r) for r in rr)))
    json.dump(out, open(DATA / "field29_mpc.json", "w"), indent=1, ensure_ascii=False, default=str)
    print(f"[{time.time() - T0:.0f} s]")


def figure():
    res = json.load(open(DATA / "field29_mpc.json")); a24 = json.load(open(DATA / "field27_push_estimated.json"))
    en = ["MPC open loop", "MPC + ankle K1", "MPC + ankle K1 + footstep (dead 4 cm)", "MPC + ankle K1 + footstep (dead 6 cm, k 0.5)"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1, 1.4]})
    fr = [g["frac"] for g in res["grid"]] + [0.79]; nm = en + ["preplanned + ankle K1 (stage 23)"]
    ax[0].barh(range(len(fr)), fr, color=["tab:gray", "tab:blue", "tab:green", "tab:olive", "k"])
    ax[0].set_yticks(range(len(fr))); ax[0].set_yticklabels(nm, fontsize=7); ax[0].invert_yaxis(); ax[0].set_xlabel("fraction of 24 gaits walked")
    ax[0].set_title("walking range (true state)", fontsize=9)
    rows = [("preplanned + stepping (stage 24)", a24["res"][0]["seq"][:3] + ["−−−−−"])] + [(n, r["seq"]) for n, r in zip(en, res["push"])]
    sm = {"○": 2, "△": 1, "×": 0, "−": np.nan}
    M = np.array([[sm[ch] for sq in r for ch in sq] for _, r in rows], float)
    ax[1].imshow(M, cmap="RdYlGn", vmin=0, vmax=2, aspect="auto")
    ax[1].set_yticks(range(len(rows))); ax[1].set_yticklabels([n for n, _ in rows], fontsize=7)
    ax[1].set_xticks(range(20)); ax[1].set_xticklabels([f"{f // 10}" for _ in range(4) for f in res["FS"]], fontsize=6)
    for k in (1, 2, 3):
        ax[1].axvline(k * 5 - 0.5, color="k", lw=1.5)
    for k, t_ in enumerate(["forward", "backward", "lateral (swing side)", "lateral (stance side)"]):
        ax[1].text(k * 5 + 2, -0.8, t_, ha="center", fontsize=7)
    ax[1].set_xlabel("impulse [N s] (green walked, yellow drifted, red fell, white not tested)")
    fig.tight_layout(); fig.savefig(OUT / "field29_mpc.png", dpi=130)
    print("図 → figures/field/field29_mpc.png")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "main"
    if arg == "test":
        t0 = time.time()
        r = execute_mpc(W.Cfg(extra={"arms": {"A": F23.A_STAR}}), w=WW, **CASES[list(CASES)[2]])
        print({k: v for k, v in r.items() if k != "rec"}, f"{time.time() - t0:.1f}s")
    else:
        {"main": main, "figure": figure}[arg]()
