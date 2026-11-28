"""番外編 08: 番外編 04 のブラインド解析を、体との接触と手を組む力を入れてやり直す（事前登録、docs/22）.

番外編 04（docs/17）では、文献の肩・肘の筋力と可動域だけでは最良 1.99 回転で、普通の 3A（3.43）に届かなかった（F72）。
「実際の 3A では、腕を胸に押し付ける・手をつかむといった接触が、腕を保つ力の大部分を担っているはず」という読みを確かめる。

番外編 04 から変えるのは、筋力の制約の中身だけ。最適化（変数・初期値・2 段・反復の数・乱数の種）は同じ。
  必要な関節トルク τ_req（逆動力学） = 筋肉のトルク τ_m + 接触の力の寄与 Σ J_pᵀ f_p
  各時刻で、接触の力 f_p を「筋肉の上限からの、はみ出しの最大」が最小になるように選ぶ（線形計画）。
  その τ_m を、番外編 04 と同じ 8 つの上限（屈曲・伸展・外転・内転・外旋・内旋・肘の屈曲・伸展）で判定する。

接触（体の中の力なので、全体の角運動量は変えない）
  胴体: 腕の 7 点（上腕 3・前腕 3・手 1）のうち、胴体のカプセルの表面から 1 cm 以内の点。
        力は外向きの法線の向き（押すだけ）と摩擦（μ = 0.5、四角錐で近似）。大きさの上限 2000 N（数値の箱）
  手を組む: 左右の手の中心が 3 cm 以内（手の球の半径 4 cm × 2 + 3 cm、すなわち手の中心の y ≤ 0.055 m）。
        左右対称なので、力は左右の向き（y）だけ。引く・押すの両方、大きさ ≤ 220 N（握力の約半分、仮定）
  左右の手が中心線を越えて重なることは禁止（手の中心の y ≥ 0.04 m）。番外編 04 の上限ありの解はこれを満たしている

条件
  contact : limited + 胴体との接触
  clasp   : limited + 手を組む
  both    : limited + 両方
  （対照は番外編 04 の limited。同じ種・同じコードなので再実行しない）
実行: python bangai08_contact_blind.py [run COND SEED|check|metrics|figure]
"""
from __future__ import annotations

import json
import sys
import time

import mujoco
import numpy as np
from scipy.optimize import linprog

import bangai04_shoulder_blind as P
from common import DATA, ROOT

OUT = ROOT / "figures" / "field"
MU = 0.5
F_MAX = 2000.0
F_CLASP = 220.0
TOUCH = 0.01                    # 胴体の表面から 1 cm 以内を接触とみなす
HAND_R = 0.04
CLASP_Y = HAND_R + 0.015        # 手の中心の y ≤ 0.055 m で手を組める
HAND_Y_MIN = 0.04               # 手が中心線を越えない
CONDS = dict(contact=dict(contact=True, clasp=False), clasp=dict(contact=False, clasp=True),
             both=dict(contact=True, clasp=True))
SEEDS = P.SEEDS
PT_BODY = ["uaL"] * 3 + ["faL"] * 3 + ["hdL"]


class ContactProblem(P.Problem):
    def __init__(self, cond, **kw):
        super().__init__("limited", **kw)
        self.cond = cond
        self.opt = CONDS[cond]
        m = self.sk.m
        self.bid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in set(PT_BODY)}
        self.iv = np.r_[self.sk.jadr["L"]] - 1       # 左腕の qvel の番地
        self.ia = np.r_[self.sk.jadr["L"], self.sk.jadr["R"]]
        T = self.tq
        # τ の下限・上限（番外編 04 の向き: [sf, sa, sr, el]）
        self.tlo = -np.array([T["flex"], T["add"], T["ir"], T["ele"]])
        self.thi = np.array([T["ext"], T["abd"], T["er"], T["elf"]])
        self.lp_stats = [0, 0]
        self.ckpt, self.ckpt_best, self.t_start = None, -np.inf, time.time()

    def trunk_normal(self, p):
        """胴体のカプセルの外向きの法線と、表面からの距離（腕の太さを引いた値）."""
        z = np.clip(p[2], 0.0, self.zc7)
        v = p - np.array([0.0, 0.0, z])
        r = np.linalg.norm(v)
        return v / max(r, 1e-9), r - self.B["chest"] - P.CLEAR

    def muscle_torque(self, k, t, q, tau_req, st):
        """時刻 k の筋肉のトルク（左腕 4 関節）. 接触がなければ τ_req のまま."""
        pts = self.points(q)
        cand = []
        if self.opt["contact"]:
            for i, p in enumerate(pts):
                n, dist = self.trunk_normal(p)
                if dist < TOUCH:
                    cand.append(("c", i, p, n))
        clasp = self.opt["clasp"] and pts[-1][1] <= CLASP_Y
        if not cand and not clasp:
            return tau_req
        if k % 2 == 1:          # 計算時間のため、接触のある時刻は 4 ms ごと（奇数の刻みは判定から外す: None）
            return None
        m, d = self.sk.m, self.sk.d
        qa, qad, _ = st["plan"](t)
        d.qpos[:7] = st["y"][k]; d.qpos[self.ia] = qa
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        R = d.xmat[self.sk.trunk].reshape(3, 3)
        o = d.xpos[self.sk.trunk]
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
        cols = []        # 各列: 単位の力（係数）あたりの関節トルクの寄与 Jᵀ f
        bnds = []
        for _, i, p, n in cand:
            mujoco.mj_jac(m, d, jacp, jacr, o + R @ p, self.bid[PT_BODY[i]])
            Jt = jacp[:, self.iv].T                     # (4, 3)
            nw = R @ n
            a = np.cross(nw, [0, 0, 1.0]) if abs(nw[2]) < 0.9 else np.cross(nw, [1.0, 0, 0])
            a /= np.linalg.norm(a); b = np.cross(nw, a)
            for tdir in (a, -a, b, -b):                 # 摩擦の四角錐の 4 本の辺（係数 ≥ 0）
                cols.append(Jt @ (nw + MU * tdir)); bnds.append((0.0, F_MAX / 4))
        if clasp:
            mujoco.mj_jac(m, d, jacp, jacr, o + R @ pts[-1], self.bid["hdL"])
            cols.append(jacp[:, self.iv].T @ R[:, 1]); bnds.append((-F_CLASP, F_CLASP))
        A = np.array(cols).T                           # (4, nc)
        nc = A.shape[1]
        # 変数 [c (nc), s]. 最小化 s + 1e-6 Σ|c|。 tlo − s·w ≤ τ_req − A c ≤ thi + s·w、w = 10 N·m
        w = 10.0
        cobj = np.r_[1e-6 * np.ones(nc), 1.0]
        A_ub = np.r_[np.c_[A, -w * np.ones(4)], np.c_[-A, -w * np.ones(4)]]
        b_ub = np.r_[tau_req - self.tlo, self.thi - tau_req]
        res = linprog(cobj, A_ub=A_ub, b_ub=b_ub, bounds=bnds + [(None, None)], method="highs")
        self.lp_stats[0] += 1
        if res.status != 0:
            self.lp_stats[1] += 1
            return tau_req
        return tau_req - A @ res.x[:nc]

    def evaluate(self, x):
        key = x.tobytes()
        if key in self.cache:
            return self.cache[key]
        self.nev += 1
        r = self.sk.fly(None, P.L_NOM, P.T_AIR, dt=self.dt, plan=self.plan(x), tau_samples=True, return_states=True)
        st = r["states"]
        cs = self.spline(x)
        tt = r["tau_t"]
        tau = [self.muscle_torque(int(round(t / self.dt)), t, cs(t), r["tau_all"][j, :4], st) for j, t in enumerate(tt)]
        tau = np.array([v for v in tau if v is not None])
        T = self.tq
        cons = [T["ext"] - tau[:, 0].max(), T["flex"] + tau[:, 0].min(),
                T["abd"] - tau[:, 1].max(), T["add"] + tau[:, 1].min(),
                T["er"] - tau[:, 2].max(), T["ir"] + tau[:, 2].min(),
                T["elf"] - tau[:, 3].max(), T["ele"] + tau[:, 3].min()]
        cons = [c / 10.0 for c in cons]
        ts = np.linspace(0, P.T_AIR, 141)
        Q = cs(ts)
        cons += list(np.degrees(Q.min(0) - self.lo) / 10.0) + list(np.degrees(self.hi - Q.max(0)) / 10.0)
        clr = min(self.clearance(q) for q in Q[::2])
        hand_y = min(self.points(q)[-1][1] for q in Q[::2])
        cons += [clr / 0.02, (hand_y - HAND_Y_MIN) / 0.02, (P.TILT_LAND - r["tilt_land"]) / 2.0, (P.TILT_MAX - r["tilt_max"]) / 2.0]
        r.pop("states", None)
        out = (r["turns"], np.array(cons), r)
        self.cache[key] = out
        # 途中経過の保存（F87: 環境の再起動で走行中の計算が消えたため）。実行できる点で回転数が最大のもの
        if self.ckpt and min(cons) >= -1e-3 and r["turns"] > self.ckpt_best:
            self.ckpt_best = r["turns"]
            json.dump(dict(cond=self.cond, x=x.tolist(), turns=r["turns"], nev=self.nev, sec=time.time() - self.t_start),
                      open(self.ckpt, "w"))
        return out


def contact_profile(prob, x):
    """解の時刻ごとの接触の有無（指標用）."""
    cs = prob.spline(np.asarray(x))
    ts = np.linspace(0, P.T_AIR, 141)
    tc, tk = [], []
    for t in ts:
        pts = prob.points(cs(t))
        tc.append(any(prob.trunk_normal(p)[1] < TOUCH for p in pts))
        tk.append(pts[-1][1] <= CLASP_Y)
    return float(np.mean(tc)), float(np.mean(tk))


def run(cond, seed):
    p = ContactProblem(cond)
    (DATA / "bangai08").mkdir(exist_ok=True)
    p.ckpt = DATA / "bangai08" / f"{cond}_{seed}_partial.json"
    r = p.solve(seed)
    r.update(P.smoothness(p, r["x"]))
    r["lp_calls"], r["lp_fail"] = p.lp_stats
    r["frac_trunk_contact"], r["frac_clasp"] = contact_profile(p, r["x"])
    f = DATA / "bangai08" / f"{cond}_{seed}.json"
    f.parent.mkdir(exist_ok=True)
    json.dump(r, open(f, "w"), default=float)
    # ブラインド: 数値だけを出し、腕の位置は出さない
    print(f"{cond:8s} seed {seed}: {r['turns']:.3f} 回転  可否 {r['feasible']}  最小の余裕 {r['cmin']:+.3f}  "
          f"SPARC {r['sparc']:.3f}  LDLJ {r['ldlj']:.2f}  反転 {r['reversals']}  反復 {r['nit']}  LP {p.lp_stats}  {r['sec']:.0f} s", flush=True)


def check():
    """計算の確認（事前登録の前に回してよい）: 接触なしの条件で番外編 04 の limited の解を評価し直して一致するか、1 回の評価の時間."""
    out = {}
    r04 = json.load(open(DATA / "bangai04" / "limited_13.json"))
    x = np.array(r04["x"])
    p0 = P.Problem("limited")
    n0, c0, _ = p0.evaluate(x)
    for cond in CONDS:
        p = ContactProblem(cond)
        t0 = time.time()
        n, c, _ = p.evaluate(x)
        out[cond] = dict(turns=n, cmin_torque=float(c[:8].min()), cmin=float(c.min()), sec=time.time() - t0, lp=p.lp_stats)
    out["bangai04"] = dict(turns=n0, cmin_torque=float(c0[:8].min()), cmin=float(c0.min()))
    # 手を胸で組む姿勢（番外編 03 の tuck）を保つのに、接触ありで筋肉のトルクはいくら要るか（モデルの確認）
    print(json.dumps(out, indent=1, default=float))
    json.dump(out, open(DATA / "bangai08_check.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "run":
        run(sys.argv[2], int(sys.argv[3]))
    elif mode == "check":
        check()
