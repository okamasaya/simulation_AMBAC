"""番外編 04: 肩の可動域と筋力の上限を入れて、腕の軌道を「答えを見ずに」最適化する（ブラインド解析）.

番外編 03（docs/16）の結論は「難しさは肩の力と可動域」。ここではその上限を文献から先に決めて記録し（docs/17 の 17.1〜17.4、
このスクリプトと一緒に最適化の前にコミット）、そのあとで腕の軌道を自由な形で最適化する。

ブラインドにする点
  - 初期値は可動域の中の一様乱数だけ。番外編 03 の姿勢（胸で組む・頭上で合わせる・片腕ずつ…）は使わない
  - 目的は「回転数を最大に」だけ。腕をどこに置くかは何も指定しない
  - 滑らかさの指標・判定の決まりは、最適化の前に固定する（下の smoothness と docs/17.4）
  - 全部の条件を回し終えるまで、軌道の中身（どこに腕があるか）は見ない（ログは数値だけ）

腕の軌道: 各関節（肩 3 軸＋肘）の角度を 0.07 s ごとの節点の値で表し、clamped の 3 次スプラインでつなぐ（t = 0 と t_air で open、速度 0）。
主の解析は左右対称（右腕は左腕の鏡像）: 4 関節 × 9 節点 = 36 変数。
条件
  limited   : 可動域と筋力の上限（文献、60°/s の求心性）
  free      : 可動域と筋力の上限なし（体との交差の禁止と着氷の傾きの制約だけは残す）
  weak      : 筋力 × 0.7（180°/s 程度の速さでの低下）
  strong    : 筋力 × 1.3（鍛えた選手）
  rom180    : 肩の屈曲の上限を 169° → 180°（教科書の値）
実行: python bangai04_shoulder_blind.py [run COND SEED|all|metrics|figure]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

import bangai03_axel_arms as A
from common import DATA, ROOT

OUT = ROOT / "figures" / "field"
D = np.radians
MASS = A.REF["mass"]

# ---------------------------------------------------------------- 17.2 上限（事前登録。docs/17 の表と同じ）
# トルク: 体重あたり [N·m/kg] × 62 kg。正負は MuJoCo の関節の向き（左腕）で、右腕は sr の向きが反対。
TQ_PER_KG = dict(
    flex=1.14,   # 肩の屈曲  Lategan 2011（438 人のうち 116 人、60°/s、1.14 N·m/kg）
    ext=1.23,    # 肩の伸展  同上（1.23 N·m/kg）
    abd=67.1 / 76.52,   # 肩の外転  Coetzee ら 2002（20 人、60°/s、67.1 N·m、体重 76.5 kg）
    add=125.7 / 76.52,  # 肩の内転  同上（125.7 N·m）
    er=0.56,     # 肩の外旋  Lategan 2011（0.56 N·m/kg）
    ir=0.72,     # 肩の内旋  同上（0.72 N·m/kg）
    elf=50.8 / 74.2,    # 肘の屈曲  Papathanasiou ら（40 人、60°/s、肩の外転 45°、50.8 N·m、74.2 kg）
    ele=49.85 / 74.2,   # 肘の伸展  同上（49.85 N·m）
)
# 可動域 [deg]（MuJoCo の関節の座標。sf は負が屈曲、sa は正が外転、sr は左腕で正が外旋、el は正が屈曲）
ROM = dict(
    flex=168.8,  # 肩の屈曲  Soucie ら 2011（CDC、男性 20〜44 歳、他動）
    ext=60.0,    # 肩の伸展  AAOS の教科書の値
    abd=180.0,   # 肩の外転  同上
    add=30.0,    # 肩の内転（前額面で体の前を横切る分）  仮定（体と交差しない範囲は別の制約）
    er=90.0, ir=70.0,   # 肩の外旋・内旋  AAOS
    elf=144.6,   # 肘の屈曲  Soucie ら 2011
    ele=0.0,     # 肘の伸展（過伸展なし）  Soucie ら 2011（0.8°）
)
L_NOM, T_AIR = A.L_NOM, A.T_AIR_NOM
KNOT_DT = 0.07
TILT_LAND, TILT_MAX = 10.0, 20.0     # 着氷の傾き ≤ 10°、空中の傾き ≤ 20°（King らの 3A の空中の傾き 2〜22°）
CLEAR = 0.04                          # 腕の太さ（半径）

CONDS = dict(
    limited=dict(tq=1.0, rom=True, flex=None),
    free=dict(tq=None, rom=False, flex=None),
    weak=dict(tq=0.7, rom=True, flex=None),
    strong=dict(tq=1.3, rom=True, flex=None),
    rom180=dict(tq=1.0, rom=True, flex=180.0),
)
SEEDS = (11, 12, 13, 14)


def limits(cond):
    c = CONDS[cond]
    tq = None if c["tq"] is None else {k: v * MASS * c["tq"] for k, v in TQ_PER_KG.items()}
    rom = dict(ROM)
    if c["flex"]:
        rom["flex"] = c["flex"]
    # 関節の座標での [下限, 上限]（rad）
    lo = np.array([-D(rom["flex"]), -D(rom["add"]), -D(rom["ir"]), D(rom["ele"])])
    hi = np.array([D(rom["ext"]), D(rom["abd"]), D(rom["er"]), D(rom["elf"])])
    return tq, lo, hi, c["rom"]


class Problem:
    def __init__(self, cond, dt=2e-3, knot_dt=KNOT_DT):
        self.cond = cond
        self.B = A.load_body(A.REF["body"], A.REF["height"], A.REF["mass"], A.REF["boot"])
        self.sk = A.Skater(self.B)
        self.dt = dt
        self.knots = np.arange(knot_dt, T_AIR - 1e-9, knot_dt)
        self.knots = self.knots[self.knots < T_AIR - 0.02]
        self.nk = len(self.knots)
        self.q_open = self.sk.poses["L"]["open"]
        self.tq, self.lo, self.hi, self.use_rom = limits(cond)
        S = self.B["seg"]
        self.zc7 = self.B["H"] - S["head"]["len"] - self.B["hipH"]
        self.zsh = self.zc7 - self.B["sh_below"]
        self.zhead = self.zc7 + 0.5 * S["head"]["len"]
        self.cache = {}
        self.nev = 0

    # ---- 変数 ↔ 軌道
    def spline(self, x):
        V = x.reshape(4, self.nk)
        ts = np.r_[0.0, self.knots, T_AIR]
        Q = np.c_[self.q_open, V, self.q_open]          # (4, nk+2)
        return CubicSpline(ts, Q.T, bc_type="clamped")

    def plan(self, x):
        cs = self.spline(x)
        d1, d2 = cs.derivative(1), cs.derivative(2)
        mir = np.array([1, 1, -1, 1.0])
        def f(t):
            t = min(max(t, 0.0), T_AIR)
            q, qd, qdd = cs(t), d1(t), d2(t)
            return np.r_[q, q * mir], np.r_[qd, qd * mir], np.r_[qdd, qdd * mir]
        return f

    def points(self, q):
        """左腕の上腕・前腕・手の先の点（胴体の座標）."""
        e, w = self.sk._fk("L", q)
        sh = np.array([0, self.B["wsh"], self.zsh])
        S = self.B["seg"]
        hand = w + (w - e) / np.linalg.norm(w - e) * 0.6 * S["hand"]["len"]
        pts = [sh + s * e for s in (0.4, 0.7, 1.0)] + [sh + e + s * (w - e) for s in (0.33, 0.67, 1.0)] + [sh + hand]
        return np.array(pts)

    def clearance(self, q):
        """体（胴体のカプセル、頭の球）からの距離の最小 − 腕の太さ. 負なら体に食い込む."""
        P = self.points(q)
        z = np.clip(P[:, 2], 0.0, self.zc7)
        d_trunk = np.hypot(P[:, 0], P[:, 1]) - self.B["chest"] - CLEAR
        d_trunk = np.where((P[:, 2] >= 0) & (P[:, 2] <= self.zc7), d_trunk,
                           np.linalg.norm(P - np.c_[0 * z, 0 * z, z], axis=1) - self.B["chest"] - CLEAR)
        d_head = np.linalg.norm(P - np.array([0, 0, self.zhead]), axis=1) - 0.10 - CLEAR
        return float(min(d_trunk.min(), d_head.min()))

    # ---- 1 回の評価
    def evaluate(self, x):
        key = x.tobytes()
        if key in self.cache:
            return self.cache[key]
        self.nev += 1
        r = self.sk.fly(None, L_NOM, T_AIR, dt=self.dt, plan=self.plan(x), tau_samples=True)
        tau = r["tau_all"][:, :4]          # 左腕（右は鏡像なので同じ大きさ）
        cs = self.spline(x)
        ts = np.linspace(0, T_AIR, 141)
        Q = cs(ts)
        cons = []
        if self.tq is not None:
            T = self.tq
            cons += [T["ext"] - tau[:, 0].max(), T["flex"] + tau[:, 0].min(),
                     T["abd"] - tau[:, 1].max(), T["add"] + tau[:, 1].min(),
                     T["er"] - tau[:, 2].max(), T["ir"] + tau[:, 2].min(),
                     T["elf"] - tau[:, 3].max(), T["ele"] + tau[:, 3].min()]
            cons = [c / 10.0 for c in cons]               # 10 N·m を 1 に
        if self.use_rom:
            cons += list(np.degrees(Q.min(0) - self.lo) / 10.0) + list(np.degrees(self.hi - Q.max(0)) / 10.0)
        clr = min(self.clearance(q) for q in Q[::2])
        cons += [clr / 0.02, (TILT_LAND - r["tilt_land"]) / 2.0, (TILT_MAX - r["tilt_max"]) / 2.0]
        out = (r["turns"], np.array(cons), r)
        self.cache[key] = out
        return out

    def solve(self, seed, maxiter=60):
        rng = np.random.default_rng(seed)
        lo = self.lo if self.use_rom else np.array([-np.pi, -np.pi / 2, -np.pi / 2, 0.0])
        hi = self.hi if self.use_rom else np.array([np.pi / 3, np.pi, np.pi / 2, D(150)])
        # 初期値: 可動域の中の一様乱数のうち、体と交差しないもの（既知の姿勢は使わない）
        for _ in range(500):
            x0 = (lo[:, None] + (hi - lo)[:, None] * rng.uniform(size=(4, self.nk))).ravel()
            cs = self.spline(x0)
            if min(self.clearance(q) for q in cs(np.linspace(0, T_AIR, 71))) > 0:
                break
        h = 1e-3
        def fg(x):
            n, c, _ = self.evaluate(x)
            return n, c
        def jac_all(x):
            n0, c0 = fg(x)
            gn = np.zeros_like(x); gc = np.zeros((len(c0), len(x)))
            for i in range(len(x)):
                xp = x.copy(); xp[i] += h
                n1, c1 = fg(xp)
                gn[i] = (n1 - n0) / h; gc[:, i] = (c1 - c0) / h
            return gn, gc
        jc = {}
        def J(x):
            k = x.tobytes()
            if k not in jc:
                jc.clear(); jc[k] = jac_all(x)
            return jc[k]
        t0 = time.time()
        hist = []
        def cb(x):
            n, c, _ = self.evaluate(x)
            hist.append([n, float(c.min())])
        res = minimize(lambda x: -fg(x)[0], x0, jac=lambda x: -J(x)[0], method="SLSQP",
                       constraints=[dict(type="ineq", fun=lambda x: fg(x)[1], jac=lambda x: J(x)[1])],
                       options=dict(maxiter=maxiter, ftol=1e-5))
        n, c, r = self.evaluate(res.x)
        return dict(cond=self.cond, seed=seed, x=res.x.tolist(), x0=x0.tolist(), turns=n, cmin=float(c.min()),
                    feasible=bool(c.min() >= -1e-3), nit=int(res.nit), msg=str(res.message), nev=self.nev,
                    sec=time.time() - t0, hist=hist, tilt_land=r["tilt_land"], tilt_max=r["tilt_max"])


# ---------------------------------------------------------------- 17.4 滑らかさの指標（事前登録）
def sparc(speed, fs, padlevel=4, fc=10.0, amp_th=0.05):
    """Spectral arc length（Balasubramanian ら 2015）. 0 に近いほど滑らか（負の値）."""
    n = int(2 ** (np.ceil(np.log2(len(speed))) + padlevel))
    f = np.arange(0, fs, fs / n)
    M = np.abs(np.fft.fft(speed, n))
    M = M / M.max()
    sel = f <= fc
    f, M = f[sel], M[sel]
    idx = np.nonzero(M >= amp_th)[0]
    f, M = f[idx[0]:idx[-1] + 1], M[idx[0]:idx[-1] + 1]
    df = np.diff(f) / (f[-1] - f[0])
    dM = np.diff(M)
    return float(-np.sum(np.sqrt(df**2 + dM**2)))


def ldlj(pos, fs):
    """Log dimensionless jerk（Balasubramanian ら 2015、速さの最大で正規化）. 0 に近いほど滑らか（負の値）."""
    dt = 1 / fs
    v = np.gradient(pos, dt, axis=0)
    j = np.gradient(np.gradient(v, dt, axis=0), dt, axis=0)
    T = (len(pos) - 1) * dt
    vpk = np.linalg.norm(v, axis=1).max()
    return float(-np.log(T**3 / vpk**2 * np.trapezoid(np.sum(j**2, axis=1), dx=dt)))


def smoothness(prob, x):
    cs = prob.spline(np.asarray(x))
    fs = 1000.0
    ts = np.arange(0, T_AIR + 1e-9, 1 / fs)
    Q = cs(ts)
    W = np.array([prob.points(q)[-2] for q in Q])        # 手首（胴体の座標）
    v = np.linalg.norm(np.gradient(W, 1 / fs, axis=0), axis=1)
    qd = cs.derivative(1)(ts)
    rev = int(sum(np.sum(np.diff(np.sign(qd[:, j][np.abs(qd[:, j]) > 0.2])) != 0) for j in range(4)))
    return dict(sparc=sparc(v, fs), ldlj=ldlj(W, fs), reversals=rev,
                path=float(np.sum(np.linalg.norm(np.diff(W, axis=0), axis=1))))


# ---------------------------------------------------------------- 集計（17.4 の判定）と図
def classify(prob, x):
    cs = prob.spline(np.asarray(x))
    ts = np.linspace(0.25, 0.45, 21)
    W = np.array([prob.points(q)[-2] for q in cs(ts)])
    up = W[:, 2] - prob.zsh
    rad = np.hypot(W[:, 0], W[:, 1])
    if np.median(up) >= 0.3:
        return "頭上"
    if np.median(up) < 0 and np.median(rad) <= 0.25:
        return "締め"
    return "その他"


def metrics():
    res = {}
    for c in CONDS:
        p = Problem(c)
        rows = []
        for s in SEEDS:
            f = DATA / "bangai04" / f"{c}_{s}.json"
            if not f.exists():
                continue
            r = json.load(open(f))
            r["shape"] = classify(p, r["x"])
            # 上限（limited）での可否も見る（free などの解が人の力で実行できるか）
            if c != "limited":
                pl = Problem("limited")
                _, cl, _ = pl.evaluate(np.asarray(r["x"]))
                r["feasible_under_limited"] = bool(cl.min() >= -1e-3)
                r["cmin_under_limited"] = float(cl.min())
            rows.append(r)
        res[c] = rows
    summ = {}
    for c, rows in res.items():
        ok = [r for r in rows if r["feasible"]]
        def ms(k):
            v = np.array([r[k] for r in ok])
            return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan"), len(v))
        summ[c] = dict(n=len(rows), n_ok=len(ok), turns=ms("turns"), sparc=ms("sparc"), ldlj=ms("ldlj"),
                       rev=ms("reversals"), best=max(ok, key=lambda r: r["turns"])["seed"] if ok else None,
                       shapes=[r["shape"] for r in rows])
    # 判定（limited 対 free）
    verdict = {}
    for k in ("sparc", "ldlj"):
        a, b = summ["limited"][k], summ["free"][k]
        diff = a[0] - b[0]
        sd = float(np.hypot(a[1], b[1]))
        verdict[k] = dict(diff=diff, sd=sd, ratio=diff / sd if sd > 0 else float("nan"))
    ok = [verdict[k]["ratio"] > 2 for k in ("sparc", "ldlj")]
    neg = [verdict[k]["diff"] < 0 for k in ("sparc", "ldlj")]
    verdict["H1"] = "支持" if all(ok) else "弱い支持" if any(ok) else "反する" if all(neg) else "区別できない"
    out = dict(summary=summ, verdict=verdict,
               rows={c: [{k: v for k, v in r.items() if k not in ("x", "x0", "hist")} for r in rows] for c, rows in res.items()})
    json.dump(out, open(DATA / "bangai04_summary.json", "w"), indent=1, ensure_ascii=False, default=float)
    for c, sm in summ.items():
        print(f"{c:8s} 可 {sm['n_ok']}/{sm['n']}  回転 {sm['turns'][0]:.3f}±{sm['turns'][1]:.3f}  SPARC {sm['sparc'][0]:.3f}±{sm['sparc'][1]:.3f}  "
              f"LDLJ {sm['ldlj'][0]:.2f}±{sm['ldlj'][1]:.2f}  反転 {sm['rev'][0]:.1f}  形 {sm['shapes']}")
    print("判定:", json.dumps(verdict, ensure_ascii=False, default=lambda x: round(float(x), 3)))
    for c, rows in res.items():
        for r in rows:
            extra = f"  limited で可 {r['feasible_under_limited']} ({r['cmin_under_limited']:+.2f})" if "feasible_under_limited" in r else ""
            print(f"  {c:8s} {r['seed']}: {r['turns']:.3f} 可 {r['feasible']} 形 {r['shape']}{extra}")
    return out


def comp_check():
    """計算の誤差: 最良の limited の解を 1 ms で計算し直す・節点の値を 1e-3 rad 揺らす."""
    s = json.load(open(DATA / "bangai04_summary.json"))["summary"]["limited"]["best"]
    r = json.load(open(DATA / "bangai04" / f"limited_{s}.json"))
    x = np.asarray(r["x"])
    p1 = Problem("limited", dt=1e-3)
    n1, c1, _ = p1.evaluate(x)
    p2 = Problem("limited")
    n2, c2, _ = p2.evaluate(x)
    out = dict(seed=s, turns_2ms=n2, turns_1ms=n1, cmin_2ms=float(c2.min()), cmin_1ms=float(c1.min()))
    print(out)
    json.dump(out, open(DATA / "bangai04_comp.json", "w"), indent=1, default=float)


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    S = json.load(open(DATA / "bangai04_summary.json"))
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    cols = dict(limited="C0", free="C3", weak="C7", strong="C2", rom180="C4")
    jp = dict(limited="上限あり", free="上限なし", weak="筋力 ×0.7", strong="筋力 ×1.3", rom180="屈曲 180°")
    # (a) 滑らかさ: SPARC と LDLJ（種ごと）
    ax = axs[0, 0]
    for i, c in enumerate(CONDS):
        for r in S["rows"][c]:
            ax.plot(r["sparc"], r["ldlj"], "o" if r["feasible"] else "x", color=cols[c], ms=8,
                    label=jp[c] if r is S["rows"][c][0] else None)
    ax.set_xlabel("SPARC（0 に近いほど滑らか）"); ax.set_ylabel("LDLJ（0 に近いほど滑らか）")
    ax.set_title("(a) 手首の動きの滑らかさ（○ 制約を満たす、× 満たさない）"); ax.legend(fontsize=8)
    # (b) 回転数
    ax = axs[0, 1]
    for i, c in enumerate(CONDS):
        v = [r["turns"] for r in S["rows"][c]]
        ax.scatter([i] * len(v), v, color=cols[c])
    ax.axhline(3.43, color="k", ls="--", lw=1); ax.text(0, 3.45, "胸で組む普通の 3A（番外編 03）", fontsize=8)
    ax.set_xticks(range(len(CONDS))); ax.set_xticklabels([jp[c] for c in CONDS])
    ax.set_ylabel("回転数"); ax.set_title("(b) 最大にした回転数")
    # (c)〜(f) 最良の解の関節角と手首の軌跡（limited と free）
    for j, c in enumerate(("limited", "free")):
        best = S["summary"][c]["best"]
        if best is None:
            continue
        r = json.load(open(DATA / "bangai04" / f"{c}_{best}.json"))
        p = Problem(c)
        cs = p.spline(np.asarray(r["x"]))
        ts = np.linspace(0, T_AIR, 351)
        Q = np.degrees(cs(ts))
        ax = axs[0, 2] if j == 0 else axs[1, 2]
        for k, nm in enumerate(("肩 sf（負 = 屈曲）", "肩 sa（正 = 外転）", "肩 sr（正 = 外旋）", "肘（正 = 屈曲）")):
            ax.plot(ts, Q[:, k], label=nm)
        ax.set_xlabel("時間 [s]"); ax.set_ylabel("関節角 [°]"); ax.legend(fontsize=7)
        ax.set_title(f"({'c' if j == 0 else 'f'}) {jp[c]} の最良の解（種 {best}、{r['turns']:.2f} 回転）")
        ax = axs[1, j]
        W = np.array([p.points(q)[-2] for q in cs(ts)])
        sc = ax.scatter(W[:, 1], W[:, 2] - p.zsh, c=ts, cmap="viridis", s=6)
        ax.add_patch(plt.Circle((0, p.zhead - p.zsh), 0.10, fill=False))
        ax.plot([-p.B["chest"], -p.B["chest"], p.B["chest"], p.B["chest"]], [-p.zsh, 0.05, 0.05, -p.zsh], "k-", lw=1)
        ax.plot(p.B["wsh"], 0, "ks")
        ax.set_aspect("equal"); ax.set_xlabel("左右 [m]"); ax.set_ylabel("肩からの高さ [m]")
        ax.set_title(f"({'d' if j == 0 else 'e'}) 左の手首の軌跡（正面から、色は時間）: {jp[c]}")
        plt.colorbar(sc, ax=ax, label="時間 [s]")
    fig.tight_layout(); fig.savefig(OUT / "bangai04_shoulder_blind.png", dpi=130)
    print("図 → figures/field/bangai04_shoulder_blind.png")


# ---------------------------------------------------------------- 実行
def run(cond, seed):
    p = Problem(cond)
    r = p.solve(seed)
    r.update(smoothness(p, r["x"]))
    f = DATA / "bangai04" / f"{cond}_{seed}.json"
    f.parent.mkdir(exist_ok=True)
    json.dump(r, open(f, "w"), default=float)
    # ブラインド: ここでは数値だけを出し、腕の位置は出さない
    print(f"{cond:8s} seed {seed}: {r['turns']:.3f} 回転  可否 {r['feasible']}  最小の余裕 {r['cmin']:+.3f}  "
          f"SPARC {r['sparc']:.3f}  LDLJ {r['ldlj']:.2f}  反転 {r['reversals']}  反復 {r['nit']}  {r['sec']:.0f} s", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "run":
        run(sys.argv[2], int(sys.argv[3]))
    elif mode == "metrics":
        metrics()
    elif mode == "comp":
        comp_check()
    elif mode == "figure":
        figure()
    elif mode == "all":
        for c in CONDS:
            for s in SEEDS:
                run(c, s)
