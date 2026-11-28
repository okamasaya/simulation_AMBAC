"""番外編 07: 踏み切りの外乱を、空中の腕の使い方でどこまで打ち消せるか（頑健性、事前登録のブラインド検証）.

ユーザーの着想: 桑田投手のリリースポイント（外乱を吸収し、最後に微調整）、武豊騎手のぶれないジョッキーカメラ。
着目点は「制御の方法、体の止め方・固定の仕方」。

外乱（踏み切りで決まり、空中では変えられない）
  δL:   角運動量の大きさのずれ          N(0, σ_L)、σ_L = 1.0（主）, 2.0（強）
  δt:   滞空時間のずれ（跳んだ高さ）    N(0, 0.02 s)
  θ0:   L の向きの鉛直からの傾き        |N(0, σ_θ)|、方位は一様、σ_θ = 3°（主）, 6°（強）

腕の制御（空中だけ。胴体の動きは番外編 03 と同じく保存則から MuJoCo で解く）
  S0  開ループ: いつも同じ締め（tuck）。着氷の 0.12 s 前に開く（着氷の時刻は体で分かるとする）
  S1  最後の微調整（桑田型）: t = 0.30 s で、その時点の状態から開く時刻 t_o を選び直す
  S2  早い調整: 踏み切りの直後（t = 0）に締めの強さ λ ∈ [0.4, 1] を選ぶ（開く時刻は S0 と同じ）
  S3  S1 + 片腕の非対称（片腕を頭上の方向へ量 a、時刻 t_a から）で軸の傾きも直す
選び方: S1・S2 は選手の「体の中のモデル」として準静的な式（番外編 06）で回転数を合わせる。
S3 の非対称は格子（腕 2 × 量 4 × 時刻 3）を MuJoCo で試して一番よいもの（理想の上限。選手が本当にできるかは別）。

判定（事前登録、docs/20 の 20.2）
  回転:  |N − 3.43| < 0.25（ISU の 1/4）
  傾き:  着氷時の胴体の軸の傾き < 10°（感度 5°, 15°）
  成功:  両方

使い方: python bangai07_robust.py [check|run LEVEL [n] [seed]|summary|figure]
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool

import numpy as np

import bangai03_axel_arms as A
from common import DATA, ROOT

OUT = ROOT / "figures" / "field"
N_TGT = A.N_AIR_NOM
ROT_TOL, TILT_TOL = 0.25, 10.0
TILT_SENS = (5.0, 10.0, 15.0)
T_TR = A.T_TR
T_OPEN_TR = 0.10                 # 開く動きの長さ（番外編 03 と同じ 0.10 s）
LEAD = 0.12                      # 着氷の何秒前に開き始めるか（S0）
T_C = 0.30                       # S1 の判断の時刻
LEVELS = {"main": dict(sL=1.0, sT=0.02, sth=3.0), "strong": dict(sL=2.0, sT=0.02, sth=6.0)}
STRATS = ("S0", "S1", "S2", "S3")
ASYM_GRID = [(side, a, ta) for side in "LR" for a in (0.25, 0.5, 0.75, 1.0) for ta in (0.25, 0.35, 0.45)]
DT = 2e-3                        # check で 1 ms と比べる

_SK = None


def skater():
    global _SK
    if _SK is None:
        B = A.load_body(A.REF["body"], A.REF["height"], A.REF["mass"], A.REF["boot"])
        _SK = A.Skater(B)
        for p in ("tuck", "up", "open"):
            _SK.add_pose(p)
    return _SK


def _step(t, t0, T):
    s = np.clip((t - t0) / T, 0.0, 1.0)
    inside = 0.0 < (t - t0) / T < 1.0
    S = 10 * s**3 - 15 * s**4 + 6 * s**5
    dS = (30 * s**2 - 60 * s**3 + 30 * s**4) / T if inside else 0.0
    ddS = (60 * s - 180 * s**2 + 120 * s**3) / T**2 if inside else 0.0
    return S, dS, ddS


def make_plan(sk, lam=1.0, t_o=None, t_air=A.T_AIR_NOM, asym=None):
    """腕の関節角の時間関数. λ(t): 0 → lam（t_tr で）→ 0（t_o から 0.10 s）. asym = (side, a, t_a): 片腕だけ tuck → up へ a だけ."""
    t_o = t_air - LEAD if t_o is None else t_o
    P = sk.poses
    qo = {s: P[s]["open"] for s in "LR"}
    qt = {s: P[s]["tuck"] for s in "LR"}
    qu = {s: P[s]["up"] for s in "LR"}

    def f(t):
        S1, d1, dd1 = _step(t, 0.0, T_TR)
        S2, d2, dd2 = _step(t, t_o, T_OPEN_TR)
        lamt = lam * S1 * (1 - S2)
        dl = lam * (d1 * (1 - S2) - S1 * d2)
        ddl = lam * (dd1 * (1 - S2) - 2 * d1 * d2 - S1 * dd2)
        out = []
        for s in "LR":
            q = qo[s] + lamt * (qt[s] - qo[s])
            dq = dl * (qt[s] - qo[s])
            ddq = ddl * (qt[s] - qo[s])
            if asym is not None and asym[0] == s:
                side, a, ta = asym
                S3, d3, dd3 = _step(t, ta, T_TR)
                # 片腕の上げは、開くときに一緒に戻る（(1 − S2) を掛ける）
                m, dm, ddm = a * S3 * (1 - S2), a * (d3 * (1 - S2) - S3 * d2), a * (dd3 * (1 - S2) - 2 * d3 * d2 - S3 * dd2)
                q = q + m * (qu[s] - qt[s]); dq = dq + dm * (qu[s] - qt[s]); ddq = ddq + ddm * (qu[s] - qt[s])
            out.append((q, dq, ddq))
        return tuple(np.r_[out[0][k], out[1][k]] for k in range(3))
    return f


# ---------------------------------------------------------------- 選手の「体の中のモデル」（準静的）
_ICURVE = None
LAM = np.linspace(0, 1, 41)


def icurve():
    global _ICURVE
    if _ICURVE is None:
        sk = skater()
        qo, qor = sk.poses["L"]["open"], sk.poses["R"]["open"]
        ql, qr = sk.poses["L"]["tuck"], sk.poses["R"]["tuck"]
        _ICURVE = np.array([sk.inertia(None, qL=qo + (ql - qo) * l, qR=qor + (qr - qor) * l)[0][2, 2] for l in LAM])
    return _ICURVE


def qs_turns(L, t_air, lam=1.0, t_o=None, dt=1e-3):
    t_o = t_air - LEAD if t_o is None else t_o
    t = np.arange(0, t_air + dt / 2, dt)
    s1 = np.array([_step(x, 0, T_TR)[0] for x in t])
    s2 = np.array([_step(x, t_o, T_OPEN_TR)[0] for x in t])
    I = np.interp(lam * s1 * (1 - s2), LAM, icurve())
    return float(np.trapezoid(L / I, t) / (2 * np.pi))


def choose_t_o(L, t_air):
    """S1: 開く時刻を、回転数が目標になるように選ぶ（範囲は T_C + 0.05 〜 t_air − 0.10。届かなければ端）."""
    lo, hi = T_C + 0.05, t_air - T_OPEN_TR
    f = lambda x: qs_turns(L, t_air, t_o=x) - N_TGT
    if f(lo) >= 0:
        return lo
    if f(hi) <= 0:
        return hi
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def choose_lam(L, t_air):
    """S2: 締めの強さを選ぶ（0.4〜1。回転を増やす方向には締め以上はできない）."""
    lo, hi = 0.4, 1.0
    f = lambda x: qs_turns(L, t_air, lam=x) - N_TGT
    if f(hi) <= 0:
        return hi
    if f(lo) >= 0:
        return lo
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- 1 回の試行
def fly(L, t_air, th, ph, lam=1.0, t_o=None, asym=None, dt=DT):
    sk = skater()
    Lvec = L * np.array([np.sin(np.radians(th)) * np.cos(ph), np.sin(np.radians(th)) * np.sin(ph), np.cos(np.radians(th))])
    plan = make_plan(sk, lam, t_o, t_air, asym)
    r = sk.fly(None, L, t_air, dt=dt, torque=False, plan=plan, Lvec=Lvec)
    return dict(N=r["turns"], tilt=r["tilt_land"])


def trial(args):
    i, L, t_air, th, ph = args
    out = dict(i=i, L=L, t_air=t_air, th=th, ph=ph)
    r = fly(L, t_air, th, ph)
    out["S0"] = dict(r, t_o=t_air - LEAD, lam=1.0)
    t_o = choose_t_o(L, t_air)
    r1 = fly(L, t_air, th, ph, t_o=t_o)
    out["S1"] = dict(r1, t_o=t_o, lam=1.0)
    lam = choose_lam(L, t_air)
    out["S2"] = dict(fly(L, t_air, th, ph, lam=lam), t_o=t_air - LEAD, lam=lam)
    best = dict(r1, t_o=t_o, lam=1.0, asym=None)
    cost = lambda r: r["tilt"] / TILT_TOL + abs(r["N"] - N_TGT) / ROT_TOL
    for g in ASYM_GRID:
        if g[2] >= t_o:
            continue
        r3 = fly(L, t_air, th, ph, t_o=t_o, asym=g)
        if cost(r3) < cost(best):
            best = dict(r3, t_o=t_o, lam=1.0, asym=list(g))
    out["S3"] = best
    return out


def run(level="main", n=200, seed=11):
    cfg = LEVELS[level]
    rng = np.random.default_rng(seed)
    L = A.L_NOM + cfg["sL"] * rng.standard_normal(n)
    ta = A.T_AIR_NOM + cfg["sT"] * rng.standard_normal(n)
    th = np.abs(cfg["sth"] * rng.standard_normal(n))
    ph = rng.uniform(0, 2 * np.pi, n)
    args = [(i, float(L[i]), float(ta[i]), float(th[i]), float(ph[i])) for i in range(n)]
    t0 = time.time()
    with Pool(4) as p:
        res = []
        for k, r in enumerate(p.imap(trial, args, chunksize=2)):
            res.append(r)
            if (k + 1) % 20 == 0:
                print(f"{k + 1}/{n}  {time.time() - t0:.0f} s", flush=True)
    json.dump(dict(level=level, cfg=cfg, n=n, seed=seed, trials=res), open(DATA / f"bangai07_{level}.json", "w"), indent=1)
    summary(level)


def check():
    """計算の誤差: dt 2 ms と 1 ms、体の中のモデル（準静的）と MuJoCo の差."""
    out = {}
    for dt in (2e-3, 1e-3):
        out[f"dt{dt}"] = fly(A.L_NOM, A.T_AIR_NOM, 0.0, 0.0, dt=dt)
    out["qs"] = qs_turns(A.L_NOM, A.T_AIR_NOM)
    for L in (30.8, 34.8):
        t_o = choose_t_o(L, A.T_AIR_NOM)
        out[f"S1_L{L}"] = dict(fly(L, A.T_AIR_NOM, 0, 0, t_o=t_o), t_o=t_o)
    out["tilt5"] = fly(A.L_NOM, A.T_AIR_NOM, 5.0, 0.0)
    print(json.dumps(out, indent=1))
    json.dump(out, open(DATA / "bangai07_check.json", "w"), indent=1)


def _arr(T, s, k):
    return np.array([t[s][k] for t in T])


def summary(level="main"):
    D = json.load(open(DATA / f"bangai07_{level}.json"))
    T = D["trials"]
    n = len(T)
    out = {}
    print(f"[{level}] n = {n}  {D['cfg']}")
    dL = np.array([t["L"] for t in T]) - A.L_NOM
    for s in STRATS:
        dN = _arr(T, s, "N") - N_TGT
        tl = _arr(T, s, "tilt")
        rot = np.abs(dN) < ROT_TOL
        o = dict(dN_mean=float(dN.mean()), dN_sd=float(dN.std(ddof=1)), dN_rms=float(np.sqrt(np.mean(dN**2))),
                 tilt_mean=float(tl.mean()), tilt_q95=float(np.percentile(tl, 95)),
                 p_rot=float(rot.mean()), t_o_sd=float(_arr(T, s, "t_o").std(ddof=1)),
                 r_to_dL=float(np.corrcoef(_arr(T, s, "t_o"), dL)[0, 1]) if _arr(T, s, "t_o").std() > 0 else float("nan"))
        for tol in TILT_SENS:
            ok = rot & (tl < tol)
            p = ok.mean()
            o[f"p_success_{tol:g}"] = float(p)
            o[f"p_success_{tol:g}_se"] = float(np.sqrt(p * (1 - p) / n))
        out[s] = o
        print(f"{s}  ΔN {o['dN_mean']:+.3f} ± {o['dN_sd']:.3f}  傾き 平均 {o['tilt_mean']:.1f}° 95% {o['tilt_q95']:.1f}°  "
              f"回転 OK {o['p_rot'] * 100:.0f}%  成功(10°) {o['p_success_10'] * 100:.1f} ± {o['p_success_10_se'] * 100:.1f}%  "
              f"開く時刻の SD {o['t_o_sd'] * 1e3:.1f} ms  r(t_o, δL) {o['r_to_dL']:+.2f}")
    # 傾きの打ち消し: S3 が S1 から減らした傾き、初期の傾きとの関係
    th = np.array([t["th"] for t in T])
    red = _arr(T, "S1", "tilt") - _arr(T, "S3", "tilt")
    out["tilt_reduction"] = dict(mean=float(red.mean()), max=float(red.max()),
                                 slope_vs_th=float(np.polyfit(th, _arr(T, "S3", "tilt"), 1)[0]))
    print("S3 で減った傾き", {k: round(v, 2) for k, v in out["tilt_reduction"].items()})
    json.dump(out, open(DATA / f"bangai07_{level}_summary.json", "w"), indent=1)
    return out


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    names = {"S0": "S0 開ループ", "S1": "S1 最後の微調整", "S2": "S2 早い調整", "S3": "S1 + 片腕で傾き"}
    T = json.load(open(DATA / "bangai07_main.json"))["trials"]
    dL = np.array([t["L"] for t in T]) - A.L_NOM
    ax = axs[0]
    for k, s in enumerate(("S0", "S1", "S2")):
        ax.scatter(dL, _arr(T, s, "N") - N_TGT, s=8, color=f"C{k}", label=names[s])
    ax.axhspan(-ROT_TOL, ROT_TOL, color="0.92", zorder=0)
    ax.set_xlabel("踏み切りの L のずれ [kg m²/s]"); ax.set_ylabel("回転数のずれ（着氷）")
    ax.set_title("(a) 回転数: 最後に開く時刻で吸収する"); ax.legend(fontsize=8)
    ax = axs[1]
    ax.scatter(dL, _arr(T, "S0", "t_o") * 1e3, s=8, color="C0", label=names["S0"])
    ax.scatter(dL, _arr(T, "S1", "t_o") * 1e3, s=8, color="C1", label=names["S1"])
    ax.set_xlabel("踏み切りの L のずれ [kg m²/s]"); ax.set_ylabel("腕を開き始める時刻 [ms]")
    ax.set_title("(b)「リリースポイント」のばらつき: 開く時刻"); ax.legend(fontsize=8)
    ax = axs[2]
    th = np.array([t["th"] for t in T])
    for k, s in enumerate(("S1", "S3")):
        ax.scatter(th, _arr(T, s, "tilt"), s=8, color=f"C{k + 1 if k == 0 else 3}", label=names[s])
    ax.axhline(TILT_TOL, color="k", ls="--")
    ax.set_xlabel("踏み切りの L の傾き [°]"); ax.set_ylabel("着氷時の胴体の軸の傾き [°]")
    ax.set_title("(c) 軸の傾き: 腕でどこまで直せるか"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "bangai07_robust.png", dpi=130)
    print("図 → figures/field/bangai07_robust.png")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "run":
        run(sys.argv[2], *(int(x) for x in sys.argv[3:]))
    else:
        {"check": check, "summary": lambda: [summary(l) for l in LEVELS], "figure": figure}[mode]()
