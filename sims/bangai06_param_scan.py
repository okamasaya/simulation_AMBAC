"""番外編 06: 羽生結弦（2021）の論文の定性的な所見と矛盾しないパラメータの範囲（組み合わせの総当たり）.

論文には数値の結果がない（番外編 05、F76）。そこで、体格・角運動量 L・滞空時間・腕の形・腕の遷移時間を
組み合わせで振り、論文の所見を数値の判定に直したもの（下の C1, C2）を満たす組み合わせだけを残す。
判定の閾値は計算の前に決めて docs/19 の 19.2 に書いた（事前登録。番外編 04 と同じやり方）。

  C1 「3A を跳んで降りた（2 本、非常に安定）」: 空中の回転数 N が 3.43 ± 0.25（ISU の 1/4 回転）
  C2 「回転はしっかりと追えていた」: ジャイロ（±2000 °/s で頭打ち）の積分で数え落とす回転 < 0.125（45°）
  （着氷の膝・回転角が怪しい、高さが取れない、は 60 Hz・16 g・ドリフトの限界で、パラメータを絞らない）

速く回すため、腕を保持している間の胴体の角速度は ω = L / I(t)（準静的）で計算する。MuJoCo の飛行（番外編 05）と
check() で比べる。N も ω も L / I にしか依存しないので、s = L / (I_tuck / I_tuck,ref) が本当のパラメータ（縮退）。

使い方: python bangai06_param_scan.py [check|scan|figure]
"""
from __future__ import annotations

import json
import sys

import numpy as np

import bangai03_axel_arms as A
from common import DATA, ROOT, smooth_step

OUT = ROOT / "figures" / "field"
GYRO_LIM = 2000.0
N_NOM = A.N_AIR_NOM            # 3.43
C1_TOL = 0.25                  # 事前登録
C2_TOL = 0.125                 # 事前登録（感度: 0.05, 0.25）
SENS_C2 = (0.05, 0.125, 0.25)

POSES = {"tuck": ("tuck", "tuck"), "half": ("half", "half"), "side": ("side", "side"),
         "uppar": ("uppar", "uppar"), "up": ("up", "up"), "tano": ("up", "tuck")}
POSE_JP = {"tuck": "胸で組む", "half": "半分の締め", "side": "横に伸ばす", "uppar": "肩幅で真上",
           "up": "頭上で合わせる", "tano": "片手だけ頭上"}
HEIGHTS = np.round(np.arange(1.60, 1.801, 0.02), 3)
MASSES = np.arange(50.0, 70.01, 2.0)
L_GRID = np.arange(24.0, 42.01, 0.25)
T_AIR = np.round(np.arange(0.60, 0.801, 0.01), 3)
T_TR = (0.07, 0.10, 0.15)
LAM = np.linspace(0, 1, 41)


def izz_curves(sk):
    """腕を open → 各姿勢へ λ で動かしたときの I_zz(λ)."""
    out = {}
    for k, (pl, pr) in POSES.items():
        qo, qor = sk.poses["L"]["open"], sk.poses["R"]["open"]
        ql, qr = sk.poses["L"][pl], sk.poses["R"][pr]
        out[k] = np.array([sk.inertia(None, qL=qo + (ql - qo) * l, qR=qor + (qr - qor) * l)[0][2, 2] for l in LAM])
    return out


def profile(curve, t_air, t_tr, dt=1e-3):
    """時間の格子と 1/I(t)."""
    t = np.arange(0, t_air + dt / 2, dt)
    lam = smooth_step(t / t_tr)[0] * (1 - smooth_step((t - (t_air - A.T_OPEN)) / t_tr)[0])
    return t, 1.0 / np.interp(lam, LAM, curve)


def turns_and_loss(L, t, invI):
    """N と、ジャイロの頭打ちで数え落とす回転（L は配列可）."""
    w = np.degrees(np.outer(np.atleast_1d(L), invI))           # °/s
    N = np.trapezoid(w, t, axis=1) / 360
    loss = np.trapezoid(np.clip(w - GYRO_LIM, 0, None), t, axis=1) / 360
    return N, loss, w.max(axis=1)


def check():
    """準静的な式と MuJoCo の飛行（番外編 05 の結果）の比較."""
    B = A.load_body(A.REF["body"], A.REF["height"], A.REF["mass"], A.REF["boot"])
    sk = A.Skater(B)
    cur = izz_curves(sk)
    ref = json.load(open(DATA / "bangai05_sensor_check.json"))
    names = {"tuck": "胸で組む（普通の 3A）", "up": "頭上で手を合わせる", "uppar": "肩幅で真上", "half": "半分の締め"}
    out = {}
    for k, nm in names.items():
        t, invI = profile(cur[k], A.T_AIR_NOM, A.T_TR)
        N, loss, wmax = turns_and_loss(A.L_NOM, t, invI)
        r = ref[nm]
        out[k] = dict(N_qs=float(N[0]), N_mj=r["turns"], w_qs=float(wmax[0]), w_mj=r["wz_peak"],
                      loss_qs=float(loss[0]), loss_mj=-r["dturn_clip"])
        print(f"{k:6s} N {N[0]:.3f} / MuJoCo {r['turns']:.3f}   ω {wmax[0]:.0f} / {r['wz_peak']:.0f} °/s   "
              f"数え落とし {loss[0]:.3f} / {-r['dturn_clip']:.3f}")
    json.dump(out, open(DATA / "bangai06_check.json", "w"), indent=1)
    return out


def scan():
    """全組み合わせ: 体 11 × 11、L 73、滞空 21、遷移 3、腕 6 ≈ 1100 万."""
    rows = []          # 体ごと・姿勢ごと・滞空・遷移ごとに、L の格子で C1/C2 を判定
    acc = {}           # 集計
    for h in HEIGHTS:
        for m in MASSES:
            B = A.load_body(A.REF["body"], float(h), float(m), A.REF["boot"])
            sk = A.Skater(B)
            cur = izz_curves(sk)
            It = cur["tuck"][-1]
            for k, c in cur.items():
                for ta in T_AIR:
                    for tt in T_TR:
                        t, invI = profile(c, float(ta), tt)
                        N, loss, wmax = turns_and_loss(L_GRID, t, invI)
                        c1 = np.abs(N - N_NOM) < C1_TOL
                        for tol in SENS_C2:
                            ok = c1 & (loss < tol)
                            acc.setdefault((k, tol), []).append(ok.sum())
                        ok = c1 & (loss < C2_TOL)
                        for j in np.nonzero(c1)[0]:
                            rows.append((k, float(h), float(m), float(L_GRID[j]), float(ta), tt, float(N[j]),
                                         float(loss[j]), float(wmax[j]), bool(ok[j]), float(It)))
        print(f"身長 {h:.2f} m まで", flush=True)
    n_all = len(HEIGHTS) * len(MASSES) * len(L_GRID) * len(T_AIR) * len(T_TR)
    keys = ["pose", "h", "m", "L", "t_air", "t_tr", "N", "loss", "wmax", "ok", "I_tuck"]
    res = dict(n_per_pose=n_all, C1_TOL=C1_TOL, C2_TOL=C2_TOL, keys=keys, rows=rows,
               frac={f"{k}|{tol}": float(np.sum(v)) / n_all for (k, tol), v in acc.items()})
    json.dump(res, open(DATA / "bangai06_scan.json", "w"))
    summarize(res)
    return res


def summarize(res=None):
    res = res or json.load(open(DATA / "bangai06_scan.json"))
    K = res["keys"]
    R = {k: np.array([r[i] for r in res["rows"]]) for i, k in enumerate(K)}
    ok = R["ok"].astype(bool)
    s = R["L"] / (R["I_tuck"] / 0.9436)           # 縮退したパラメータ: 基準の体の I_tuck に直した L
    out = {}
    print(f"組み合わせ（腕の形ごと）{res['n_per_pose']}")
    for p in POSES:
        sel = R["pose"] == p
        c1 = sel.sum()
        c2 = (sel & ok).sum()
        o = dict(n_C1=int(c1), n_both=int(c2), frac_C2_given_C1=float(c2 / c1) if c1 else float("nan"),
                 frac_all={tol: res["frac"][f"{p}|{tol}"] for tol in SENS_C2})
        if c2:
            q = lambda x: np.percentile(x[sel & ok], [5, 50, 95]).round(3).tolist()
            o.update(L=q(R["L"]), s=q(s), t_air=q(R["t_air"]), wmax=q(R["wmax"]), N=q(R["N"]),
                     t_air_min=float(R["t_air"][sel & ok].min()))
        out[p] = o
        print(f"{POSE_JP[p]:8s} C1 を満たす {c1:7d}  C2 も {c2:7d}（{o['frac_C2_given_C1'] * 100:5.1f}%）  "
              + (f"滞空 {o['t_air']}（最短 {o['t_air_min']:.2f}）  L {o['L']}  s {o['s']}  ω {o['wmax']}" if c2 else ""))
    json.dump(out, open(DATA / "bangai06_summary.json", "w"), indent=1, ensure_ascii=False)
    return out


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    res = json.load(open(DATA / "bangai06_scan.json"))
    K = res["keys"]
    R = {k: np.array([r[i] for r in res["rows"]]) for i, k in enumerate(K)}
    ok = R["ok"].astype(bool)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    # (a) 滞空時間ごとの「C1 を満たす中で C2 も満たす割合」
    ax = axs[0]
    for i, p in enumerate(POSES):
        sel = R["pose"] == p
        fr = [ok[sel & (R["t_air"] == ta)].mean() if np.any(sel & (R["t_air"] == ta)) else np.nan for ta in T_AIR]
        ax.plot(T_AIR, np.array(fr) * 100, "o-", ms=3, color=f"C{i}", label=POSE_JP[p])
    ax.axvspan(A.T_AIR_LO, A.T_AIR_HI, color="0.9", zorder=0)
    ax.text(A.T_AIR_LO + 0.002, 3, "文献の 3A\n0.68〜0.76 s", fontsize=8)
    ax.set_xlabel("滞空時間 [s]"); ax.set_ylabel("「追えた」と矛盾しない割合 [%]")
    ax.set_title("(a) 3A を降りた組み合わせのうち、回転を追えるもの"); ax.legend(fontsize=8)
    # (b) 胸で組む: L と体の I の平面
    ax = axs[1]
    sel = R["pose"] == "tuck"
    sc = ax.scatter(R["I_tuck"][sel & ~ok], R["L"][sel & ~ok], s=2, color="0.75", label="3A は降りるが、追えない")
    ax.scatter(R["I_tuck"][sel & ok], R["L"][sel & ok], s=2, color="C0", label="両方を満たす")
    ax.axhspan(A.L_NOM - A.L_SD, A.L_NOM + A.L_SD, color="C1", alpha=0.15, label="文献の L（32.8 ± 2.0）")
    ax.set_xlabel("体の I（胸で組む）[kg m²]"); ax.set_ylabel("L [kg m²/s]")
    ax.set_title("(b) 胸で組む 3A：L と体格は L/I でしか決まらない"); ax.legend(fontsize=8, markerscale=4)
    # (c) 腕の形ごとの、残った組み合わせの滞空時間の分布
    ax = axs[2]
    data = [R["t_air"][(R["pose"] == p) & ok] for p in POSES]
    ax.boxplot([d if len(d) else [np.nan] for d in data], whis=(5, 95), showfliers=False)
    ax.set_xticks(range(1, len(POSES) + 1)); ax.set_xticklabels([POSE_JP[p] for p in POSES], fontsize=8, rotation=20)
    ax.axhspan(A.T_AIR_LO, A.T_AIR_HI, color="0.9", zorder=0)
    ax.set_ylabel("滞空時間 [s]"); ax.set_title("(c) 両方を満たす組み合わせの滞空時間（5〜95%）")
    fig.tight_layout(); fig.savefig(OUT / "bangai06_param_scan.png", dpi=130)
    print("図 → figures/field/bangai06_param_scan.png")



def literature_weight():
    """19.4（結果を見た後に追加）: 文献の事前分布 L ~ N(32.8, 2.0)、滞空 U(0.68, 0.76)、体は一様で重みをつけた
    「C1 と C2 を両方満たす確率」を腕の形ごとに出す. 総当たりの rows は C1 を満たすものだけなので、分母は解析的に出す."""
    res = json.load(open(DATA / "bangai06_scan.json"))
    K = res["keys"]
    R = {k: np.array([r[i] for r in res["rows"]]) for i, k in enumerate(K)}
    ok = R["ok"].astype(bool)
    wL = lambda L: np.exp(-0.5 * ((L - A.L_NOM) / A.L_SD) ** 2)
    in_t = lambda ta: (ta >= A.T_AIR_LO - 1e-9) & (ta <= A.T_AIR_HI + 1e-9)
    nt = np.sum(in_t(T_AIR))
    Z = wL(L_GRID).sum() * len(HEIGHTS) * len(MASSES) * nt * len(T_TR)   # 全組み合わせの重みの和
    out = {}
    for p in POSES:
        sel = R["pose"] == p
        w = wL(R["L"]) * in_t(R["t_air"])
        p1 = float(w[sel].sum() / Z)
        p12 = float(w[sel & ok].sum() / Z)
        out[p] = dict(P_C1=p1, P_C1_C2=p12, P_C2_given_C1=p12 / p1 if p1 else float("nan"))
        print(f"{POSE_JP[p]:8s} P(3A を降りる) {p1 * 100:5.1f}%   P(降りて、追える) {p12 * 100:5.1f}%   P(追える | 降りる) "
              f"{out[p]['P_C2_given_C1'] * 100:5.1f}%")
    # 滞空時間の閾値: 胸で組む・頭上で、追える割合が 50% を切る滞空
    for p in ("tuck", "up"):
        sel = R["pose"] == p
        fr = {float(ta): float(ok[sel & (R["t_air"] == ta)].mean()) for ta in T_AIR if np.any(sel & (R["t_air"] == ta))}
        out[p]["frac_by_t_air"] = fr
    json.dump(out, open(DATA / "bangai06_literature.json", "w"), indent=1, ensure_ascii=False)
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "scan"
    {"check": check, "scan": scan, "summarize": summarize, "figure": figure, "lit": literature_weight}[mode]()
