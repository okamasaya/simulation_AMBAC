"""番外編 01：アルミ棒 1 本と iPhone 2 台（phyphox）で、剛体シミュレーターを現実と照合する解析.

記事：articles/ambac-bangai-bar-iphone-imu.md（英語版 docs/Dev_Comminuty/03_side_story_bar_iphone_imu.md）

照合の 3 層
  1 層目（基準モデル）：部品表（BOM）から手で書いた式。慣性モーメント I、全質量 M、重心までの距離 d、
                        周期は楕円積分で厳密に T(θ0) = 4K(sin(θ0/2)) / ω0、ω0 = sqrt(M g d / I)
  2 層目（シミュレーション場）：同じ部品表を MuJoCo で 1 つの剛体に組み立て、仮想の IMU（ジャイロ）を置く
  3 層目（実物）：phyphox の CSV（2 台）。この解析は、実測にも、1・2 層目から作った合成データにも同じようにかける

実験 A（振り子）：2 台のジャイロ ω1(t), ω2(t) に、減衰つきの実体振り子の運動方程式
      θ'' = −ω0² sin θ − 2γ θ' − β θ'|θ'|
  の解を、センサーの校正値（バイアス b1, b2、2 台目の目盛りと向き s2、時計のずれ τ2 と歩度 κ2）と同時に当てはめる
  （拘束条件つきの当てはめ。kinematic fit と同じく、推定値・共分散・χ² を同時に出す）。
  さらに、数周期ずつ区切って、前の区間の推定値と共分散を事前情報として逐次更新する（カルマンフィルタと同じ形）。
実験 B（水平に回す）：各時刻の真の角速度 ω_j を未知数として当てはめる（関節角を未知数にした field04c と同じ考え方）。
      ジャイロ 1：ω1 = ω + b1、ジャイロ 2：ω2 = s2 ω(t + τ2) + b2
      加速度（外向き）：f = −ω² r_k + b_a,k + C cos φ + S sin φ（最後の 2 項は回転面の傾きによる重力）
  雑音の入った ω を 2 乗して回帰に入れる偏り（errors-in-variables、docs/13 の F2）を避ける。

誤差の評価
  統計：当てはめの共分散。雑音の時間相関は AR(1) として静止区間から推定し、残差を白色化して当てはめる（GLS）
  系統：解析の選択を 1 つずつ変えたときのずれ（部分集合のずれは Barlow の基準で有意なものだけ数える）、
        仮定した量（時計の歩度など）、部品表の測定誤差（ヤコビアンとモンテカルロの両方）
  計算：ODE の許容誤差、MuJoCo の積分法と時間刻み（2 層目）
  自己検証：合成データ（正解が分かっている）を多数回作って当てはめ、pull =（推定 − 正解）/ σ が平均 0・標準偏差 1 か

使い方（リポジトリの直下で。Windows は PowerShell で $env:PYTHONUTF8=1 を推奨）
  python sims/bangai01_bar_imu.py predict                    部品表からの予測（記事の表）
  python sims/bangai01_bar_imu.py synth                      合成データ（phyphox と同じ形の CSV）を data/bangai01/ に書く
  python sims/bangai01_bar_imu.py analyze <config.json>      実測（または合成）データを解析する
  python sims/bangai01_bar_imu.py toys [--n 40]              合成データでの pull の検定
  python sims/bangai01_bar_imu.py mujoco                     2 層目：MuJoCo の組み立てと積分法の確認（mujoco が必要）
  python sims/bangai01_bar_imu.py all                        上をすべて（約 5〜10 分）
出力：data/bangai01/*.json, data/bangai01_summary.txt, figures/bangai/*.png
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")
import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, minimize_scalar
from scipy.sparse.linalg import splu
from scipy.special import ellipk
from scipy.stats import chi2 as chi2dist

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "bangai01"
FIG = ROOT / "figures" / "bangai"
SUMMARY = ROOT / "data" / "bangai01_summary.txt"

# 図の色（dataviz の既定の色の順：青、橙、青緑）
C1, C2, C3, CG = "#257de0", "#f47230", "#00b894", "#6b6b6b"

# ---------------------------------------------------------------------------------------------
# 部品表（BOM）。値はすべて [値, 標準偏差]、SI 単位。r は支点（回転軸）から iPhone の中心までの距離
#   bar.width：断面の幅（I に m w²/12 を加える。アングル断面の扱いの不確かさとして sd を大きめに取る）
#   phone*.com_offset：iPhone の重心が外形の中心からずれている量（棒に沿う向き）。不明なので 0 ± 3 mm
# ---------------------------------------------------------------------------------------------
DEFAULT_BOM = {
    "g": [9.798, 0.0005],
    "bar": {"mass": [0.204, 0.001], "length": [1.000, 0.001], "pivot_h": [0.020, 0.001], "width": [0.020, 0.010]},
    "phones": [
        {"mass": [0.144, 0.001], "r": [0.30, 0.001], "a": [0.1384, 0.0005], "b": [0.0673, 0.0005], "com_offset": [0.0, 0.003]},
        {"mass": [0.144, 0.001], "r": [0.90, 0.001], "a": [0.1384, 0.0005], "b": [0.0673, 0.0005], "com_offset": [0.0, 0.003]},
    ],
}
CLOCK_PPM = 50.0      # 1 台目の時計の歩度の誤差の仮定（SI 秒に対して）。この実験では決められない
GYRO_SCALE = 0.005    # ジャイロの目盛りの誤差の仮定（実験 B の r に 2 倍で効く）
ACC_SCALE = 0.005     # 加速度センサーの目盛りの誤差の仮定（実験 B の r に 1 倍で効く）


def flatten_bom(bom):
    items = [("g", *bom["g"])]
    items += [(f"bar.{k}", *v) for k, v in bom["bar"].items()]
    for i, ph in enumerate(bom["phones"]):
        items += [(f"phone{i + 1}.{k}", *v) for k, v in ph.items()]
    return [x[0] for x in items], np.array([x[1] for x in items], float), np.array([x[2] for x in items], float)


def physics(names, vals):
    """部品表 → I（支点まわり）, M, d, S = M d, ω0, T0"""
    p = dict(zip(names, vals))
    mb, L, h, w = p["bar.mass"], p["bar.length"], p["bar.pivot_h"], p["bar.width"]
    M, S = mb, mb * (L / 2 - h)
    I = mb * (L**2 + w**2) / 12 + mb * (L / 2 - h) ** 2
    k = 1
    while f"phone{k}.mass" in p:
        m, r = p[f"phone{k}.mass"], p[f"phone{k}.r"] + p[f"phone{k}.com_offset"]
        M += m; S += m * r
        I += m * (p[f"phone{k}.a"] ** 2 + p[f"phone{k}.b"] ** 2) / 12 + m * r**2
        k += 1
    w0 = math.sqrt(p["g"] * S / I)
    return dict(I=I, M=M, d=S / M, S=S, w0=w0, T0=2 * math.pi / w0)


def period_exact(T0, theta0):
    """振幅 θ0 の周期（減衰なし）。T/T0 = 2K(k)/π、k = sin(θ0/2)"""
    return T0 * 2 * ellipk(math.sin(theta0 / 2) ** 2) / math.pi


def predict(bom, n_mc=20000, seed=1):
    """予測の T0 と、部品表の誤差の伝搬（ヤコビアン：入力ごとの寄与、モンテカルロ：非線形の確認）"""
    names, vals, sds = flatten_bom(bom)
    nom = physics(names, vals)
    contrib = {}
    for i, n in enumerate(names):
        if sds[i] == 0:
            continue
        v = vals.copy(); v[i] += sds[i]
        up = physics(names, v)["T0"]
        v[i] -= 2 * sds[i]
        dn = physics(names, v)["T0"]
        contrib[n] = (up - dn) / 2
    sd_jac = math.sqrt(sum(c**2 for c in contrib.values()))
    rng = np.random.default_rng(seed)
    T = np.array([physics(names, vals + sds * rng.standard_normal(len(vals)))["T0"] for _ in range(n_mc)])
    return dict(nom=nom, sd_T0_jac=sd_jac, sd_T0_mc=float(T.std()), mean_T0_mc=float(T.mean()), contrib=contrib)


# ---------------------------------------------------------------------------------------------
# phyphox の CSV の読み書き
#   phyphox の書き出し（「CSV (Comma, decimal point)」など）は、センサーごとの CSV と meta/time.csv（記録の開始時刻）を含む。
#   列名は "Time (s)", "Gyroscope x (rad/s)", "Acceleration x (m/s^2)" など。区切り（, ; タブ）と小数点（. ,）は自動で判定する。
#   列名はゆるく照合する（実物の最初の CSV で、列が正しく選ばれたかを必ず確かめる）。
# ---------------------------------------------------------------------------------------------
def _parse_csv(text):
    lines = [ln for ln in text.splitlines() if ln.strip()]
    head = lines[0]
    if head.count(";") > head.count(","):
        sep, comma_dec = ";", True
    elif "\t" in head:
        sep, comma_dec = "\t", None
    else:
        sep, comma_dec = ",", False
    rows = list(csv.reader(lines, delimiter=sep))
    header = [h.strip() for h in rows[0]]
    if comma_dec is None:   # タブ区切り：最初の行で小数点を判定
        comma_dec = any("," in c for c in rows[1]) if len(rows) > 1 else False
    out = []
    for row in rows[1:]:
        vals = []
        for c in row[: len(header)]:
            c = c.strip()
            if comma_dec:
                c = c.replace(",", ".")
            try:
                vals.append(float(c))
            except ValueError:
                vals.append(np.nan)
        vals += [np.nan] * (len(header) - len(vals))
        out.append(vals)
    return header, np.array(out, float).reshape(-1, len(header))


def _sensor_columns(header, kind):
    pat = re.compile(r"gyro|rotation|rate" if kind == "gyro" else r"acceleration|accel", re.I)
    cols = {}
    for j, h in enumerate(header):
        if kind == "acc" and re.search(r"gyro", h, re.I):
            continue
        if pat.search(h):
            m = re.search(r"\b([xyz])\b", h, re.I)
            if m and m.group(1).lower() not in cols:
                cols[m.group(1).lower()] = j
    return cols if len(cols) == 3 else None


def load_phone(path):
    """phyphox の書き出し（フォルダまたは zip）を読む。戻り値：dict(t_gyro, gyro(N,3), t_acc, acc(N,3) or None, start_unix, acc_kind)"""
    path = Path(path)
    files = {}
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.lower().endswith(".csv"):
                    files[n.replace("\\", "/")] = z.read(n).decode("utf-8-sig", errors="replace")
    else:
        for f in sorted(path.rglob("*.csv")):
            files[f.relative_to(path).as_posix()] = f.read_text(encoding="utf-8-sig", errors="replace")
    ph = dict(t_gyro=None, gyro=None, t_acc=None, acc=None, start_unix=None, acc_kind=None, source=str(path))
    for name, text in files.items():
        header, arr = _parse_csv(text)
        if name.lower().endswith("meta/time.csv"):
            hl = [h.lower() for h in header]
            if "experiment time" in hl and "system time" in hl:
                ie, isy = hl.index("experiment time"), hl.index("system time")
                with io.StringIO(text) as fh:
                    rows = list(csv.reader(fh, delimiter=";" if header and ";" in text.splitlines()[0] else ","))
                for row, vals in zip(rows[1:], arr):
                    if row and row[0].strip().upper() == "START" and np.isfinite(vals[isy]):
                        ph["start_unix"] = float(vals[isy] - vals[ie])
                        break
            continue
        if "meta/" in name.lower():
            continue
        it = next((j for j, h in enumerate(header) if re.match(r"\s*time", h, re.I)), None)
        if it is None:
            continue
        for kind in ("gyro", "acc"):
            if ph[kind] is not None:
                continue
            cols = _sensor_columns(header, kind)
            if cols is None:
                continue
            sub = arr[:, [it, cols["x"], cols["y"], cols["z"]]]
            sub = sub[np.all(np.isfinite(sub), axis=1)]
            sub = sub[np.argsort(sub[:, 0], kind="stable")]
            ph["t_" + kind], ph[kind] = sub[:, 0], sub[:, 1:]
            if kind == "acc":
                ph["acc_kind"] = "linear" if re.search(r"linear", header[cols["x"]], re.I) else "with_g"
    if ph["gyro"] is None:
        raise ValueError(f"{path}: ジャイロの列が見つからない（列名：{list(files)}）")
    return ph


def write_phyphox(d, ph, unix0=1795000000.0):
    """合成データを phyphox の書き出しと同じ形で書く（ジャイロだけなら Raw Data.csv、加速度もあればセンサーごと）"""
    d = Path(d); (d / "meta").mkdir(parents=True, exist_ok=True)
    def dump(fn, t, arr, names):
        with open(d / fn, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, quoting=csv.QUOTE_NONNUMERIC)
            w.writerow(["Time (s)"] + names)
            for ti, row in zip(t, arr):
                fh.write(",".join(f"{v:.9E}" for v in [ti, *row]) + "\n")
    g_names = [f"Gyroscope {a} (rad/s)" for a in "xyz"]
    if ph["acc"] is None:
        absol = np.linalg.norm(ph["gyro"], axis=1)[:, None]
        dump("Raw Data.csv", ph["t_gyro"], np.hstack([ph["gyro"], absol]), g_names + ["Absolute (rad/s)"])
    else:
        dump("Gyroscope.csv", ph["t_gyro"], ph["gyro"], g_names)
        dump("Accelerometer.csv", ph["t_acc"], ph["acc"], [f"Acceleration {a} (m/s^2)" for a in "xyz"])
    t_end = float(ph["t_gyro"][-1])
    with open(d / "meta" / "time.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_ALL)
        w.writerow(["event", "experiment time", "system time", "system time text"])
        for ev, te in (("START", 0.0), ("PAUSE", t_end)):
            su = ph["start_unix"] + te
            w.writerow([ev, f"{te:.6f}", f"{su:.3f}", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(su)) + " UTC+00:00"])
    with open(d / "meta" / "device.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_ALL)
        w.writerow(["property", "value"]); w.writerow(["deviceModel", "iPhone14,6 (synthetic)"])


# ---------------------------------------------------------------------------------------------
# 雑音・静止区間・時刻合わせ
# ---------------------------------------------------------------------------------------------
AX = {"x": 0, "y": 1, "z": 2}


def ar1_noise(rng, n, sigma, rho):
    e = rng.standard_normal(n) * sigma * math.sqrt(1 - rho**2)
    x = np.empty(n); x[0] = rng.standard_normal() * sigma
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def static_stats(t, y, win):
    """静止区間の平均（バイアス）、標準偏差、1 サンプルずれの自己相関"""
    m = (t >= win[0]) & (t <= win[1])
    if m.sum() < 30:
        raise ValueError(f"静止区間 {win} のサンプルが少ない（{m.sum()} 点）")
    z = y[m] - y[m].mean()
    rho = float(np.dot(z[1:], z[:-1]) / np.dot(z, z))
    return float(y[m].mean()), float(z.std(ddof=1)), rho, int(m.sum())


def _xcorr_peak(a, b, dt, lo, hi):
    """b(t) を lag だけずらして a に合わせる相関（lag>0：b を後ろへ）。lo..hi の範囲で最大の lag（放物線で補間）"""
    n = len(a) + len(b)
    nf = 1 << (n - 1).bit_length()
    c = np.fft.irfft(np.fft.rfft(a, nf) * np.conj(np.fft.rfft(b, nf)), nf)
    lags = np.r_[np.arange(0, len(a)), np.arange(-len(b) + 1, 0)]
    cc = np.r_[c[: len(a)], c[nf - len(b) + 1:]]
    o = np.argsort(lags); lags, cc = lags[o], cc[o]
    m = (lags * dt >= lo) & (lags * dt <= hi)
    i = np.flatnonzero(m)[np.argmax(cc[m])]
    if 0 < i < len(cc) - 1:
        y0, y1, y2 = cc[i - 1], cc[i], cc[i + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    else:
        frac = 0.0
    return (lags[i] + frac) * dt


def align(t1, w1, t2, w2, coarse=0.0, search=5.0, dt=0.002):
    """2 台の時計のずれ τ（1 台目の時刻 = 2 台目の時刻 + τ）。
    包絡線（0.25 s の移動 RMS）の相関で大まかに決め（周期的な信号の 1 周期ずれを防ぐ）、生の信号の相関で詰める。
    2 台目の向き（符号）は、生の信号の相関の符号で決める"""
    g = np.arange(t1[0], t1[-1], dt)
    a = np.interp(g, t1, w1 - np.median(w1))
    gb = np.arange(t2[0], t2[-1], dt)
    b = np.interp(gb, t2, w2 - np.median(w2))
    k = max(1, int(0.25 / dt))
    env = lambda x: np.sqrt(np.convolve(x**2, np.ones(k) / k, mode="same"))
    ea, eb = env(a), env(b)
    shift0 = gb[0] - g[0]   # 配列の先頭の時刻差
    lag_e = _xcorr_peak(ea - ea.mean(), eb - eb.mean(), dt, coarse + shift0 - search, coarse + shift0 + search) - shift0
    best = None
    for sgn in (1, -1):
        lag = _xcorr_peak(a, sgn * b, dt, lag_e + shift0 - 0.6, lag_e + shift0 + 0.6) - shift0
        bb = np.interp(g, gb + lag, sgn * b, left=0, right=0)   # 2 台目の標本の 1 台目の時刻 = gb + τ
        score = float(np.dot(a, bb))
        if best is None or score > best[2]:
            best = (lag, sgn, score)
    # 仕上げ：2 台目 = a·(1 台目) + c の残差が最小になる τ を、±0.1 s の格子（1 ms）と 1 次元の最小化で探す。
    # 相関のピークは、記録の長さの違いや平均の引き方でずれる（回転の合成データで 21 ms ずれた。docs/13 の F14）
    def cost(tau):
        m = (t2 + tau > t1[0]) & (t2 + tau < t1[-1])
        y = np.interp(t2[m] + tau, t1, w1)
        A = np.c_[y, np.ones(m.sum())]
        coef = np.linalg.lstsq(A, w2[m], rcond=None)[0]
        return float(np.mean((w2[m] - A @ coef) ** 2))
    grid = best[0] + np.arange(-0.1, 0.1001, 0.001)
    c = [cost(x) for x in grid]
    i = int(np.argmin(c))
    r = minimize_scalar(cost, bounds=(grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]), method="bounded",
                        options=dict(xatol=1e-6))
    return dict(tau=float(r.x), sign=best[1], tau_xcorr=float(best[0]), tau_envelope=float(lag_e))


# ---------------------------------------------------------------------------------------------
# 実験 A：振り子
# ---------------------------------------------------------------------------------------------
PNAMES = ["w0", "gam", "beta", "th_a", "om_a", "b1", "b2", "s2", "tau2", "kap2"]
# 弱い事前情報（物理的にありうる範囲。一括の当てはめの結果にはほとんど影響しない。短い区間の逐次更新を安定させる）
WEAK_PRIOR = {"gam": (0.0, 1.0), "beta": (0.0, 1.0), "kap2": (0.0, 100e-6)}


def _pend_rhs(t, y, w0, gam, beta):
    return [y[1], -w0**2 * math.sin(y[0]) - 2 * gam * y[1] - beta * y[1] * abs(y[1])]


def pend_solve(x, t_a, t_end, rtol=1e-11):
    w0, gam, beta, th, om = x[:5]
    return solve_ivp(_pend_rhs, (t_a, t_end), [th, om], method="DOP853", rtol=rtol, atol=1e-12,
                     dense_output=True, args=(w0, gam, beta)).sol


def _whiten(r, rho):
    if rho == 0:
        return r
    w = np.empty_like(r)
    w[0] = r[0]
    w[1:] = (r[1:] - rho * r[:-1]) / math.sqrt(1 - rho**2)
    return w


def fit_pendulum(D, t_a, t_b, x0, fixed=None, use2=True, whiten=True, rtol=1e-11, prior=None, scale1=1.0):
    """D: 前処理したデータ（t1, w1, t2, w2, sig1, sig2, rho1, rho2）。t_a..t_b（1 台目の時刻）を当てはめる。
    fixed: {名前: 値}。prior: (names, mean, cov) を事前情報に加える（逐次更新）。scale1: 1 台目の目盛りの仮定"""
    fixed = dict(fixed or {})
    if not use2:
        fixed.update(b2=0.0, s2=1.0, tau2=0.0, kap2=0.0)
    free = [n for n in PNAMES if n not in fixed]
    m1 = (D["t1"] >= t_a) & (D["t1"] <= t_b)
    t1, w1 = D["t1"][m1], D["w1"][m1] / scale1
    tm2_0 = (1 + x0["kap2"]) * D["t2"] + x0["tau2"]
    m2 = (tm2_0 >= t_a + 0.05) & (tm2_0 <= t_b - 0.05)
    t2, w2 = D["t2"][m2], D["w2"][m2]
    rho1, rho2 = (D["rho1"], D["rho2"]) if whiten else (0.0, 0.0)
    pri = []
    for n, (mu, sd) in WEAK_PRIOR.items():
        if n in free:
            pri.append((free.index(n), mu, sd))
    if prior is not None:
        pn, pm, pc = prior
        idx = [free.index(n) for n in pn]
        Lp = np.linalg.cholesky(np.linalg.inv(pc))

    def unpack(z):
        x = dict(fixed); x.update(zip(free, z)); return x

    def resid(z):
        x = unpack(z)
        sol = pend_solve([x[k] for k in PNAMES[:5]], t_a, t_b + 0.2, rtol)
        out = [_whiten((w1 - sol(t1)[1] - x["b1"]) / D["sig1"], rho1)]
        if use2 and len(t2):
            tm = (1 + x["kap2"]) * t2 + x["tau2"]
            out.append(_whiten((w2 - x["s2"] * sol(tm)[1] - x["b2"]) / D["sig2"], rho2))
        out.append(np.array([(z[i] - mu) / sd for i, mu, sd in pri]))
        if prior is not None:
            out.append(Lp.T @ (z[idx] - pm))
        return np.concatenate(out)

    z0 = np.array([x0[n] for n in free], float)
    sol = least_squares(resid, z0, method="lm", x_scale="jac", xtol=1e-12, ftol=1e-12, max_nfev=400 * (len(free) + 1))
    J = sol.jac
    cov = np.linalg.inv(J.T @ J)
    n_data = len(t1) + (len(t2) if use2 else 0)
    chi2 = float(np.sum(sol.fun[:n_data] ** 2))
    x = unpack(sol.x)
    sd = dict(zip(free, np.sqrt(np.diag(cov))))
    return dict(x=x, sd=sd, cov=cov, free=free, chi2=chi2, ndf=n_data - len(free), n1=len(t1), n2=len(t2) if use2 else 0,
                status=int(sol.status), nfev=int(sol.nfev), resid=sol.fun[:n_data])


def zero_crossings(t, y):
    i = np.flatnonzero((y[:-1] > 0) & (y[1:] <= 0))
    return t[i] - y[i] * (t[i + 1] - t[i]) / (y[i + 1] - y[i])


def zc_method(t, w, b, w0_guess, t_a, t_b):
    """別の方法（照合用）：下向きの零点の時刻 tc_i を、振幅の補正つきで tc_i = t0 + T0 Σ_{j<i} f(A_j) に直線で当てはめる"""
    m = (t >= t_a - 0.05) & (t <= t_b)
    tt, y = t[m], w[m] - b
    tc = zero_crossings(tt, y)
    f = []
    for k in range(len(tc) - 1):
        s = (tt >= tc[k]) & (tt < tc[k + 1])
        wpk = 0.5 * (y[s].max() - y[s].min())
        A = 2 * math.asin(min(0.999, wpk / (2 * w0_guess)))
        f.append(2 * ellipk(math.sin(A / 2) ** 2) / math.pi)
    X = np.c_[np.ones(len(tc)), np.r_[0, np.cumsum(f)]]
    coef, res, *_ = np.linalg.lstsq(X, tc, rcond=None)
    r = tc - X @ coef
    s2 = r @ r / max(1, len(tc) - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    return dict(T0=float(coef[1]), sd=float(math.sqrt(cov[1, 1])), n_cycles=len(tc) - 1, rms_ms=float(r.std() * 1e3))


def prepare_pendulum(ph1, ph2, cfg):
    """2 台のデータから、ジャイロの軸を選び、静止区間の統計、時刻合わせ、当てはめの初期値を作る"""
    a1, a2 = AX[cfg.get("gyro_axis", ["z", "z"])[0]], AX[cfg.get("gyro_axis", ["z", "z"])[1]]
    t1, w1 = ph1["t_gyro"], ph1["gyro"][:, a1]
    t2, w2 = ph2["t_gyro"], ph2["gyro"][:, a2]
    st = cfg.get("static", [0.0, 2.5])
    b1, sig1, rho1, _ = static_stats(t1, w1, st)
    b2, sig2, rho2, _ = static_stats(t2, w2, st)
    coarse = 0.0
    if ph1.get("start_unix") and ph2.get("start_unix"):
        coarse = ph2["start_unix"] - ph1["start_unix"]
    al = align(t1, w1, t2, w2, coarse=coarse, search=cfg.get("align_search", 5.0))
    sign = cfg.get("phone2_sign") or al["sign"]
    D = dict(t1=t1, w1=w1, t2=t2, w2=w2, sig1=sig1, sig2=sig2, rho1=rho1, rho2=rho2)
    # 放した時刻：|ω1| が最大の半分を初めて超えた時刻。その後の下向きの零点（θ が最大）から skip 周期あとを当てはめの始めにする
    y = w1 - b1
    # 最大値ではなく 99 パーセンタイルを使う（1 点だけの外れ値に強くする。合成データの最後の 1 点が外挿で壊れていて失敗した）
    i_rel = np.flatnonzero(np.abs(y) > 0.5 * np.percentile(np.abs(y), 99))[0]
    tc = zero_crossings(t1, y)
    tc = tc[tc > t1[i_rel]]
    Tg = float(np.median(np.diff(tc)))
    skip = int(cfg.get("skip_cycles", 1))
    t_a = float(tc[skip]); t_b = float(cfg.get("fit_end") or tc[-1])
    w0g = 2 * math.pi / Tg
    seg = (t1 > t_a) & (t1 < t_a + Tg)
    A = 2 * math.asin(min(0.99, 0.5 * (y[seg].max() - y[seg].min()) / (2 * w0g)))
    w0g *= 2 * ellipk(math.sin(A / 2) ** 2) / math.pi
    pk = [(t1[(t1 >= tc[k]) & (t1 < tc[k + 1])].mean(), np.ptp(y[(t1 >= tc[k]) & (t1 < tc[k + 1])])) for k in range(len(tc) - 1)]
    pk = np.array(pk)
    gam0 = max(0.0, -np.polyfit(pk[:, 0], np.log(pk[:, 1]), 1)[0])
    x0 = dict(w0=w0g, gam=gam0, beta=0.0, th_a=A, om_a=0.0, b1=b1, b2=b2,
              s2=sign * float(np.std(w2[(t2 + al["tau"] > t_a) & (t2 + al["tau"] < t_b)]) /
                              np.std(w1[(t1 > t_a) & (t1 < t_b)])),
              tau2=al["tau"], kap2=0.0)
    return D, dict(t_a=t_a, t_b=t_b, x0=x0, static=dict(b1=b1, sig1=sig1, rho1=rho1, b2=b2, sig2=sig2, rho2=rho2),
                   align=dict(coarse_meta=coarse, **al), T_guess=Tg, n_cycles=len(tc) - 1 - skip)


def sequential_pendulum(D, P, n_cyc=4):
    """数周期ずつの区間に分け、前の区間までの推定値と共分散を事前情報にして逐次更新する。各区間の初期状態は区間ごとに当てはめる"""
    glob = ["w0", "gam", "beta", "b1", "b2", "s2", "tau2", "kap2"]
    t_a, t_b, T = P["t_a"], P["t_b"], P["T_guess"]
    edges = list(np.arange(t_a, t_b - 0.5 * T, n_cyc * T)) + [t_b]
    x = dict(P["x0"]); prior = None; out = []
    for k in range(len(edges) - 1):
        if k > 0:   # 前の区間の解を伸ばして、新しい区間の初期状態にする
            sol = pend_solve([x[n] for n in PNAMES[:5]], edges[k - 1], edges[k] + 0.01)
            x["th_a"], x["om_a"] = sol(edges[k])
        f = fit_pendulum(D, edges[k], edges[k + 1], x, prior=prior)
        x = dict(f["x"])
        idx = [f["free"].index(n) for n in glob]
        prior = (glob, np.array([x[n] for n in glob]), f["cov"][np.ix_(idx, idx)])
        out.append(dict(t_end=edges[k + 1], w0=x["w0"], sd_w0=f["sd"]["w0"], chi2=f["chi2"], ndf=f["ndf"]))
    return out


def analyze_pendulum(ph1, ph2, cfg, verbose=True, variations=True):
    D, P = prepare_pendulum(ph1, ph2, cfg)
    t_a, t_b, x0 = P["t_a"], P["t_b"], P["x0"]
    nom = fit_pendulum(D, t_a, t_b, x0)
    w0, sdw = nom["x"]["w0"], nom["sd"]["w0"]
    T0 = 2 * math.pi / w0; sdT = T0 * sdw / w0
    res = dict(prep={k: v for k, v in P.items() if k != "x0"}, x0=x0, fit=dict(x=nom["x"], sd=nom["sd"], chi2=nom["chi2"], ndf=nom["ndf"],
               p=float(chi2dist.sf(nom["chi2"], nom["ndf"])), n1=nom["n1"], n2=nom["n2"]),
               T0=T0, sd_T0_stat=sdT)
    r = nom["resid"][: nom["n1"]]
    res["resid_rho_after_whitening"] = float(np.dot(r[1:], r[:-1]) / np.dot(r, r))
    res["zc"] = zc_method(D["t1"], D["w1"], nom["x"]["b1"], w0, t_a, t_b)
    if not variations:
        return res, D, P, nom
    res["seq"] = sequential_pendulum(D, P, n_cyc=int(cfg.get("seq_cycles", 4)))
    # ---- 系統誤差：解析の選択を 1 つずつ変える ------------------------------------------------------
    xn = nom["x"]
    var = []
    def add(label, kind, f):
        dT = 2 * math.pi / f["x"]["w0"] - T0
        sdv = 2 * math.pi / f["x"]["w0"] ** 2 * f["sd"]["w0"]
        e = dict(label=label, kind=kind, dT=dT, sd=sdv)
        if kind == "subset":   # Barlow：部分集合どうしの統計の差 sqrt(|σv² − σn²|) で割った有意性
            e["barlow"] = sdT and abs(dT) / math.sqrt(abs(sdv**2 - sdT**2) + 1e-30)
            e["counted"] = bool(e["barlow"] > 2)
        else:
            e["counted"] = True
        var.append(e)
    add("damping model: linear only (beta=0)", "model", fit_pendulum(D, t_a, t_b, xn, fixed={"beta": 0.0}))
    add("gyro-1 scale +1%", "model", fit_pendulum(D, t_a, t_b, xn, scale1=1.01))
    add("ODE tolerance 1e-11 -> 1e-8", "computational", fit_pendulum(D, t_a, t_b, xn, rtol=1e-8))
    add("noise correlation ignored (OLS)", "method", fit_pendulum(D, t_a, t_b, xn, whiten=False))
    var[-1]["counted"] = False   # 値ではなく σ の見積もりの問題（toys で確かめる）
    t_late = t_a + 2 * P["T_guess"]
    sol = pend_solve([xn[n] for n in PNAMES[:5]], t_a, t_late + 0.01)
    xl = dict(xn); xl["th_a"], xl["om_a"] = sol(t_late)
    add("fit starts 2 cycles later", "subset", fit_pendulum(D, t_late, t_b, xl))
    add("first 60% of the record only", "subset", fit_pendulum(D, t_a, t_a + 0.6 * (t_b - t_a), xn))
    add("phone 1 only", "subset", fit_pendulum(D, t_a, t_b, xn, use2=False))
    var.append(dict(label=f"clock rate of phone 1 (assumed ±{CLOCK_PPM:.0f} ppm)", kind="assumed", dT=T0 * CLOCK_PPM * 1e-6, sd=0.0, counted=True))
    res["variations"] = var
    res["sd_T0_syst"] = math.sqrt(sum(v["dT"] ** 2 for v in var if v["counted"]))
    # ---- 予測との比較 --------------------------------------------------------------------------
    bom = cfg.get("bom", DEFAULT_BOM)
    pr = predict(bom)
    res["pred"] = dict(T0=pr["nom"]["T0"], sd=max(pr["sd_T0_jac"], pr["sd_T0_mc"]), sd_jac=pr["sd_T0_jac"], sd_mc=pr["sd_T0_mc"],
                       I=pr["nom"]["I"], M=pr["nom"]["M"], d=pr["nom"]["d"], contrib=pr["contrib"])
    sd_meas = math.hypot(sdT, res["sd_T0_syst"])
    res["pull_vs_pred"] = (T0 - res["pred"]["T0"]) / math.hypot(sd_meas, res["pred"]["sd"])
    res["ratio_T_meas_pred"] = T0 / res["pred"]["T0"]
    names, vals, _ = flatten_bom(bom)
    ph = physics(names, vals)
    res["I_meas"] = ph["S"] * dict(zip(names, vals))["g"] / w0**2    # 測った ω0 と、計量した M d から逆算した I
    if verbose:
        print_pendulum(res)
    return res, D, P, nom


def print_pendulum(res):
    f = res["fit"]
    print(f"  当てはめ：T0 = {res['T0']:.6f} ± {res['sd_T0_stat']:.6f}（統計） ± {res.get('sd_T0_syst', float('nan')):.6f}（系統） s, "
          f"χ²/ndf = {f['chi2']:.1f}/{f['ndf']}（p = {f['p']:.2f}）")
    print(f"  校正値：b1 = {f['x']['b1']*1e3:+.2f}, b2 = {f['x']['b2']*1e3:+.2f} mrad/s, s2 = {f['x']['s2']:+.5f}, "
          f"τ2 = {f['x']['tau2']*1e3:.2f} ± {f['sd']['tau2']*1e3:.2f} ms, κ2 = {f['x']['kap2']*1e6:+.1f} ± {f['sd']['kap2']*1e6:.1f} ppm, "
          f"γ = {f['x']['gam']*1e3:.2f} ± {f['sd']['gam']*1e3:.2f} /ks, β = {f['x']['beta']:.4f} ± {f['sd']['beta']:.4f}")
    zc = res["zc"]
    print(f"  照合（零点の方法、1 台目）：T0 = {zc['T0']:.6f} ± {zc['sd']:.6f} s（{zc['n_cycles']} 周期、零点の残差 {zc['rms_ms']:.2f} ms）")
    if "variations" in res:
        for v in res["variations"]:
            extra = f", Barlow {v['barlow']:.1f}σ" if "barlow" in v else ""
            print(f"    {v['label']:<44s} ΔT0 = {v['dT']*1e6:+8.1f} µs{extra}{'' if v['counted'] else '（数えない）'}")
        p = res["pred"]
        print(f"  予測：T0 = {p['T0']:.6f} ± {p['sd']:.6f} s（ヤコビアン {p['sd_jac']*1e6:.0f} µs、MC {p['sd_mc']*1e6:.0f} µs）")
        print(f"  測定 / 予測 = {res['ratio_T_meas_pred']:.5f}、pull = {res['pull_vs_pred']:+.2f}、"
              f"I（測定から逆算）= {res['I_meas']:.5f}、I（部品表）= {p['I']:.5f} kg·m²")


# ---------------------------------------------------------------------------------------------
# 実験 B：水平に回す
# ---------------------------------------------------------------------------------------------
BNAMES = ["r1", "r2", "ba1", "ba2", "b1", "b2", "s2", "tau2", "C", "S"]


def _gauss_newton(resid, z0, S, max_it=30, tol=1e-8):
    """疎なガウス・ニュートン法（ステップを半分にする線探索つき）。戻り値：(z, χ², J)"""
    from scipy.optimize._numdiff import approx_derivative, group_columns
    groups = group_columns(S)
    z = np.array(z0, float)
    r = resid(z); chi2 = float(r @ r)
    for _ in range(max_it):
        J = sparse.csc_matrix(approx_derivative(resid, z, method="2-point", sparsity=(S, groups), f0=r))
        H = (J.T @ J).tocsc()
        H = H + sparse.diags(np.full(H.shape[0], 1e-12 * H.diagonal().mean()))
        dz = splu(H).solve(-(J.T @ r))
        lam = 1.0
        while lam > 1e-4:
            zn = z + lam * dz; rn = resid(zn); cn = float(rn @ rn)
            if cn <= chi2:
                break
            lam /= 2
        done = chi2 - cn < tol * max(1.0, chi2)
        if cn <= chi2:
            z, r, chi2 = zn, rn, cn
        if done:
            break
    J = sparse.csc_matrix(approx_derivative(resid, z, method="2-point", sparsity=(S, groups), f0=r))
    return z, chi2, J


def analyze_rotation(ph1, ph2, cfg, verbose=True, variations=True, fixed=None):
    rc = cfg.get("rotation", {})
    ga = [AX[a] for a in cfg.get("gyro_axis", ["z", "z"])]
    ra = [AX[a] for a in rc.get("accel_axis", ["y", "y"])]
    rs = rc.get("accel_sign", [1, 1])       # +1：その軸の正の向きが回転軸から外向き
    st = cfg.get("static", [0.0, 2.5])
    T1, W1 = ph1["t_gyro"], ph1["gyro"][:, ga[0]]
    T2, W2 = ph2["t_gyro"], ph2["gyro"][:, ga[1]]
    TA1, F1 = ph1["t_acc"], rs[0] * ph1["acc"][:, ra[0]]
    TA2, F2 = ph2["t_acc"], rs[1] * ph2["acc"][:, ra[1]]
    sg = [static_stats(T1, W1, st), static_stats(T2, W2, st)]
    sa = [static_stats(TA1, F1, st), static_stats(TA2, F2, st)]
    coarse = (ph2["start_unix"] - ph1["start_unix"]) if (ph1.get("start_unix") and ph2.get("start_unix")) else 0.0
    al = align(T1, W1, T2, W2, coarse=coarse, search=cfg.get("align_search", 5.0))
    sign = cfg.get("phone2_sign") or al["sign"]
    # 当てはめの区間：|ω| が omega_min を超えている間
    wmin = rc.get("omega_min", 0.5)
    on = np.flatnonzero(np.abs(W1 - sg[0][0]) > wmin)
    t_a, t_b = float(T1[on[0]]), float(T1[on[-1]])
    if rc.get("window"):
        t_a, t_b = rc["window"]
    m = (T1 >= t_a) & (T1 <= t_b)
    tg = T1[m]; nl = len(tg)
    phi_full = np.r_[0, np.cumsum(0.5 * (W1[1:] + W1[:-1] - 2 * sg[0][0]) * np.diff(T1))]
    phi = lambda t: np.interp(t, T1, phi_full)

    def interp_idx(t):
        j = np.clip(np.searchsorted(tg, t) - 1, 0, nl - 2)
        u = np.clip((t - tg[j]) / (tg[j + 1] - tg[j]), 0, 1)
        return j, u
    mA1 = (TA1 >= t_a) & (TA1 <= t_b); ta1, f1 = TA1[mA1], F1[mA1]
    tm2 = T2 + al["tau"]
    mG2 = (tm2 >= t_a + 0.05) & (tm2 <= t_b - 0.05); t2g, w2 = T2[mG2], W2[mG2]
    tmA2 = TA2 + al["tau"]
    mA2 = (tmA2 >= t_a + 0.05) & (tmA2 <= t_b - 0.05); t2a, f2 = TA2[mA2], F2[mA2]
    w1 = W1[m]
    fixed = dict(fixed or {})
    free = [n for n in BNAMES if n not in fixed]
    npar = len(free)
    streams = [("g1", w1, sg[0]), ("g2", w2, sg[1]), ("a1", f1, sa[0]), ("a2", f2, sa[1])]
    noise = rc.get("noise_override") or [(st_[1], st_[2]) for _, _, st_ in streams]   # (σ, ρ)。静止区間から（診断用に上書きできる）
    whiten = rc.get("whiten", True)

    def model(z):
        x = dict(fixed); x.update(zip(free, z[:npar])); om = z[npar:]
        def at(t):
            j, u = interp_idx(t); return om[j] * (1 - u) + om[j + 1] * u
        o2g = at(t2g + x["tau2"]); oa1 = at(ta1); oa2 = at(t2a + x["tau2"])
        pa1, pa2 = phi(ta1), phi(t2a + x["tau2"])
        return x, [om + x["b1"], x["s2"] * o2g + x["b2"],
                   -x["r1"] * oa1**2 + x["ba1"] + x["C"] * np.cos(pa1) + x["S"] * np.sin(pa1),
                   -x["r2"] * oa2**2 + x["ba2"] + x["C"] * np.cos(pa2) + x["S"] * np.sin(pa2)]

    # 静止区間で測ったジャイロのバイアスを事前情報にする（平均の誤差は AR(1) の分だけ大きくする）。
    # これがないと、ω のずれ c とバイアスの組がほとんど区別できず、r が偏る（docs/13 の F15）
    bpri = []
    if rc.get("bias_prior", True):
        for k, n in enumerate(("b1", "b2")):
            if n in free:
                mu, sd_, rho_, nn = sg[k]
                bpri.append((free.index(n), mu, sd_ * math.sqrt((1 + rho_) / (1 - rho_) / nn)))

    # 過程の模型（カルマン平滑化と同じ考え方）：棒には慣性があるので ω は滑らか。角加速度 α を酔歩（強さ q）とし、
    # 隣り合う区間の α の差に σ = q sqrt(Δt) の事前情報を置く。これがないと（または q が緩すぎると）、各時刻の ω が
    # 1 台目の雑音に追従し、τ2 のヤコビアン（= ω の傾き）に雑音が入って情報を過大に見積もる（τ2 の pull の幅 2.1〜2.4、
    # docs/13 の F17）。q = 1.5 rad/s^2.5 は、手で押すときの角加速度の変化（約 13 rad/s³）を通す大きさ。q は系統誤差として 3 倍・1/3 倍に振る
    q = float(rc.get("process_q", 1.5))
    dtg = np.diff(tg)
    sig_q = q * np.sqrt(0.5 * (dtg[1:] + dtg[:-1])) if q > 0 else None

    def resid(z):
        _, mod = model(z)
        out = [_whiten((y - mo) / nz[0], nz[1] if whiten else 0.0) for (_, y, _s), mo, nz in zip(streams, mod, noise)]
        out.append(np.array([(z[i] - mu) / sd_ for i, mu, sd_ in bpri]))
        if sig_q is not None:
            al_ = np.diff(z[npar:]) / dtg
            out.append(np.diff(al_) / sig_q)
        return np.concatenate(out)

    # ヤコビアンの疎な構造（白色化で 1 つ前の行の依存も加わる）
    rows, cols, r0 = [], [], 0
    for name, y, _ in streams:
        n = len(y)
        if name == "g1":
            js = [np.arange(n)]
        else:
            tt = {"g2": t2g + al["tau"], "a1": ta1, "a2": t2a + al["tau"]}[name]
            j, _u = interp_idx(tt); js = [j - 1, j, j + 1, j + 2]
        for jj in js:
            for sh in (0, 1):
                ok = (np.arange(n) + sh < n)
                rows.append(np.arange(n)[ok] + sh + r0); cols.append(np.clip(jj[ok], 0, nl - 1) + npar)
        for k in range(npar):
            rows.append(np.arange(n) + r0); cols.append(np.full(n, k))
        r0 += n
    for i, _mu, _sd in bpri:
        rows.append(np.array([r0])); cols.append(np.array([i])); r0 += 1
    if sig_q is not None:
        nq = nl - 2
        for sh in (0, 1, 2):
            rows.append(np.arange(nq) + r0); cols.append(np.arange(nq) + sh + npar)
        r0 += nq
    S_ = sparse.coo_matrix((np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(r0, npar + nl)).tocsr()
    S_.data[:] = 1
    # 初期値：素朴な回帰 f = −r ω² + b
    o1 = w1 - sg[0][0]
    A1 = np.c_[-np.interp(ta1, tg, o1) ** 2, np.ones(len(ta1))]
    A2 = np.c_[-np.interp(t2a + al["tau"], tg, o1) ** 2, np.ones(len(t2a))]
    n1 = np.linalg.lstsq(A1, f1, rcond=None)[0]; n2 = np.linalg.lstsq(A2, f2, rcond=None)[0]
    s20 = sign * float(np.std(w2) / np.std(w1))
    x0 = dict(r1=n1[0], r2=n2[0], ba1=n1[1], ba2=n2[1], b1=sg[0][0], b2=sg[1][0], s2=s20, tau2=al["tau"], C=0.0, S=0.0)
    z0 = np.r_[[fixed.get(n, x0[n]) for n in free], o1]
    # 解法：疎なヤコビアン（差分、列のまとめ計算）と疎な直接法（LU）によるガウス・ニュートン法。収束の判定が明確で、
    # 共分散も同じ行列から出る（least_squares の trf + lsmr と同じ解になることを確かめた）
    z, chi2, J = _gauss_newton(resid, z0, S_)
    H = (J.T @ J).tocsc()
    cov = splu(H).solve(np.eye(H.shape[0])[:, :npar])[:npar]
    cov = (cov + cov.T) / 2
    x, _ = model(z)
    sd = dict(zip(free, np.sqrt(np.diag(cov))))
    # χ² は測定の行だけで数える（滑らかさ・バイアスの事前情報の行を含めると ndf が水増しされ、p 値が 1 に張り付く）
    n_data = sum(len(y) for _, y, _s in streams)
    chi2 = float(np.sum(resid(z)[:n_data] ** 2))
    # 未知の ω は事前情報で縛られているので、自由度は「実効的なパラメータの数」tr(H⁻¹ J_dᵀ J_d) で数える
    Jd = J[:n_data]
    p_eff = float(np.sum(splu(H).solve(np.eye(H.shape[0])) * (Jd.T @ Jd).toarray())) if (verbose or variations) else float(J.shape[1])
    ndf = n_data - p_eff
    sol = type("S", (), dict(status=1, x=z))
    res = dict(x={k: float(v) for k, v in x.items()}, sd=sd, chi2=chi2, ndf=ndf, p_eff=p_eff, p=float(chi2dist.sf(chi2, ndf)),
               window=[t_a, t_b], n_latent=nl, naive=dict(r1=float(n1[0]), r2=float(n2[0])), align=dict(coarse_meta=coarse, **al),
               static=dict(gyro=[s[1:3] for s in sg], acc=[s[1:3] for s in sa]), status=int(sol.status))
    ir1, ir2 = free.index("r1"), free.index("r2")
    ratio = x["r2"] / x["r1"]
    sr = ratio * math.sqrt(cov[ir1, ir1] / x["r1"] ** 2 + cov[ir2, ir2] / x["r2"] ** 2 - 2 * cov[ir1, ir2] / (x["r1"] * x["r2"]))
    res["ratio"] = dict(value=ratio, sd=sr)
    bom = cfg.get("bom", DEFAULT_BOM)
    tape = [bom["phones"][0]["r"][0], bom["phones"][1]["r"][0]]
    res["tape_r"] = tape
    res["offset_mm"] = [(x["r1"] - tape[0]) * 1e3, (x["r2"] - tape[1]) * 1e3]
    # 系統：仮定した目盛りの誤差（ω を λ 倍 → r は 1/λ²、加速度を λ 倍 → r は λ 倍）
    res["syst_scale"] = dict(gyro=[2 * GYRO_SCALE * x["r1"], 2 * GYRO_SCALE * x["r2"]], acc=[ACC_SCALE * x["r1"], ACC_SCALE * x["r2"]],
                             ratio_acc_relative=ratio * math.sqrt(2) * ACC_SCALE)
    if variations:
        var = []
        def addv(label, v):
            var.append(dict(label=label, dr1=v["x"]["r1"] - x["r1"], dr2=v["x"]["r2"] - x["r2"], dratio=v["ratio"]["value"] - ratio))
        for label, fx in (("no tilt term (C=S=0)", {"C": 0.0, "S": 0.0}), ("tau2 fixed to alignment", {"tau2": al["tau"]})):
            addv(label, analyze_rotation(ph1, ph2, cfg, verbose=False, variations=False, fixed=fx))
        for label, key, val in (("only |omega| > 1.5 rad/s (subset)", "omega_min", 1.5), ("no static-bias prior", "bias_prior", False),
                                ("process noise q x3", "process_q", 3 * q), ("process noise q /3", "process_q", q / 3)):
            cfg2 = json.loads(json.dumps(cfg)); cfg2.setdefault("rotation", {})[key] = val
            addv(label, analyze_rotation(ph1, ph2, cfg2, verbose=False, variations=False))
        res["variations"] = var
    if verbose:
        print(f"  当てはめ：r1 = {x['r1']:.5f} ± {sd['r1']:.5f} m、r2 = {x['r2']:.5f} ± {sd['r2']:.5f} m、"
              f"r2/r1 = {ratio:.5f} ± {sr:.5f}、ω2/ω1 = s2 = {x['s2']:+.5f} ± {sd['s2']:.5f}")
        print(f"  χ²/ndf = {chi2:.0f}/{ndf:.0f}（p = {res['p']:.2f}、実効的なパラメータの数 {p_eff:.0f}）、未知の ω の数 {nl}、τ2 = {x['tau2']*1e3:.2f} ms、"
              f"傾き C = {x['C']:+.4f}, S = {x['S']:+.4f} m/s²")
        print(f"  素朴な回帰（f 対 ω1²）：r1 = {n1[0]:.5f}, r2 = {n2[0]:.5f} m")
        print(f"  メジャーの位置との差：{res['offset_mm'][0]:+.2f}, {res['offset_mm'][1]:+.2f} mm（加速度センサーの取り付け位置 + 目盛り）")
        print(f"  仮定した目盛りの誤差による r の系統：ジャイロ ±{GYRO_SCALE*100:.1f}% → ±{res['syst_scale']['gyro'][0]*1e3:.1f}, "
              f"±{res['syst_scale']['gyro'][1]*1e3:.1f} mm、加速度 ±{ACC_SCALE*100:.1f}% → ±{res['syst_scale']['acc'][0]*1e3:.1f}, "
              f"±{res['syst_scale']['acc'][1]*1e3:.1f} mm")
        for v in res.get("variations", []):
            print(f"    {v['label']:<40s} Δr1 = {v['dr1']*1e3:+.2f} mm, Δr2 = {v['dr2']*1e3:+.2f} mm, Δ(r2/r1) = {v['dratio']:+.5f}")
    return res


# ---------------------------------------------------------------------------------------------
# 合成データ（正解が分かっている）
# ---------------------------------------------------------------------------------------------
SENSOR = dict(fs=100.0, jitter=0.02, drop=0.002, sig_g=0.004, rho_g=0.5, bias_g=0.01, sig_a=0.015, rho_a=0.3, bias_a=0.03)


def _sample_times(rng, t0, t1, fs, jitter, drop):
    n = int((t1 - t0) * fs * 1.05) + 10
    t = t0 + np.cumsum(np.r_[rng.uniform(0, 1 / fs), (1 + jitter * rng.standard_normal(n - 1)) / fs])
    t = t[t < t1]
    return t[rng.random(len(t)) > drop]


def mujoco_bar(names, vals, dt=1e-4, integrator="RK4", damping=0.0):
    """部品表を MuJoCo の 1 つの剛体に組み立てる（棒は箱、iPhone は箱、支点はヒンジ）。仮想の IMU（ジャイロ）を iPhone に置く"""
    import mujoco
    p = dict(zip(names, vals))
    L, h, w = p["bar.length"], p["bar.pivot_h"], p["bar.width"]
    geoms = [f'<geom type="box" size="{w/2} 0.001 {L/2}" pos="0 0 {-(L/2-h)}" mass="{p["bar.mass"]}"/>']
    sites, sensors = [], []
    k = 1
    while f"phone{k}.mass" in p:
        r = p[f"phone{k}.r"] + p[f"phone{k}.com_offset"]
        geoms.append(f'<geom type="box" size="{p[f"phone{k}.b"]/2} 0.00365 {p[f"phone{k}.a"]/2}" pos="0 0.012 {-r}" mass="{p[f"phone{k}.mass"]}"/>')
        sites.append(f'<site name="imu{k}" pos="0 0.012 {-r}"/>')
        sensors.append(f'<gyro name="gyro{k}" site="imu{k}"/>')
        k += 1
    xml = f"""<mujoco><option gravity="0 0 {-p['g']}" timestep="{dt}" integrator="{integrator}"/>
<worldbody><body name="bar"><joint name="pivot" type="hinge" axis="0 1 0" damping="{damping}"/>
{''.join(geoms)}{''.join(sites)}</body></worldbody><sensor>{''.join(sensors)}</sensor></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    Mfull = np.zeros((m.nv, m.nv))
    try:
        mujoco.mj_fullM(m, d, Mfull)          # MuJoCo 4 の呼び方
    except TypeError:
        mujoco.mj_fullM(m, Mfull, d.qM)       # MuJoCo 3 の呼び方
    I = float(Mfull[0, 0])
    S = float(m.body_subtreemass[1] * np.linalg.norm(d.subtree_com[1][[0, 2]]))   # ヒンジの軸（y）方向のずれは重力のトルクに効かない
    return m, d, I, S


def _mujoco_trace(names, vals, th0, om0, dur, gam, beta, dt=1e-4, integrator="RK4"):
    import mujoco
    m, d, I, S = mujoco_bar(names, vals, dt, integrator)
    m.dof_damping[0] = 2 * gam * I
    d.qpos[0], d.qvel[0] = th0, om0
    n = int(dur / dt) + 1
    om = np.empty(n)
    for i in range(n):
        om[i] = d.qvel[0]
        d.qfrc_applied[0] = -beta * I * d.qvel[0] * abs(d.qvel[0])
        mujoco.mj_step(m, d)
    return np.arange(n) * dt, om, I, S


def synth_pendulum(rng, bom=None, theta0_deg=10.0, n_cycles=22, gam=0.003, beta=0.01, tau2=0.31, kap2=20e-6,
                   s2=-1.002, ntp_err=0.03, source="ode", sensor=None):
    """振り子の合成データ。静止 3 s → 1.5 s で引き上げ → 1 s 保持（手の震え）→ 放す。2 台目は裏返し（s2 < 0）"""
    bom = bom or DEFAULT_BOM
    se = dict(SENSOR, **(sensor or {}))
    names, vals, _ = flatten_bom(bom)
    ph = physics(names, vals)
    w0 = ph["w0"]
    th0 = math.radians(theta0_deg)
    t_pull, t_hold, t_rel = 3.0, 4.5, 5.5
    trem = lambda t: 0.003 * np.sin(2 * np.pi * 7 * t) * np.clip((t - t_hold) / 0.3, 0, 1)
    tremd = lambda t: 0.003 * 2 * np.pi * 7 * np.cos(2 * np.pi * 7 * t) * np.clip((t - t_hold) / 0.3, 0, 1)
    th_r, om_r = th0 + float(trem(t_rel)), float(tremd(t_rel))
    t_end = t_rel + n_cycles * 2 * math.pi / w0
    if source == "mujoco":
        tt, oo, I_mj, S_mj = _mujoco_trace(names, vals, th_r, om_r, t_end - t_rel + 1.5, gam, beta)
        post = lambda t: np.interp(t - t_rel, tt, oo)
        w0_true = math.sqrt(dict(zip(names, vals))["g"] * S_mj / I_mj)
    else:
        sol = solve_ivp(_pend_rhs, (t_rel, t_end + 1.5), [th_r, om_r], method="DOP853", rtol=1e-12, atol=1e-13,
                        dense_output=True, args=(w0, gam, beta)).sol
        post = lambda t: sol(t)[1]
        w0_true = w0

    def omega(t):
        t = np.asarray(t, float)
        s = np.clip((t - t_pull) / (t_hold - t_pull), 0, 1)
        pre = np.where((t > t_pull) & (t < t_hold), th0 * (30 * s**2 - 60 * s**3 + 30 * s**4) / (t_hold - t_pull), 0.0) + \
            np.where((t >= t_hold) & (t < t_rel), tremd(t), 0.0)
        return np.where(t >= t_rel, post(np.maximum(t, t_rel)), pre)

    phones, truth = [], dict(w0=w0_true, T0=2 * math.pi / w0_true, gam=gam, beta=beta, tau2=tau2, kap2=kap2, s2=s2,
                             theta0_deg=theta0_deg, source=source)
    unix0 = 1795000000.0
    for k, (tau, kap, s) in enumerate(((0.0, 0.0, 1.0), (tau2, kap2, s2))):
        tl = _sample_times(rng, 0.0, (t_end + 1.0 - tau) / (1 + kap), se["fs"], se["jitter"], se["drop"])
        tm = (1 + kap) * tl + tau
        b = rng.uniform(-se["bias_g"], se["bias_g"])
        g = np.stack([0.01 * omega(tm) + ar1_noise(rng, len(tl), se["sig_g"], se["rho_g"]) + rng.uniform(-.01, .01),
                      -0.006 * omega(tm) + ar1_noise(rng, len(tl), se["sig_g"], se["rho_g"]) + rng.uniform(-.01, .01),
                      s * omega(tm) + b + ar1_noise(rng, len(tl), se["sig_g"], se["rho_g"])], 1)
        truth[f"b{k + 1}"] = b
        phones.append(dict(t_gyro=tl, gyro=g, t_acc=None, acc=None, start_unix=unix0 + tau + (rng.normal(0, ntp_err) if k else 0.0)))
    return phones, truth


def synth_rotation(rng, bom=None, dr=(0.005, -0.004), s2=1.003, tilt_deg=0.5, tau2=0.27, ntp_err=0.03, sensor=None):
    """水平に回す合成データ。静止 3 s → 手で 2 回押して回す（摩擦で減速）。加速度センサーは iPhone の中心から dr ずれている"""
    bom = bom or DEFAULT_BOM
    se = dict(SENSOR, **(sensor or {}))
    r_true = [bom["phones"][0]["r"][0] + dr[0], bom["phones"][1]["r"][0] + dr[1]]
    dt = 1e-3
    tt = np.arange(0, 26.0, dt)
    push = lambda t, t0, a: np.where((t > t0) & (t < t0 + 1.2), a * np.sin(np.pi * (t - t0) / 1.2) ** 2, 0.0)
    om = np.zeros_like(tt)
    for i in range(1, len(tt)):
        t = tt[i - 1]
        acc = push(t, 3.0, 5.0) + push(t, 13.0, 3.5) - (om[i - 1] / 7.0 if om[i - 1] > 0 else 0.0) - (0.02 if om[i - 1] > 0 else 0.0)
        om[i] = max(0.0, om[i - 1] + acc * dt)
    al = np.gradient(om, dt)
    ph_ = np.r_[0, np.cumsum(0.5 * (om[1:] + om[:-1]) * dt)]
    g = dict(zip(*flatten_bom(bom)[:2]))["g"]
    tilt = g * math.sin(math.radians(tilt_deg)); phi0 = 0.7
    phones, truth = [], dict(r=r_true, s2=s2, tau2=tau2, tilt_deg=tilt_deg)
    unix0 = 1795000000.0
    for k, (tau, s) in enumerate(((0.0, 1.0), (tau2, s2))):
        tl = _sample_times(rng, 0.0, 26.0 - tau - 0.2, se["fs"], se["jitter"], se["drop"])
        tm = tl + tau
        o = np.interp(tm, tt, om); a_ = np.interp(tm, tt, al); p_ = np.interp(tm, tt, ph_)
        b = rng.uniform(-se["bias_g"], se["bias_g"]); ba = rng.uniform(-se["bias_a"], se["bias_a"])
        gy = np.stack([ar1_noise(rng, len(tl), se["sig_g"], se["rho_g"]), ar1_noise(rng, len(tl), se["sig_g"], se["rho_g"]),
                       s * o + b + ar1_noise(rng, len(tl), se["sig_g"], se["rho_g"])], 1)
        ta = _sample_times(rng, 0.0, 26.0 - tau - 0.2, se["fs"], se["jitter"], se["drop"])
        tma = ta + tau
        oa = np.interp(tma, tt, om); aa = np.interp(tma, tt, al); pa = np.interp(tma, tt, ph_)
        acc = np.stack([aa * r_true[k] + ar1_noise(rng, len(ta), se["sig_a"], se["rho_a"]),
                        -oa**2 * r_true[k] + ba + tilt * np.cos(pa - phi0) + ar1_noise(rng, len(ta), se["sig_a"], se["rho_a"]),
                        g + ar1_noise(rng, len(ta), se["sig_a"], se["rho_a"])], 1)
        del o, a_, p_
        truth[f"b{k + 1}"], truth[f"ba{k + 1}"] = b, ba
        phones.append(dict(t_gyro=tl, gyro=gy, t_acc=ta, acc=acc, acc_kind="with_g",
                           start_unix=unix0 + tau + (rng.normal(0, ntp_err) if k else 0.0)))
    return phones, truth


def default_config(experiment):
    return {"experiment": experiment, "phones": ["phone1", "phone2"], "gyro_axis": ["z", "z"], "static": [0.0, 2.5],
            "skip_cycles": 1, "seq_cycles": 4, "phone2_sign": None,
            "rotation": {"accel_axis": ["y", "y"], "accel_sign": [1, 1], "omega_min": 0.5, "process_q": 1.5, "bias_prior": True},
            "bom": DEFAULT_BOM}


# ---------------------------------------------------------------------------------------------
# toy MC（pull の検定）
# ---------------------------------------------------------------------------------------------
def _toy_pend(i):
    rng = np.random.default_rng([20261126, i])
    src = "mujoco" if i < 0 else "ode"
    phones, tr = synth_pendulum(rng, theta0_deg=rng.uniform(7, 13), tau2=rng.uniform(0.1, 0.6), gam=rng.uniform(0.001, 0.006),
                                beta=rng.uniform(0.0, 0.03), source=src)
    cfg = default_config("pendulum")
    out = dict(i=i, truth=tr)
    try:
        res, D, P, nom = analyze_pendulum(*phones, cfg, verbose=False, variations=False)
        ols = fit_pendulum(D, P["t_a"], P["t_b"], nom["x"], whiten=False)
        out.update(T0=res["T0"], sd=res["sd_T0_stat"], chi2=res["fit"]["chi2"], ndf=res["fit"]["ndf"],
                   pull=(res["T0"] - tr["T0"]) / res["sd_T0_stat"],
                   pull_ols=(2 * math.pi / ols["x"]["w0"] - tr["T0"]) / (2 * math.pi / ols["x"]["w0"] ** 2 * ols["sd"]["w0"]),
                   pull_tau=(res["fit"]["x"]["tau2"] - tr["tau2"]) / res["fit"]["sd"]["tau2"],
                   pull_zc=(res["zc"]["T0"] - tr["T0"]) / res["zc"]["sd"])
    except Exception as e:  # noqa: BLE001  失敗も記録する
        out["error"] = repr(e)
    return out


def _toy_rot(i):
    rng = np.random.default_rng([20261127, i])
    phones, tr = synth_rotation(rng, tau2=rng.uniform(0.1, 0.6), tilt_deg=rng.uniform(0, 1.0))
    cfg = default_config("rotation")
    out = dict(i=i, truth=tr)
    try:
        res = analyze_rotation(*phones, cfg, verbose=False, variations=False)
        x, sd = res["x"], res["sd"]
        out.update(pull_r1=(x["r1"] - tr["r"][0]) / sd["r1"], pull_r2=(x["r2"] - tr["r"][1]) / sd["r2"],
                   pull_s2=(x["s2"] - tr["s2"]) / sd["s2"],
                   pull_ratio=(res["ratio"]["value"] - tr["r"][1] / tr["r"][0]) / res["ratio"]["sd"],
                   naive_bias_r1=res["naive"]["r1"] - tr["r"][0], chi2=res["chi2"], ndf=res["ndf"])
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)
    return out


def run_toys(n_pend=40, n_rot=20, procs=None):
    procs = procs or min(4, os.cpu_count() or 1)
    t0 = time.time()
    with Pool(procs) as pool:
        pend = pool.map(_toy_pend, range(n_pend))
        rot = pool.map(_toy_rot, range(n_rot))
    def st(lst, key):
        v = np.array([o[key] for o in lst if key in o])
        return dict(n=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)), sd_err=float(v.std(ddof=1) / math.sqrt(2 * (len(v) - 1))),
                    mean_err=float(v.std(ddof=1) / math.sqrt(len(v))))
    summ = dict(n_pend=n_pend, n_rot=n_rot, seconds=time.time() - t0,
                pend={k: st(pend, k) for k in ("pull", "pull_ols", "pull_tau", "pull_zc")},
                rot={k: st(rot, k) for k in ("pull_r1", "pull_r2", "pull_s2", "pull_ratio")},
                failures=[o for o in pend + rot if "error" in o])
    pv = [chi2dist.sf(o["chi2"], o["ndf"]) for o in pend if "chi2" in o]
    summ["pend_chi2_p_mean"] = float(np.mean(pv))
    summ["rot_naive_bias_r1_mm"] = float(np.mean([o["naive_bias_r1"] for o in rot if "naive_bias_r1" in o]) * 1e3)
    return summ, pend, rot


# ---------------------------------------------------------------------------------------------
# 2 層目：MuJoCo の組み立てと積分法
# ---------------------------------------------------------------------------------------------
def mujoco_check(bom=None, theta0_deg=10.0, n_per=12):
    import mujoco
    bom = bom or DEFAULT_BOM
    names, vals, _ = flatten_bom(bom)
    ph = physics(names, vals)
    _, _, I, S = mujoco_bar(names, vals)
    th0 = math.radians(theta0_deg)
    T_ex = period_exact(ph["T0"], th0)
    rows = []
    for integ, dt in (("RK4", 1e-4), ("RK4", 5e-4), ("RK4", 2e-3), ("implicitfast", 1e-4), ("implicitfast", 5e-4), ("implicitfast", 2e-3),
                      ("Euler", 5e-4), ("implicit", 5e-4)):
        m, d, _, _ = mujoco_bar(names, vals, dt, integ)
        d.qpos[0] = th0
        n = int(n_per * T_ex / dt)
        q = np.empty(n); v = np.empty(n)
        for i in range(n):
            q[i], v[i] = d.qpos[0], d.qvel[0]
            mujoco.mj_step(m, d)
        t = np.arange(n) * dt
        tc = zero_crossings(t, v)
        T = float(np.polyfit(np.arange(len(tc)), tc, 1)[0])
        E = 0.5 * I * v**2 + dict(zip(names, vals))["g"] * S * (1 - np.cos(q))
        rows.append(dict(integrator=integ, dt=dt, T=T, rel_err=T / T_ex - 1, energy_drift=float(E[-1] / E[0] - 1)))
    return dict(I_formula=ph["I"], I_mujoco=I, S_formula=ph["S"], S_mujoco=S, T_exact=T_ex, T0=ph["T0"], rows=rows)


# ---------------------------------------------------------------------------------------------
# 図
# ---------------------------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                         "font.size": 8, "legend.frameon": False})
    return plt


def fig_pendulum(res, D, P, nom, path, title):
    plt = _plt()
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.4))
    x = nom["x"]
    sol = pend_solve([x[n] for n in PNAMES[:5]], P["t_a"], P["t_b"] + 0.2)
    a = ax[0, 0]
    t1, w1 = D["t1"], D["w1"]
    tm2 = (1 + x["kap2"]) * D["t2"] + x["tau2"]
    z = (t1 > P["t_a"]) & (t1 < P["t_a"] + 4)
    z2 = (tm2 > P["t_a"]) & (tm2 < P["t_a"] + 4)
    a.plot(t1[z], w1[z] - x["b1"], ".", ms=2.5, color=C1, label="phone 1 (bias removed)")
    a.plot(tm2[z2], (D["w2"][z2] - x["b2"]) / x["s2"], ".", ms=2.5, color=C2, label="phone 2 (aligned, /s2)")
    tf = np.linspace(P["t_a"], P["t_a"] + 4, 800)
    a.plot(tf, sol(tf)[1], "-", lw=1, color=CG, label="fit (pendulum ODE)")
    a.set_xlabel("time on phone-1 clock [s]"); a.set_ylabel("angular velocity [rad/s]"); a.set_title("(a) both gyros on one rigid body")
    a.legend(fontsize=7)
    a = ax[0, 1]
    r = nom["resid"]
    a.plot(t1[(t1 >= P["t_a"]) & (t1 <= P["t_b"])], r[: nom["n1"]], ",", color=C1)
    a.set_xlabel("time [s]"); a.set_ylabel("whitened residual / σ"); a.set_title(f"(b) residuals, χ²/ndf = {res['fit']['chi2']:.0f}/{res['fit']['ndf']}")
    a = ax[1, 0]
    s = res["seq"]
    tt = [q["t_end"] - P["t_a"] for q in s]
    T = np.array([2 * math.pi / q["w0"] for q in s]); sT = np.array([2 * math.pi / q["w0"] ** 2 * q["sd_w0"] for q in s])
    a.errorbar(tt, (T - res["pred"]["T0"]) * 1e3, sT * 1e3, fmt="o", ms=4, color=C1, capsize=2, label="sequential update")
    a.axhspan(-res["pred"]["sd"] * 1e3, res["pred"]["sd"] * 1e3, color=C2, alpha=.18, lw=0, label="prediction from BOM (±1σ)")
    a.axhline((res["T0"] - res["pred"]["T0"]) * 1e3, color=CG, lw=1, ls="--", label="batch fit")
    if "truth" in res:
        a.axhline((res["truth"]["T0"] - res["pred"]["T0"]) * 1e3, color=C3, lw=1.2, label="truth (synthetic)")
    a.set_xlabel("data used [s after fit start]"); a.set_ylabel("T0 − predicted [ms]"); a.set_title("(c) sequential calibration of T0")
    a.legend(fontsize=7)
    a = ax[1, 1]
    labs = ["statistical"] + [v["label"] for v in res["variations"]] + ["prediction: BOM"]
    vals = [res["sd_T0_stat"]] + [abs(v["dT"]) for v in res["variations"]] + [res["pred"]["sd"]]
    cols = [C1] + [C2 if v["counted"] else "#c9c9c9" for v in res["variations"]] + [C3]
    y = np.arange(len(labs))[::-1]
    a.barh(y, np.array(vals) * 1e6, color=cols, height=.6)
    a.set_yticks(y); a.set_yticklabels(labs, fontsize=6.5); a.set_xscale("log"); a.set_xlabel("|ΔT0| or σ [µs]")
    a.set_title("(d) error budget (grey: not counted)")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)


def fig_toys(summ, pend, rot, mj, path):
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    bins = np.linspace(-4, 4, 17)
    xx = np.linspace(-4, 4, 200); gg = np.exp(-xx**2 / 2) / math.sqrt(2 * math.pi)
    a = ax[0]
    for key, col, lab in (("pull", C1, "GLS (whitened)"), ("pull_ols", C2, "OLS (noise correlation ignored)")):
        v = [o[key] for o in pend if key in o]
        s = summ["pend"][key]
        a.hist(v, bins=bins, density=True, histtype="step", lw=1.6, color=col, label=f"{lab}: sd {s['sd']:.2f}±{s['sd_err']:.2f}")
    a.plot(xx, gg, color=CG, lw=1, ls=":", label="N(0,1)")
    a.set_xlabel("pull of T0"); a.set_title(f"(a) pendulum, {summ['n_pend']} synthetic runs"); a.legend(fontsize=6.5)
    a = ax[1]
    for key, col, lab in (("pull_r1", C1, "r1"), ("pull_r2", C2, "r2"), ("pull_s2", C3, "s2 = ω2/ω1")):
        v = [o[key] for o in rot if key in o]
        s = summ["rot"][key]
        a.hist(v, bins=bins, density=True, histtype="step", lw=1.6, color=col, label=f"{lab}: mean {s['mean']:+.2f}, sd {s['sd']:.2f}")
    a.plot(xx, gg, color=CG, lw=1, ls=":")
    a.set_xlabel("pull"); a.set_title(f"(b) rotation, {summ['n_rot']} synthetic runs"); a.legend(fontsize=6.5)
    a = ax[2]
    if mj:
        for integ, col in (("RK4", C1), ("implicitfast", C2)):
            rr = [r for r in mj["rows"] if r["integrator"] == integ]
            a.loglog([r["dt"] * 1e3 for r in rr], [abs(r["rel_err"]) for r in rr], "o-", color=col, label=integ)
        for r in mj["rows"]:
            if r["integrator"] in ("Euler", "implicit"):
                a.loglog(r["dt"] * 1e3, abs(r["rel_err"]), "s", color=C3 if r["integrator"] == "Euler" else "#6a4ce2", label=r["integrator"])
        a.axhline(1e-3, color=CG, lw=1, ls=":", label="0.1% (target of measurement)")
        a.set_xlabel("MuJoCo timestep [ms]"); a.set_ylabel("|T_MuJoCo / T_exact − 1|"); a.set_title("(c) layer 2: integrator error of period")
        a.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def fig_rotation(res, ph1, ph2, cfg, path, title):
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    x = res["x"]
    ga = AX[cfg.get("gyro_axis", ["z", "z"])[0]]
    ra = [AX[a] for a in cfg.get("rotation", {}).get("accel_axis", ["y", "y"])]
    rs = cfg.get("rotation", {}).get("accel_sign", [1, 1])
    a = ax[0]
    a.plot(ph1["t_gyro"], ph1["gyro"][:, ga] - x["b1"], color=C1, lw=.8, label="phone 1")
    a.plot(ph2["t_gyro"] + x["tau2"], (ph2["gyro"][:, ga] - x["b2"]) / x["s2"], color=C2, lw=.8, label="phone 2 (aligned, /s2)")
    a.set_xlabel("time [s]"); a.set_ylabel("ω [rad/s]"); a.set_title("(a) angular velocity"); a.legend(fontsize=7)
    a = ax[1]
    for ph, k, col in ((ph1, 0, C1), (ph2, 1, C2)):
        tau = 0.0 if k == 0 else x["tau2"]
        om = np.interp(ph["t_acc"] + tau, ph1["t_gyro"], ph1["gyro"][:, ga] - x["b1"])
        f = rs[k] * ph["acc"][:, ra[k]] - x[f"ba{k + 1}"]
        sel = np.abs(om) > 0.3
        a.plot(om[sel] ** 2, -f[sel], ",", color=col, alpha=.5)
        w2 = np.linspace(0, (om**2).max(), 10)
        a.plot(w2, x[f"r{k + 1}"] * w2, color=col, lw=1.2, label=f"phone {k + 1}: r = {x[f'r{k + 1}']:.4f} m")
    a.set_xlabel("ω² [rad²/s²]"); a.set_ylabel("centripetal acceleration [m/s²]"); a.set_title("(b) a_c = ω² r"); a.legend(fontsize=7)
    fig.suptitle(title, fontsize=9); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------------------------
# コマンド
# ---------------------------------------------------------------------------------------------
def cmd_predict(bom=None):
    bom = bom or DEFAULT_BOM
    names, vals, _ = flatten_bom(bom)
    lines = ["予測（部品表から。g = %.3f m/s²）" % bom["g"][0]]
    configs = [("棒だけ（iPhone なし）", []), ("iPhone を 0.30 m と 0.90 m に", [0.30, 0.90]), ("iPhone を 0.48 m と 0.95 m に", [0.48, 0.95])]
    for lab, rr in configs:
        b = json.loads(json.dumps(bom))
        b["phones"] = [dict(bom["phones"][i], r=[r, bom["phones"][i]["r"][1]]) for i, r in enumerate(rr)]
        pr = predict(b, n_mc=4000)
        n = pr["nom"]
        lines.append(f"  {lab:<28s} I = {n['I']:.4f} kg·m², M = {n['M']:.3f} kg, d = {n['d']:.3f} m, T0 = {n['T0']:.4f} ± {pr['sd_T0_mc']:.4f} s, "
                     f"T(10°) = {period_exact(n['T0'], math.radians(10)):.4f} s")
    pr = predict(bom)
    lines.append(f"  基準の構成の T0 の誤差：ヤコビアン {pr['sd_T0_jac']*1e6:.0f} µs、モンテカルロ {pr['sd_T0_mc']*1e6:.0f} µs"
                 f"（平均のずれ {(pr['mean_T0_mc'] - pr['nom']['T0'])*1e6:+.1f} µs）")
    for k, v in sorted(pr["contrib"].items(), key=lambda kv: -abs(kv[1])):
        lines.append(f"    {k:<22s} {v*1e6:+8.1f} µs（{v / pr['nom']['T0'] * 100:+.4f}%）")
    print("\n".join(lines))
    return lines, pr


def cmd_synth():
    out = {}
    rng = np.random.default_rng(20261126)
    for exp, gen in (("pendulum", synth_pendulum), ("rotation", synth_rotation)):
        phones, truth = gen(rng)
        d = OUT / f"synthetic_{exp}"
        for k, ph in enumerate(phones):
            write_phyphox(d / f"phone{k + 1}", ph)
        cfg = default_config(exp)
        (d / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
        (d / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
        out[exp] = d
        print(f"  合成データ：{d.relative_to(ROOT)}（正解は truth.json）")
    return out


def cmd_analyze(cfg_path, tag=None):
    cfg_path = Path(cfg_path)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    base = cfg_path.parent
    ph1, ph2 = (load_phone(base / p) for p in cfg["phones"])
    tag = tag or base.name
    truth = json.loads((base / "truth.json").read_text()) if (base / "truth.json").exists() else None
    print(f"[解析] {cfg_path.relative_to(ROOT) if cfg_path.is_relative_to(ROOT) else cfg_path}（{cfg['experiment']}）")
    if cfg["experiment"] == "pendulum":
        res, D, P, nom = analyze_pendulum(ph1, ph2, cfg)
        if truth:
            res["truth"] = truth
            print(f"  正解（合成）：T0 = {truth['T0']:.6f} s → 統計の pull {(res['T0'] - truth['T0']) / res['sd_T0_stat']:+.2f}、"
                  f"τ2 の pull {(res['fit']['x']['tau2'] - truth['tau2']) / res['fit']['sd']['tau2']:+.2f}")
        fig_pendulum(res, D, P, nom, FIG / f"bangai01_{tag}.png", f"bangai01 pendulum: {tag}")
    else:
        res = analyze_rotation(ph1, ph2, cfg)
        if truth:
            res["truth"] = truth
            print(f"  正解（合成）：r = {truth['r'][0]:.4f}, {truth['r'][1]:.4f} m → pull "
                  f"{(res['x']['r1'] - truth['r'][0]) / res['sd']['r1']:+.2f}, {(res['x']['r2'] - truth['r'][1]) / res['sd']['r2']:+.2f}")
        fig_rotation(res, ph1, ph2, cfg, FIG / f"bangai01_{tag}.png", f"bangai01 rotation: {tag}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"result_{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return res


class Tee:
    def __init__(self, *s): self.s = s
    def write(self, x):
        for s in self.s: s.write(x)
    def flush(self):
        for s in self.s: s.flush()


def cmd_all(n_toys):
    t0 = time.time()
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = Tee(old, buf)
    try:
        print(f"番外編 01：アルミ棒 + iPhone 2 台（{time.strftime('%Y-%m-%d')}、sims/bangai01_bar_imu.py）\n")
        print("== 1 層目：部品表からの予測 =="); cmd_predict(); print()
        mj = None
        try:
            print("== 2 層目：MuJoCo の組み立てと積分法 ==")
            mj = mujoco_check()
            print(f"  I：式 {mj['I_formula']:.8f}、MuJoCo {mj['I_mujoco']:.8f} kg·m²（差 {mj['I_mujoco']/mj['I_formula']-1:+.1e}）；"
                  f" M d：式 {mj['S_formula']:.8f}、MuJoCo {mj['S_mujoco']:.8f}")
            print(f"  振幅 10° の厳密な周期 {mj['T_exact']:.6f} s に対する、MuJoCo の周期の相対誤差とエネルギーのずれ（12 周期）")
            for r in mj["rows"]:
                print(f"    {r['integrator']:<13s} dt = {r['dt']*1e3:4.1f} ms：{r['rel_err']:+.2e}、エネルギー {r['energy_drift']:+.2e}")
            (OUT / "mujoco_check.json").write_text(json.dumps(mj, indent=1), encoding="utf-8")
        except ImportError:
            print("  mujoco がないので省略（pip install -r requirements-field.txt）")
        print()
        print("== 合成データ（phyphox と同じ形の CSV）=="); dirs = cmd_synth(); print()
        print("== 3 層目の解析を、合成データにかける（実測でも同じ手順）==")
        for exp, d in dirs.items():
            cmd_analyze(d / "config.json"); print()
        if mj is not None:
            print("== MuJoCo の仮想の IMU で作った合成データ（2 層目 → 解析）==")
            rng = np.random.default_rng(7)
            phones, tr = synth_pendulum(rng, source="mujoco")
            res, *_ = analyze_pendulum(*phones, default_config("pendulum"), verbose=False, variations=False)
            print(f"  T0：解析 {res['T0']:.6f} ± {res['sd_T0_stat']:.6f} s、MuJoCo の組み立て {tr['T0']:.6f} s → pull {(res['T0'] - tr['T0']) / res['sd_T0_stat']:+.2f}")
            print()
        print(f"== toy MC：合成データ {n_toys} 回（振り子）、{max(10, n_toys)} 回（回転）で pull を検定 ==")
        summ, pend, rot = run_toys(n_toys, max(10, n_toys))
        for grp in ("pend", "rot"):
            for k, s in summ[grp].items():
                print(f"  {grp}.{k:<11s} 平均 {s['mean']:+.3f} ± {s['mean_err']:.3f}、標準偏差 {s['sd']:.3f} ± {s['sd_err']:.3f}（{s['n']} 回）")
        print(f"  振り子の χ² の p 値の平均 {summ['pend_chi2_p_mean']:.2f}（一様なら 0.5）、回転の素朴な回帰の r1 の偏り {summ['rot_naive_bias_r1_mm']:+.2f} mm")
        print(f"  当てはめに失敗した回：{len(summ['failures'])}（{summ['seconds']:.0f} s）")
        (OUT / "toys.json").write_text(json.dumps(dict(summary=summ, pend=pend, rot=rot), indent=1, default=float), encoding="utf-8")
        fig_toys(summ, pend, rot, mj, FIG / "bangai01_toys.png")
        print(f"\n（全体 {time.time() - t0:.0f} s）")
    finally:
        sys.stdout = old
    SUMMARY.write_text(buf.getvalue(), encoding="utf-8")
    print(f"→ {SUMMARY.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd")
    sp.add_parser("predict"); sp.add_parser("synth"); sp.add_parser("mujoco")
    a = sp.add_parser("analyze"); a.add_argument("config")
    t = sp.add_parser("toys"); t.add_argument("--n", type=int, default=40)
    al = sp.add_parser("all"); al.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    if args.cmd == "predict":
        cmd_predict()
    elif args.cmd == "synth":
        cmd_synth()
    elif args.cmd == "analyze":
        cmd_analyze(args.config)
    elif args.cmd == "toys":
        summ, *_ = run_toys(args.n, max(10, args.n))
        print(json.dumps({k: summ[k] for k in ("pend", "rot")}, indent=1))
    elif args.cmd == "mujoco":
        print(json.dumps(mujoco_check(), indent=1))
    else:
        cmd_all(getattr(args, "n", 40))


if __name__ == "__main__":
    main()
