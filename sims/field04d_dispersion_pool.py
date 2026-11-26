"""Field04d: 段階 4 の (b′) 微分なし推定の「1 試行のばらつき」を、独立な 2 つの 50 回の試行で比べ直す（計算はしない、既存の出力の再解析）.

field04b（統計誤差の 50 回、乱数の種 [2026, i]）と field04c（kinematic fit と同時に行った 50 回、種 [4242, i]）は、
同じ条件（RK4、0.5 ms、雑音・バイアスの設定も同じ）で (b′) を計算しているが、1 試行の標準偏差が 1.2〜1.4% と 1.5〜1.8% で食い違う。
ここでは、(1) 2 つの標本の分散の比とそのブートストラップ区間、(2) 100 回をまとめた平均と標準偏差、
(3) 同じ試行での kinematic fit と (b′) の標準偏差の比（対応のある比較）とその区間を出す。

実行: cd sims && python field04d_dispersion_pool.py（数秒）
出力: data/field04d_summary.txt
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
N_BOOT = 20000


def main():
    c = json.load(open(os.path.join(DATA, "field04c_kinfit.json")))
    t = [r for r in c["trials"] if r["kind"] == "rk4"]
    pt = np.array([r["p_true"] for r in t])
    e_c = (np.array([r["p_bp"] for r in t]) / pt - 1) * 100       # field04c の (b′)
    e_k = (np.array([r["p_kf"] for r in t]) / pt - 1) * 100       # 同じ試行の kinematic fit
    b = json.load(open(os.path.join(DATA, "field04b_error_budget.json")))
    e_b = np.array([r["err"]["bp"] for r in b["stat"]])           # field04b の (b′)

    rng = np.random.default_rng(0)
    F, R = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(e_c), len(e_c)); j = rng.integers(0, len(e_b), len(e_b))
        F.append(e_c[i].var(0, ddof=1) / e_b[j].var(0, ddof=1))
        R.append(e_k[i].std(0, ddof=1) / e_c[i].std(0, ddof=1))
    F, R = np.array(F), np.array(R)
    pool = np.vstack([e_b, e_c])
    f3 = lambda v: " / ".join(f"{x:+.2f}" for x in v)
    lines = [
        "=== 段階 4 (b′) の 1 試行のばらつきの再評価（腕 / 脚 / 慣性 [%]） ===",
        f"field04b の 50 回: 平均 {f3(e_b.mean(0))}, 標準偏差 {f3(e_b.std(0, ddof=1))}",
        f"field04c の 50 回: 平均 {f3(e_c.mean(0))}, 標準偏差 {f3(e_c.std(0, ddof=1))}",
        f"分散の比 (04c/04b): {f3(e_c.var(0, ddof=1) / e_b.var(0, ddof=1))}, ブートストラップ 95% 区間 "
        f"[{f3(np.percentile(F, 2.5, 0))}] 〜 [{f3(np.percentile(F, 97.5, 0))}], 比が 1 未満の割合 {f3((F < 1).mean(0))}",
        f"100 回をまとめた値: 平均 {f3(pool.mean(0))}（95% 信頼区間の半幅 {f3(1.96 * pool.std(0, ddof=1) / np.sqrt(len(pool)))}）, "
        f"標準偏差 {f3(pool.std(0, ddof=1))}",
        f"kinematic fit / (b′) の標準偏差の比（同じ 50 回）: {f3(e_k.std(0, ddof=1) / e_c.std(0, ddof=1))}, "
        f"ブートストラップ 95% 区間 [{f3(np.percentile(R, 2.5, 0))}] 〜 [{f3(np.percentile(R, 97.5, 0))}]",
    ]
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(DATA, "field04d_summary.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
