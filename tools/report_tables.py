"""data/ の CSV から、詳細版の報告書に載せる LaTeX の表（report/detail/tables/）を生成する.

使い方（リポジトリのルートで）: python tools/report_tables.py
"""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "report" / "detail" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def rows(name):
    with open(DATA / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def table(fname, header, body, colspec=None, caption=None):
    colspec = colspec or "l" * len(header)
    lines = [r"\begin{longtable}{" + colspec + "}"]
    if caption:
        lines.append(r"\caption{" + caption + r"}\\")
    lines += [r"\toprule", " & ".join(header) + r" \\ \midrule \endhead"]
    lines += [" & ".join(str(c) for c in r) + r" \\" for r in body]
    lines += [r"\bottomrule", r"\end{longtable}"]
    (OUT / fname).write_text("\n".join(lines) + "\n", encoding="utf-8")


D = np.degrees

# Sim01: 質量比・長さ比のスイープ
r = np.loadtxt(DATA / "sim01_sweep.csv", delimiter=",", skiprows=1)
mus = sorted(set(np.round(r[:, 0], 4)))
pick = [mus[i] for i in (3, 9, 19, 29, 39)]
lens = sorted(set(r[:, 1]))
body = []
for m in pick:
    body.append([f"{m:.2f}"] + [f"{abs(D(r[(np.isclose(r[:,0], m)) & (r[:,1]==L), 2][0])):.1f}" for L in lens])
table("sim01_sweep.tex", [r"質量比 $\mu$"] + [f"長さ比 {L}" for L in lens], body, "rrrrr",
      r"腕を 180° 振ったときの胴体の回転角 [°]（Sim01）")

# Sim04: スケーリング
r = rows("sim04_scaling.csv")
body = [[f"{float(x['scale']):g}", f"{float(x['total_mass[kg]'])/1000:.3f}", f"{float(x['tau_leg_at_0.5Hz[Nm]']):.3g}",
         f"{float(x['f_max[Hz]']):.3f}", f"{float(x['T90[s]']):.1f}", f"{float(x['h_ambac[kg m^2/s]']):.3g}",
         f"{float(x['P_pos_at_fmax[W]'])/1000:.3g}"] for x in r]
table("sim04_scaling.tex", ["$s$", "質量 [t]", r"股 $\tau$@0.5 Hz [N\,m]", r"$f_{\max}$ [Hz]", r"$T_{90}$ [s]",
                            r"$h$ [kg\,m$^2$/s]", r"正パワー [kW]"], body, "rrrrrrr", r"相似拡大の全結果（Sim04）")

# Sim05B: パレート
r = np.loadtxt(DATA / "sim05b_pareto.csv", delimiter=",", skiprows=1)
body = [[f"{x[0]:.1f}"] + [("—" if np.isnan(v) else f"{v:.2f}") for v in x[1:]] for x in r[::6]]
table("sim05b_pareto.tex", ["$T$ [s]", "スラスタのみ", "＋RW", "＋AMBAC", "＋AMBAC＋RW"], body, "rrrrr",
      r"90° 旋回の最小推進剤 [kg]（Sim05B、18 m 級）")

# Sim05B: 評価
r = rows("sim05b_eval.csv")
name = {"open-loop OC": "開ループ最適制御", "MPC": "MPC", "MPC + wobble comp.": "MPC＋揺れ補償", "learned (ES)": "学習方策"}
body = [[("公称" if x["test"] == "nominal" else "ランダム"), name[x["controller"]], f"{float(x['err_mean_deg']):.2f}",
         f"{float(x['err_p95_deg']):.2f}", f"{float(x['rate_mean_deg_s']):.4f}", f"{float(x['propellant_kg']):.2f}",
         f"{float(x['ambac_cycles']):.1f}"] for x in r]
table("sim05b_eval.tex", ["条件", "制御器", "誤差 [°]", "95\\% [°]", "残留 [°/s]", "推進剤 [kg]", "周回"],
      body, "llrrrrr", r"モデル誤差・外乱下の比較の全結果（Sim05B）")

# Sim05A: 学習の評価
r = rows("sim05a_summary.csv")
body = [[x["target_deg"], x["controller"], f"{float(x['final_err_deg']):.2f}", f"{float(x['mean_err_last5s_deg']):.2f}"] for x in r]
table("sim05a_summary.tex", ["目標 [°]", "方策", "終端誤差 [°]", "最後 5 s の平均誤差 [°]"], body, "rlrr",
      r"学習方策と専門家制御の全評価結果（Sim05A）")

# Sim06: スケーリング
r = np.loadtxt(DATA / "sim06_scaling.csv", delimiter=",", skiprows=1)
body = [[f"{x[0]:g}", f"{x[1]:.3f}", f"{x[2]:.2f}", f"{x[3]:.3f}"] for x in r]
table("sim06_scaling.tex", ["$s$", "$g$ [m/s$^2$]", r"$T_{\min}$ [s]", r"$T_{\min}/\sqrt{s\,g_E/g}$"], body, "rrrr",
      r"安全な最短動作時間とフルード則（Sim06）")

# Sim06: ヨー
r = np.loadtxt(DATA / "sim06_yaw_ratchet.csv", delimiter=",", skiprows=1)
body = [[f"{x[0]:.2f}", f"{D(x[1]):.2f}", f"{D(x[2]):.2f}", f"{D(x[3]):.2f}"] for x in r[::2]]
table("sim06_yaw.tex", [r"$T_{\text{fast}}$ [s]", "地球 [°]", "火星 [°]", "月 [°]"], body, "rrrr",
      r"速く振ってゆっくり戻す往復 1 回あたりの台車のヨー（Sim06）")

# Sim07: 定格荷重マップの抜粋
z = np.load(DATA / "sim07_load_chart.npz")
psi, alp, ch = z["psi_deg"], z["alpha_deg"], z["max_payload_kg"]
ps_pick = [0, 30, 60, 90, 120, 150, 180]; al_pick = [-30, 0, 30, 60]
body = []
for a in al_pick:
    i = np.argmin(abs(alp - a))
    body.append([f"{alp[i]:.0f}"] + [("∞" if np.isinf(ch[i, np.argmin(abs(psi - p))]) else f"{ch[i, np.argmin(abs(psi - p))]:.1f}") for p in ps_pick])
table("sim07_load_chart.tex", [r"仰角 $\alpha$ [°] ＼ 方位 $\psi$ [°]"] + [str(p) for p in ps_pick], body, "r" * 8,
      r"静的な最大荷重 [kg]（Sim07、∞ は荷物が接地面の真上に来て転倒しない）")

# Sim08
r = rows("sim08_arm_crossing.csv")
dn = {"gimbal": "2 軸ジンバル", "ball": "球関節（人の可動域）", "ball+sg": "球関節＋上肢帯", "free": "可動域制限なし"}
body = [[x["chest"].split(" ")[0], dn[x["design"]], ("可" if x["feasible"] == "True" else "不可"), f"{float(x['hand_error_m'])*100:.1f}",
         x["horizontal_adduction_deg"], x["twist_deg"], x["elbow_deg"], x["protraction_deg"]] for x in r]
table("sim08_arm_crossing.tex", ["胸", "肩の機構", "判定", "誤差 [cm]", "水平内転", "ひねり", "肘", "突出"], body, "llcrrrrr",
      r"腕組みの判定の全結果（Sim08、角度は °、ひねりは負が内旋）")

# Sim10
r = np.loadtxt(DATA / "sim10_scaling.csv", delimiter=",", skiprows=1)
body = [[f"{x[0]:g}", f"{1.8*x[0]:.1f}", f"{x[1]:.2f}", f"{x[2]:.2f}", f"{x[3]:.3f}", f"{x[4]:.3f}"] for x in r]
table("sim10_scaling.tex", ["$s$", "全高 [m]", "戦車型 [m]", "二本脚 [m]", "戦車型/$s$", "二本脚/$s$"], body, "rrrrrr",
      r"越えられる段差と相似拡大（Sim10）")
print("tables:", sorted(p.name for p in OUT.glob("*.tex")))


# ---- 数値出力の記録（付録）---------------------------------------------------
SUP = {"⁰": "0", "¹": "1", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-"}
SUB = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4"}


def latex_line(t):
    t = t.rstrip("\n")
    if set(t.strip()) <= set("=-"):          # 区切り線は短くする
        t = t.strip()[:40]
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"),
                 ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("^", r"\^{}"), ("~", r"\~{}")):
        t = t.replace(a, b)
    out = ""
    for ch in t:
        if ch in SUP:
            out += r"\textsuperscript{" + SUP[ch] + "}"
        elif ch in SUB:
            out += r"\textsubscript{" + SUB[ch] + "}"
        elif ch in "⟨⟩":
            out += r"$\langle$" if ch == "⟨" else r"$\rangle$"
        else:
            out += ch
    out = out.replace("θ̄", r"$\bar\theta$")
    return out if out.strip() else r"\mbox{}"


files = [("Sim03", "sim03_summary.txt"), ("Sim04", "sim04_summary.txt"), ("Sim05A", "sim05a_summary.txt"),
         ("Sim05B", "sim05b_summary.txt"), ("Sim06", "sim06_summary.txt"), ("Sim07", "sim07_summary.txt"),
         ("Sim08", "sim08_summary.txt"), ("Sim09", "sim09_summary.txt"), ("Sim10", "sim10_summary.txt"),
         ("詳細版の追加解析", "detail_summary.txt")]
lines = [r"\chapter{数値出力の記録}\label{app:outputs}",
         r"本文の数値の出典として、各シミュレーションの標準出力（\texttt{data/*\_summary.txt}）をそのまま収録する。"
         r"Sim01・Sim02 の数値は第~\ref{ch:planar}章の本文に記載した。", ""]
for title, f in files:
    p = DATA / f
    if not p.exists():
        continue
    lines += [r"\section{" + title + r"（\texttt{data/" + f.replace("_", r"\_") + r"}）}",
              r"{\footnotesize\raggedright\sloppy\setlength{\parindent}{0pt}\setlength{\parskip}{0pt}\emergencystretch=3em"]
    lines += [latex_line(l)[:400] + r"\par" for l in p.read_text(encoding="utf-8").splitlines()
              if not l.startswith(("Traceback", "  File"))]
    lines += ["}", ""]
(OUT / "outputs.tex").write_text("\n".join(lines), encoding="utf-8")
print("outputs.tex written")
