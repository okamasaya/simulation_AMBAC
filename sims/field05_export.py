"""Field05: 性能表と校正データの書き出し（シミュレーション場の段階 5）.

1. 性能表: 機体 × ステージごとに、評価の窓口（ambac.field.evaluate）で荷物運びの安全な最短時間を求める
   → data/field_performance.csv、フルード則による予測（練習場の値 × √(g_地球/g)）と比較
2. 校正データ: 仮想の実機のセンサーデータ（1 回分）→ data/field_calibration_dataset.csv、正解 → data/field_calibration_truth.json
3. Excel のサマリー: data/field_summary.xlsx（説明、性能表、照合結果、キャリブレーション、計測データ）
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import csv
import json

import numpy as np

from common import DATA
from ambac.field.assembly import load
from ambac.field.evaluate import evaluate

G_E = 9.80665

# ---- 1. 性能表 ----------------------------------------------------------------
robots = ["tank_arm", "tank_waist_3axis"]
stages = ["practice", "ice", "mars", "moon"]
perf = []
print("=== 1. 性能表（荷物運びの安全な最短時間: 傾き 1° 未満・ずれ 5 mm 未満）===")
for rb in robots:
    base_T = None
    for st in stages:
        r = evaluate(load("robots", rb), load("stages", st))
        T = r.get("safe_pick_place_time_s")
        if st == "practice":
            base_T = T
        g = r["gravity"]; froude = base_T * np.sqrt(G_E / g) if base_T else None
        perf.append(dict(robot=rb, stage=st, gravity=g, friction=r["friction"], safe_time_s=T,
                         froude_prediction_s=None if froude is None else round(froude, 3), **{f"at_safe_{k}": v for k, v in r.get("at_safe_time", {}).items()}))
        print(f"  {rb:18s} {st:9s} g={g:6.3f} μ={r['friction']:.1f}: 安全な最短時間 {T} s"
              + (f"（フルード則の予測 {froude:.2f} s）" if froude and st != "practice" else ""))
with open(DATA / "field_performance.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(perf[0].keys())); w.writeheader(); w.writerows(perf)

# ---- 2. 校正データ ------------------------------------------------------------------
src = open("field04_calibration.py", encoding="utf-8").read()
ns = {"__name__": "export"}; exec(src[:src.index("rows = []")], ns)
rng = np.random.default_rng(7)
p_true = ns["NOMINAL"] * np.array([1.08, 0.93, 1.05])
data, bias = ns["experiment"](p_true, rng)
rows = []
for k, d in enumerate(data):
    for i in range(len(d["t"])):
        rows.append(dict(loop=k + 1, t_s=round(float(d["t"][i]), 4), gyro_rad_s=float(d["gyro"][i]),
                         q_arm_rad=float(d["enc"][i, 0]), q_leg_rad=float(d["enc"][i, 1]),
                         tau_arm_Nm=float(d["tau"][i, 0]), tau_leg_Nm=float(d["tau"][i, 1])))
with open(DATA / "field_calibration_dataset.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
truth = dict(arm_mass_kg=float(p_true[0]), leg_mass_kg=float(p_true[1]), torso_inertia_scale=float(p_true[2]),
             gyro_bias_rad_s=float(bias), total_mass_kg=ns["M_TOT"],
             nominal=dict(arm_mass_kg=4.0, leg_mass_kg=12.0, torso_inertia_scale=1.0),
             loops_deg=[[float(np.degrees(v)) for v in L] for L in ns["LOOPS"]],
             noise=dict(gyro_deg_s=0.2, encoder_deg=0.05, torque_Nm=0.2, fs_hz=ns["FS"]))
json.dump(truth, open(DATA / "field_calibration_truth.json", "w"), indent=2, ensure_ascii=False)
print(f"\n=== 2. 校正データ: {len(rows)} 行（6 ループ）→ data/field_calibration_dataset.csv、正解 → data/field_calibration_truth.json ===")

# ---- 3. Excel のサマリー --------------------------------------------------------------
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

wb = Workbook()
head = Font(bold=True, color="FFFFFF"); fill = PatternFill("solid", fgColor="4F6D7A")


def sheet(title, header, body, widths=None):
    ws = wb.create_sheet(title)
    ws.append(header)
    for c in ws[1]:
        c.font = head; c.fill = fill; c.alignment = Alignment(wrap_text=True, vertical="center")
    for r in body:
        ws.append(r)
    for i, wdt in enumerate(widths or [16] * len(header), 1):
        ws.column_dimensions[get_column_letter(i)].width = wdt
    ws.freeze_panes = "A2"
    return ws


ws = wb.active; ws.title = "説明"
for line in ["シミュレーション場（MuJoCo）のサマリー",
             "リポジトリ: https://github.com/okamasaya/simulation_AMBAC （docs/11_simulation_field_plan.md）",
             "性能表: 荷物運び（前方の地面 → 頭上 → 後部）を、傾き 1° 未満・ずれ 5 mm 未満で実行できる最短時間",
             "照合結果: 第 1 段階の Python モデル（基準値）と MuJoCo の比較（段階 1〜3）",
             "キャリブレーション: 答えを隠した仮想の実機から、慣性パラメータを推定した結果（段階 4）",
             "計測データ: 仮想の実機のセンサーデータ（DIY_AutoML などの試験用）。正解は data/field_calibration_truth.json"]:
    ws.append([line])
ws.column_dimensions["A"].width = 110; ws["A1"].font = Font(bold=True, size=13)

sheet("性能表", ["機体", "ステージ", "重力 [m/s²]", "摩擦係数", "安全な最短時間 [s]", "フルード則の予測 [s]", "その時の最大の傾き [°]", "その時のずれ [mm]"],
      [[p["robot"], p["stage"], p["gravity"], p["friction"], p["safe_time_s"], p["froude_prediction_s"],
        p.get("at_safe_max_tilt_deg"), p.get("at_safe_slide_mm")] for p in perf], [20, 12, 12, 10, 18, 18, 18, 14])

f1 = json.load(open(DATA / "field01_space.json")); f2 = json.load(open(DATA / "field02_ground.json")); f3 = json.load(open(DATA / "field03_assembly.json"))
val = [["段階 1", "腕 1 本の往復 4 s（Sim01）", f1["sim01_T4.0"]["theta_mid_mj"], f1["sim01_T4.0"]["theta_mid_py"], -16.04, "°"],
       ["段階 1", "腕 1 本の往復 1 s（Sim01）", f1["sim01_T1.0"]["theta_mid_mj"], f1["sim01_T1.0"]["theta_mid_py"], -16.04, "°（差はサーボの追従遅れ）"],
       ["段階 1", "腕＋脚のループ 1 周（Sim02）", f1["sim02"]["per_cycle_mj"], f1["sim02"]["per_cycle_py"], 2.9778, "°"],
       ["段階 2", "荷物運び: 転倒の始まり（Sim06）", f2["testA"]["T_tip_mujoco"], f2["testA"]["T_tip_python"], 1.62, "s（基準は安全余裕 5 cm 込み）"],
       ["段階 3", "組み立てた機体: 荷物運び 1.3 s の傾き", f3["builder_check"]["tank_T1.3"]["builder"], f3["builder_check"]["tank_T1.3"]["handmade"], None, "°（右は手作りのモデル）"],
       ["段階 3", "組み立てた宇宙の人型: ループ 1 周", f3["builder_check"]["space_loop_deg"], None, 2.9778, "°"]]
sheet("照合結果", ["段階", "試験", "MuJoCo", "Python（同じ関節の動き）", "第 1 段階の基準値", "単位・注記"], val, [10, 36, 14, 22, 18, 30])

f4 = json.load(open(DATA / "field04_calibration.json"))
cal = [["(a) 正味の回転だけ", *[round(x, 2) for x in f4["rms_a"]]], ["(b) 波形（微分）", *[round(x, 2) for x in f4["rms_b"]]],
       ["(b') 波形（微分なし）", *[round(x, 2) for x in f4["rms_p"]]], ["(c) 波形（微分）＋トルク", *[round(x, 2) for x in f4["rms_c"]]]]
sheet("キャリブレーション", ["推定の方法", "腕の質量の誤差 [%]", "脚の質量の誤差 [%]", "胴体の慣性の誤差 [%]"], cal, [26, 18, 18, 20])

sheet("計測データ", list(rows[0].keys()), [list(r.values()) for r in rows], [8, 10, 14, 14, 14, 14, 14])
wb.save(DATA / "field_summary.xlsx")
print("=== 3. Excel のサマリー → data/field_summary.xlsx（シート: " + ", ".join(wb.sheetnames) + "）===")
