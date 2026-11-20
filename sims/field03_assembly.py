"""Field03: アセンブル（部品表から機体を組み立てる）とステージ（シミュレーション場の段階 3）.

1. 組み立ての検証: 部品表から組み立てた機体が、段階 1・2 で手作りしたモデルと同じ結果を出すか
2. ギャラリー: 機体 × ステージの組み合わせを画像で確認
3. 試験メニューの例: 同じ荷物運びの動きを、機体 × ステージで実行して指標を比べる
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import json

import numpy as np
import matplotlib.pyplot as plt

from common import DATA, FIG, smooth_step
from ambac.ground import min_jerk
from ambac.field import ground as fg, space as fs
from ambac.field.assembly import build, load, mass_properties
from ambac.field.menu import run_menu, snapshot

D = np.degrees
RAD = np.radians
OUT = FIG / "field"
res = {"builder_check": {}, "menu": []}

# ---- 1. 組み立ての検証 ------------------------------------------------------------
print("=== 1. 組み立ての検証（部品表 → MJCF と、手作りのモデルの比較）===")
xml_b, joints_b = build(load("robots", "tank_arm"), load("stages", "practice"))
xml_h = fg.tank_arm_mjcf()
mb, cb = mass_properties(xml_b); mh, ch = mass_properties(xml_h)
print(f"  戦車型＋腕: 総質量 {mb:.3f} / {mh:.3f} kg, 重心 {np.round(cb, 4)} / {np.round(ch, 4)}")
Q0, Q1 = RAD([-20.0, -50.0]), RAD([130.0, 60.0])
for T in (1.5, 1.3):
    _, sb = run_menu(xml_b, joints_b, dict(zip(joints_b, Q0)), dict(zip(joints_b, Q1)), T, min_jerk)
    oh = fg.run(xml_h, Q0, lambda tau, t: Q0 + (Q1 - Q0) * min_jerk(tau), T)
    tilt_h = D(np.abs(oh["pitch"][oh["t"] >= 0]).max())
    res["builder_check"][f"tank_T{T}"] = dict(builder=sb["max_tilt_deg"], handmade=tilt_h)
    print(f"  荷物運び T={T} s: 最大の傾き 組み立て {sb['max_tilt_deg']:.3f}° / 手作り {tilt_h:.3f}°")
xml_s, joints_s = build(load("robots", "space_arm_leg"), load("stages", "space"))
corners = RAD(np.array([[-90, -90], [90, -90], [90, -30], [-90, -30], [-90, -90]], float))


def ref2(t):
    t = t % 4.0; k = min(int(t // 1.0), 3); s, _ = smooth_step(t - k)
    return corners[k] + (corners[k + 1] - corners[k]) * s


os_ = fs.run(xml_s, corners[0], ref2, 4.5)
per = D(os_["theta"][np.argmin(abs(os_["t"] - 4.0))])
res["builder_check"]["space_loop_deg"] = per
print(f"  宇宙の人型（Sim02 のループ 1 周）: 組み立て {per:+.4f}°（手作りの段階 1: +2.963°、基準 Sim02: +2.9778°）")

# ---- 2. ギャラリー -------------------------------------------------------------------
robots = ["arm_only", "tank_arm", "tank_waist_3axis", "space_arm_leg"]
stages = ["practice", "steps", "slope", "moon", "space"]
pose = {"arm_pitch": RAD(20), "arm_elbow1": RAD(-40), "arm_yaw": RAD(-20), "leg_pitch": RAD(-60), "waist": RAD(10)}
pairs = [("arm_only", "practice"), ("tank_arm", "steps"), ("tank_waist_3axis", "slope"), ("tank_arm", "moon"), ("space_arm_leg", "space")]
imgs = []
for rb, st in pairs:
    xml, jn = build(load("robots", rb), load("stages", st))
    kw = dict(lookat=(0, 0, 1.5), distance=3.5) if st == "space" else {}
    imgs.append((rb, st, snapshot(xml, jn, pose, **kw)))
fig, ax = plt.subplots(1, len(imgs), figsize=(4 * len(imgs), 3.6))
for a, (rb, st, im) in zip(ax, imgs):
    a.imshow(im); a.set_title(f"{rb}\non {st}", fontsize=10); a.axis("off")
fig.suptitle("Field03: assembled robots on stages (MuJoCo)", y=1.0); fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "field03_gallery.png", dpi=110)

# ---- 3. 試験メニューの例: 荷物運び（T = 2.0 s）を機体 × ステージで ---------------------------
print("\n=== 3. 試験メニュー: 荷物運び（前方の地面 → 頭上 → 後部, T = 2.0 s）===")
print("  機体                ステージ   最大の傾き  ずれ[mm]  浮き上がり[s]  最終の傾き")
menu_q0 = {"arm_pitch": Q0[0], "arm_elbow1": Q0[1]}
menu_q1 = {"arm_pitch": Q1[0], "arm_elbow1": Q1[1]}
for rb in ("tank_arm", "tank_waist_3axis"):
    for st in ("practice", "ice", "slope", "mars", "moon"):
        xml, jn = build(load("robots", rb), load("stages", st))
        _, s = run_menu(xml, jn, menu_q0, menu_q1, 2.0, min_jerk)
        res["menu"].append(dict(robot=rb, stage=st, **s))
        print(f"  {rb:18s}  {st:8s}  {s['max_tilt_deg']:7.2f}°  {s['slide_mm']:8.1f}  {s['airborne_s']:8.2f}      {s['final_tilt_deg']:6.2f}°")
json.dump(res, open(DATA / "field03_assembly.json", "w"), indent=2, ensure_ascii=False)
