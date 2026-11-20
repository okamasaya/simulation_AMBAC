"""評価の窓口: 機体の部品表（JSON）とステージ（JSON）を入れると、性能表（JSON）を返す.

DIY_AutoML やシミュレーテッドアニーリングから、評価関数として呼び出すことを想定している。

コマンドとして（リポジトリのルートで）:
    PYTHONPATH=src python -m ambac.field.evaluate configs/field/robots/tank_arm.json configs/field/stages/moon.json
関数として:
    from ambac.field.evaluate import evaluate
    result = evaluate(robot_dict, stage_dict)
"""
from __future__ import annotations

import json
import sys

import numpy as np

from .assembly import build
from .menu import run_menu

# 荷物運びの標準動作（前方の地面 → 頭上 → 後部）。腕の俯仰（pitch）と肘を使う
PICK_PLACE = ({"pitch": np.radians(-20.0), "elbow1": np.radians(-50.0)},
              {"pitch": np.radians(130.0), "elbow1": np.radians(60.0)})
TILT_LIMIT_DEG, SLIDE_LIMIT_MM = 1.0, 5.0


def _min_jerk(t):
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def _targets(joints, arm_name):
    q0, q1 = PICK_PLACE
    a = {f"{arm_name}_{k}": v for k, v in q0.items() if f"{arm_name}_{k}" in joints}
    b = {f"{arm_name}_{k}": v for k, v in q1.items() if f"{arm_name}_{k}" in joints}
    return a, b


def safe_time(robot, stage, t_lo=0.3, t_hi=12.0, iters=8):
    """荷物運びを、傾き 1° 未満・ずれ 5 mm 未満で実行できる最短時間（二分法）。土台が固定なら None"""
    if robot["base"]["type"] == "fixed" or not robot.get("arms"):
        return None, None
    xml, joints = build(robot, stage)
    a, b = _targets(joints, robot["arms"][0]["name"])

    def ok(T):
        _, s = run_menu(xml, joints, a, b, T, _min_jerk)
        return s["max_tilt_deg"] < TILT_LIMIT_DEG and s["slide_mm"] < SLIDE_LIMIT_MM, s
    good, s_hi = ok(t_hi)
    if not good:
        return np.inf, s_hi
    s_best = s_hi
    for _ in range(iters):
        mid = (t_lo + t_hi) / 2
        g, s = ok(mid)
        if g:
            t_hi, s_best = mid, s
        else:
            t_lo = mid
    return t_hi, s_best


def evaluate(robot, stage):
    """性能表（辞書）を返す"""
    out = {"robot": robot.get("name"), "stage": stage.get("name"), "gravity": stage.get("gravity"),
           "friction": stage.get("friction")}
    if stage.get("gravity", 9.8) == 0:
        out["note"] = "宇宙では荷物運びの転倒・滑りは評価しない"
        return out
    T, s = safe_time(robot, stage)
    out["safe_pick_place_time_s"] = None if T is None else (None if np.isinf(T) else round(float(T), 3))
    out["pick_place_feasible"] = T is not None and not np.isinf(T)
    if s:
        out["at_safe_time"] = {k: round(v, 3) for k, v in s.items()}
    return out


if __name__ == "__main__":
    robot = json.load(open(sys.argv[1], encoding="utf-8")); stage = json.load(open(sys.argv[2], encoding="utf-8"))
    print(json.dumps(evaluate(robot, stage), ensure_ascii=False, indent=2))
