"""番外編 05: 羽生 (2021) の慣性センサーの実験と、番外編 03 のモデルの突き合わせ.

羽生結弦 (2021) 「無線・慣性センサー式モーションキャプチャシステムのフィギュアスケートでの利活用に関するフィージビリティスタディ」
人間科学研究 補遺号 2021, 1–7。Perception Neuron 2.0（センサー 31 個）を着けて、1 回転・3 回転のループ・フリップ、1 回転半・3 回転半のアクセルを各 2 回。

論文には数値の結果（回転速度・慣性モーメント・腕の角度・高さ）が載っていない（F76）。載っているのは定性的な所見だけ:
  (a) 3 回転半のアクセルでも「回転はしっかりと追えていた」
  (b) 高さと跳躍距離は慣性式では取れない
  (c) 着氷時の膝の角度と回転の角度の精度は疑わしい
そこで、センサーの仕様（ジャイロ ±2000 °/s、加速度 ±16 g、31 個で 60 Hz。販売店の仕様表）と、番外編 03 のモデルが予測する
各部の角速度・加速度を比べ、所見 (a)〜(c) と矛盾しないかを調べる。

実行: python bangai05_sensor_check.py [main|mc|figure]
"""
from __future__ import annotations

import json
import sys

import mujoco
import numpy as np

import bangai03_axel_arms as A
from common import DATA, ROOT

OUT = ROOT / "figures" / "field"
GYRO_LIM = 2000.0          # °/s
ACC_LIM = 16.0             # g
FS = 60.0                  # 31 個のときの出力（120 Hz は 17 個まで）
G0 = 9.81

# センサーの位置（体の名前、局所の点）。Perception Neuron の 31 個のうち、体幹と手足の主なもの（指は除く）
def sites(B):
    S = B["seg"]
    zc7 = B["H"] - S["head"]["len"] - B["hipH"]
    zsh = zc7 - B["sh_below"]
    out = {"頭": ("trunk", [0, 0, zc7 + 0.6 * S["head"]["len"]]),
           "背中（胸の高さ）": ("trunk", [-0.10, 0, zsh - 0.10]),
           "腰": ("trunk", [-0.10, 0, 0.05])}
    for side, jp in (("L", "左"), ("R", "右")):
        out[f"{jp}上腕"] = (f"ua{side}", [0, 0, -0.6 * S["upperarm"]["len"]])
        out[f"{jp}前腕"] = (f"fa{side}", [0, 0, -0.7 * S["forearm"]["len"]])
        out[f"{jp}手"] = (f"fa{side}", [0, 0, -S["forearm"]["len"] - 0.4 * S["hand"]["len"]])
        out[f"{jp}太もも"] = (f"th{side}", [0, 0, -0.5 * S["thigh"]["len"]])
        out[f"{jp}すね"] = (f"sh{side}", [0, 0, -0.6 * S["shank"]["len"]])
        out[f"{jp}足"] = (f"ft{side}", [0, 0, -0.3 * S["foot"]["len"]])
    return out


def sensor_signals(sk, r):
    """fly(return_states=True) の結果から、各センサーの角速度の大きさ（°/s）と比力の大きさ（g）、胴体の長軸まわりの角速度."""
    m, d = sk.m, sk.d
    st = r["states"]
    ia = np.r_[sk.jadr["L"], sk.jadr["R"]]
    iv = ia - 1
    SS = sites(sk.B)
    ids = {k: (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b), np.array(p, float)) for k, (b, p) in SS.items()}
    n = len(st["t"])
    pos = {k: np.zeros((n, 3)) for k in SS}
    gyro = {k: np.zeros(n) for k in SS}
    wz = np.zeros(n)
    res = np.zeros(6)
    for i, t in enumerate(st["t"]):
        qa, qad, _ = st["plan"](t)
        d.qpos[:7] = st["y"][i]; d.qpos[ia] = qa
        d.qvel[:6] = st["vb"][i]; d.qvel[iv] = qad
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); mujoco.mj_comVel(m, d)
        for k, (bid, p) in ids.items():
            R = d.xmat[bid].reshape(3, 3)
            pos[k][i] = d.xpos[bid] + R @ p
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, res, 0)
            gyro[k][i] = np.degrees(np.linalg.norm(res[:3]))
        Rt = d.xmat[sk.trunk].reshape(3, 3)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, sk.trunk, res, 0)
        wz[i] = np.degrees(Rt[:, 2] @ res[:3])
    dt = st["t"][1] - st["t"][0]
    acc = {}
    for k in SS:
        a = np.gradient(np.gradient(pos[k], dt, axis=0), dt, axis=0)
        f = a - np.array([0, 0, -G0])           # 比力（加速度計が測る量）
        acc[k] = np.linalg.norm(f, axis=1) / G0
    return dict(t=st["t"], gyro=gyro, acc=acc, wz=wz)


def lost_rotation(t, wz, lim=GYRO_LIM, fs=FS):
    """ジャイロが ±lim で飽和し、fs で標本化したときに数え落とす回転（回転数）."""
    ts = np.arange(0, t[-1], 1 / fs)
    w = np.interp(ts, t, wz)
    true = np.trapezoid(wz, t) / 360
    clip = np.trapezoid(np.clip(w, -lim, lim), ts) / 360
    samp = np.trapezoid(w, ts) / 360
    return true, clip - true, samp - true


def run_case(sk, sched, L, t_air, dt=1e-3):
    r = sk.fly(sched, L, t_air, dt=dt, torque=False, return_states=True)
    s = sensor_signals(sk, r)
    true, dclip, dsamp = lost_rotation(s["t"], s["wz"])
    # 着氷の直前・直後の過渡を除くため、空中の 0.02〜t_air−0.02 s だけで最大を見る
    sel = (s["t"] > 0.02) & (s["t"] < t_air - 0.02)
    hold = (s["t"] > A.T_TR + 0.03) & (s["t"] < t_air - A.T_OPEN - 0.03)
    out = dict(turns=r["turns"], wz_peak=float(np.abs(s["wz"]).max()), turns_from_gyro=true, dturn_clip=dclip, dturn_60Hz=dsamp,
               gyro_peak={k: float(v[sel].max()) for k, v in s["gyro"].items()},
               acc_peak={k: float(v[sel].max()) for k, v in s["acc"].items()},
               # 腕を動かしている間（0.10 s の仮定の遷移）を除いた「保持中」の最大。F77 の切り分け
               acc_hold={k: float(v[hold].max()) for k, v in s["acc"].items()},
               t_gyro_sat=float(np.sum(np.abs(s["wz"]) > GYRO_LIM) * dt))
    return out, s


CASES = {"胸で組む（普通の 3A）": "tuck", "頭上で手を合わせる": "up", "肩幅で真上": "uppar", "半分の締め": "half"}


def main():
    B = A.load_body(A.REF["body"], A.REF["height"], A.REF["mass"], A.REF["boot"])
    sk = A.Skater(B)
    res = {}
    for nm, p in CASES.items():
        o, _ = run_case(sk, A.std_schedule(p), A.L_NOM, A.T_AIR_NOM)
        res[nm] = o
        sat_g = [k for k, v in o["gyro_peak"].items() if v > GYRO_LIM]
        sat_a = [k for k, v in o["acc_peak"].items() if v > ACC_LIM]
        print(f"{nm:14s} {o['turns']:.3f} 回転  胴体の最大 {o['wz_peak']:.0f} °/s（飽和 {o['t_gyro_sat'] * 1e3:.0f} ms）  "
              f"飽和で数え落とす {o['dturn_clip']:+.3f} 回転  60 Hz {o['dturn_60Hz']:+.4f}")
        print(f"    ジャイロ > 2000: {sat_g}")
        print(f"    加速度 > 16 g: {[(k, round(o['acc_peak'][k], 1)) for k in sat_a]}")
        print(f"    保持中の加速度 > 16 g: {[(k, round(v, 1)) for k, v in o['acc_hold'].items() if v > ACC_LIM]}  最大 {max(o['acc_hold'].values()):.1f} g")
    # 測定の装備の質量: センサー 31 × 約 1 g、電池 約 160 g（論文の値）。電池を腰の後ろ（軸から 0.12 m）と仮定
    I_tuck = sk.inertia("tuck")[0][2, 2]
    dI = 0.160 * 0.12**2 + 0.031 * 0.15**2
    res["装備"] = dict(dI=dI, dI_rel=dI / I_tuck, dturn=-res["胸で組む（普通の 3A）"]["turns"] * dI / (I_tuck + dI))
    print("装備の慣性モーメント:", {k: round(v, 5) for k, v in res["装備"].items()})
    json.dump(res, open(DATA / "bangai05_sensor_check.json", "w"), indent=1, ensure_ascii=False, default=float)
    return res


def mc(n=300, seed=5):
    """統計: 文献の L（32.8 ± 2.0）と滞空（0.68〜0.76 s）、体のばらつき（番外編 03 と同じ）のもとで、普通の 3A の胴体の最大角速度の分布."""
    rng = np.random.default_rng(seed)
    B = A.load_body(A.REF["body"], A.REF["height"], A.REF["mass"], A.REF["boot"])
    sk = A.Skater(B)
    # N も最大の ω も L に比例し、t_air にはほぼよらない（保持の間の ω = L / I）。I のばらつきは番外編 03 の MC（± 6.2%）
    base, _ = run_case(sk, A.std_schedule("tuck"), A.L_NOM, A.T_AIR_NOM)
    up, _ = run_case(sk, A.std_schedule("up"), A.L_NOM, A.T_AIR_NOM)
    err = json.load(open(DATA / "bangai03_errors.json"))["stat"]
    cvI = err["I_tuck_sd"] / err["I_tuck_mean"]
    L = A.L_NOM + A.L_SD * rng.standard_normal(n)
    fI = 1 + cvI * rng.standard_normal(n)
    w_tuck = base["wz_peak"] * L / A.L_NOM / fI
    w_up = up["wz_peak"] * L / A.L_NOM / fI
    out = dict(n=n, cvI=cvI, w_tuck_mean=w_tuck.mean(), w_tuck_sd=w_tuck.std(ddof=1), p_tuck_sat=float(np.mean(w_tuck > GYRO_LIM)),
               w_up_mean=w_up.mean(), w_up_sd=w_up.std(ddof=1), p_up_sat=float(np.mean(w_up > GYRO_LIM)),
               w_tuck_q=np.percentile(w_tuck, [5, 50, 95]).tolist(), w_up_q=np.percentile(w_up, [5, 50, 95]).tolist(),
               # King らの 4.9 rev/s（平均、5 人）は回転の平均の速さ。最大との比をモデルから
               ratio_peak_to_mean=base["wz_peak"] / (360 * base["turns"] / A.T_AIR_NOM))
    print(json.dumps(out, indent=1, default=lambda x: round(float(x), 4)))
    json.dump(out, open(DATA / "bangai05_mc.json", "w"), indent=1, default=float)
    return out


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    B = A.load_body(A.REF["body"], A.REF["height"], A.REF["mass"], A.REF["boot"])
    sk = A.Skater(B)
    R = json.load(open(DATA / "bangai05_sensor_check.json"))
    M = json.load(open(DATA / "bangai05_mc.json"))
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    ax = axs[0]
    for (nm, p), c in zip(list(CASES.items())[:3], ("C0", "C3", "C4")):
        _, s = run_case(sk, A.std_schedule(p), A.L_NOM, A.T_AIR_NOM)
        ax.plot(s["t"], s["wz"], color=c, label=f"{nm}")
    ax.axhline(GYRO_LIM, color="k", ls="--"); ax.text(0.01, GYRO_LIM + 30, "ジャイロの上限 2000 °/s（Perception Neuron 2.0）", fontsize=8)
    ax.set_xlabel("踏み切りからの時間 [s]"); ax.set_ylabel("胴体の長軸まわりの角速度 [°/s]")
    ax.set_title("(a) 胴体のセンサーが測る角速度（L 32.8、滞空 0.70 s）"); ax.legend(fontsize=8, loc="lower center")
    ax = axs[1]
    rng = np.random.default_rng(1)
    for key, c, lab in (("tuck", "C0", "胸で組む"), ("up", "C3", "頭上で合わせる")):
        mu, sd = M[f"w_{key}_mean"], M[f"w_{key}_sd"]
        x = np.linspace(1400, 3000, 400)
        ax.plot(x, np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), color=c,
                label=f"{lab}: {mu:.0f} ± {sd:.0f} °/s、上限を超える確率 {M[f'p_{key}_sat'] * 100:.0f}%")
    ax.axvline(GYRO_LIM, color="k", ls="--")
    ax.set_xlabel("胴体の最大角速度 [°/s]"); ax.set_ylabel("確率密度")
    ax.set_title("(b) 文献の L・体のばらつきのもとでの最大角速度"); ax.legend(fontsize=8)
    ax = axs[2]
    o = R["胸で組む（普通の 3A）"]
    keys = list(o["acc_peak"].keys())
    v1 = [o["acc_hold"][k] for k in keys]   # 遷移中の値は仮定の 0.10 s に強く依存するので使わない（F77）
    v2 = [R["頭上で手を合わせる"]["acc_hold"][k] for k in keys]
    y = np.arange(len(keys))
    ax.barh(y - 0.2, v1, 0.4, color="C0", label="胸で組む")
    ax.barh(y + 0.2, v2, 0.4, color="C3", label="頭上で合わせる")
    ax.axvline(ACC_LIM, color="k", ls="--"); ax.text(ACC_LIM + 0.3, len(keys) - 1, "加速度の上限 16 g", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(keys, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("腕を保持している間の比力の最大 [g]（着氷の衝撃・腕の遷移は含まない）"); ax.set_title("(c) 各センサーの加速度"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "bangai05_sensor_check.png", dpi=130)
    print("図 → figures/field/bangai05_sensor_check.png")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "main"
    {"main": main, "mc": mc, "figure": figure}[mode]()
