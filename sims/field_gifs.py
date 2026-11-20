"""シミュレーション場の主な場面を GIF 動画にする（目で動きを確かめる・説明に使う）.

出力: figures/field/gifs/*.gif（ラベルは英語）
  01_space_loop.gif        宇宙：腕と脚のループで向きを変える（段階 1、Sim02）
  02_pick_place_waist.gif  荷物運び：補正なし（転倒） vs 腰の補正（段階 6）
  03_biped_vs_tank.gif     二本脚：腰なし（転倒） / 腰あり vs 戦車型＋腰（段階 9）
  04_track_step.gif        履帯で 0.15 m の段差：腕を後ろ（転倒） vs 能動の腕（段階 10）
  05_track_length.gif      短い履帯で 0.20 m（転倒） vs 長い履帯で 0.30 m（段階 10）
  06_turning.gif           その場旋回と、腕を速く振ると車体が回る（段階 11）
実行: cd sims && MUJOCO_GL=osmesa python field_gifs.py [番号 ...]（約 15 分）
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import json
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import DATA, FIG
from ambac.field.assembly import build, load
from ambac.field.evaluate import _min_jerk, _targets
from ambac.field.ground import pitch_yaw

OUT = FIG / "field" / "gifs"; OUT.mkdir(parents=True, exist_ok=True)
R = np.radians
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    FONT_S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
except OSError:
    FONT = FONT_S = ImageFont.load_default()


class Scene:
    """1 つの画面: モデル、初期化、毎ステップの制御、カメラ、ラベル"""

    def __init__(self, xml, title, init=None, control=None, cam=None, follow=False, info=None):
        self.m = mujoco.MjModel.from_xml_string(xml); self.d = mujoco.MjData(self.m)
        self.title, self.control, self.follow, self.info = title, control, follow, info
        if init:
            init(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)
        c = dict(distance=3.0, azimuth=90, elevation=-10, lookat=(0, 0, 0.5)); c.update(cam or {})
        self.cam = mujoco.MjvCamera(); self.cam.distance = c["distance"]; self.cam.azimuth = c["azimuth"]
        self.cam.elevation = c["elevation"]; self.lookat = np.array(c["lookat"], float)

    def step_to(self, t):
        while self.d.time < t:
            if self.control:
                self.control(self.m, self.d)
            mujoco.mj_step(self.m, self.d)

    def frame(self, renderer):
        la = self.lookat.copy()
        if self.follow:
            la[0] += self.d.qpos[0]; la[1] += self.d.qpos[1]
        self.cam.lookat[:] = la
        renderer.update_scene(self.d, camera=self.cam)
        img = Image.fromarray(renderer.render())
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 0, img.width, 38], fill=(255, 255, 255))
        dr.text((6, 3), self.title, fill=(0, 0, 0), font=FONT)
        line = f"t = {self.d.time:4.1f} s"
        if self.info:
            line += "   " + self.info(self.m, self.d)
        dr.text((6, 21), line, fill=(60, 60, 60), font=FONT_S)
        return img


def record(name, scenes, T, fps=10, speed=1.0, size=(360, 270)):
    """scenes を左右に並べて T 秒（シミュレーションの時間）を記録。speed は再生の倍速"""
    rends = [mujoco.Renderer(sc.m, size[1], size[0]) for sc in scenes]
    frames = []
    n = int(T / speed * fps)
    for k in range(n + 1):
        t = k * speed / fps
        imgs = []
        for sc, rd in zip(scenes, rends):
            sc.step_to(t)
            imgs.append(sc.frame(rd))
        canvas = Image.new("RGB", (size[0] * len(imgs) + 4 * (len(imgs) - 1), size[1]), (200, 200, 200))
        for i, im in enumerate(imgs):
            canvas.paste(im, (i * (size[0] + 4), 0))
        frames.append(canvas.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    for rd in rends:
        rd.close()
    path = OUT / f"{name}.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print(f"  {path.name}: {len(frames)} コマ、{path.stat().st_size / 1e6:.1f} MB", flush=True)


def set_pose(m, d, pose):
    for j, v in pose.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        d.qpos[m.jnt_qposadr[jid]] = v
        a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
        if a >= 0:
            d.ctrl[a] = v


def tilt_info(m, d):
    p, _ = pitch_yaw(d.qpos)
    return f"base tilt {np.degrees(abs(p)):4.1f} deg"


# ---- 1. 宇宙のループ ----------------------------------------------------------------
def visual(xml):
    """見た目だけを変える（質量・慣性は inertial で別に決めているので物理は変わらない）: 胴体を半透明に、手足を太く"""
    return xml.replace('rgba="0.55 0.6 0.55 1"', 'rgba="0.55 0.6 0.55 0.45"').replace('size="0.035"', 'size="0.05"')


def gif_space():
    from ambac.field.space import base_angle
    rb = load("robots", "space_arm_leg"); rb["arms"][1]["color"] = "0.95 0.55 0.15 1"
    xml, joints = build(rb, load("stages", "space")); xml = visual(xml)
    C = R(np.array([[-90, -90], [90, -90], [90, -30], [-90, -30], [-90, -90]], float))

    def ref(t):
        t = max(t - 0.5, 0) % 4.0; k = min(int(t // 1.0), 3); u = t - k; s = 10 * u**3 - 15 * u**4 + 6 * u**5
        return C[k] + (C[k + 1] - C[k]) * s

    def ctl(m, d):
        q = ref(d.time); d.ctrl[0], d.ctrl[1] = q
    th0 = {}

    def info(m, d):
        a = np.degrees(base_angle(d.qpos)); th0.setdefault("a", a)
        loops = int(max(d.time - 0.5, 0) // 4)
        return f"loops {loops}   body rotation {a - th0['a']:+5.1f} deg"
    sc = Scene(xml, "Space: arm (blue) + leg (orange) loop, L = 0", init=lambda m, d: set_pose(m, d, dict(zip(joints, C[0]))),
               control=ctl, cam=dict(distance=4.0, azimuth=90, elevation=0, lookat=(0, 0, 1.5)), info=info)
    record("01_space_loop", [sc], T=32.5, fps=10, speed=4.0, size=(420, 360))


# ---- 2. 荷物運び：補正なし vs 腰 ---------------------------------------------------------------
def menu_scene(robot, title, T, offsets=None, stage="practice"):
    xml, joints = build(robot, load("stages", stage)); a, b = _targets(joints, "arm")
    A = np.array([a.get(j, 0.0) for j in joints]); B = np.array([b.get(j, 0.0) for j in joints])
    idx = {j: i for i, j in enumerate(joints)}

    def ext(u):
        e = np.zeros(len(joints))
        if offsets:
            for j, v in offsets(u).items():
                e[idx[j]] = v
        return e
    settle = 0.5

    def init(m, d):
        set_pose(m, d, dict(zip(joints, A + ext(0.0))))

    def ctl(m, d):
        u = np.clip((d.time - settle) / T, 0, 1)
        act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in joints]
        d.ctrl[act] = A + (B - A) * _min_jerk(u) + ext(u)
    return Scene(xml, title, init=init, control=ctl, cam=dict(distance=3.4, azimuth=60, elevation=-12, lookat=(0, 0, 0.7)), info=tilt_info)


def gif_pick_place():
    import field06_waist as f6
    r6 = json.load(open(DATA / "field06_waist.json"))["practice"]
    rb = load("robots", "tank_waist_3axis"); T = r6["waist"]["T"]
    s1 = menu_scene(rb, f"Pick & place in {T:.2f} s: no compensation", T)
    s2 = menu_scene(rb, f"Pick & place in {T:.2f} s: waist strategy", T, offsets=f6.offsets_of("waist", np.array(r6["waist"]["coef"])))
    record("02_pick_place_waist", [s1, s2], T=0.5 + T + 1.5, fps=12, speed=0.25)


# ---- 3. 二本脚 vs 戦車型 --------------------------------------------------------------------
def gif_biped():
    import field06_waist as f6
    import field09_biped as f9
    r9 = json.load(open(DATA / "field09_biped.json"))["practice"]; r6 = json.load(open(DATA / "field06_waist.json"))["practice"]
    prof = R(np.array(r9["balance_profile_deg"])); off9 = f9.make_offsets(prof)
    Tb = r9["static+dyn"]["T"]
    s1 = menu_scene(f9.BIPED, f"Biped, no waist, {Tb:.2f} s", Tb)
    s2 = menu_scene(f9.BIPED, f"Biped + waist, {Tb:.2f} s", Tb, offsets=off9("static+dyn", np.array(r9["static+dyn"]["coef"])))
    Tt = r6["waist"]["T"]
    s3 = menu_scene(load("robots", "tank_waist_3axis"), f"Tank + waist, {Tt:.2f} s", Tt,
                    offsets=f6.offsets_of("waist", np.array(r6["waist"]["coef"])))
    for s in (s1, s2, s3):
        s.cam.distance = 3.8; s.lookat = np.array([0, 0, 0.9])
    record("03_biped_vs_tank", [s1, s2, s3], T=0.5 + Tb + 1.5, fps=12, speed=0.25, size=(300, 280))


# ---- 4・5. 履帯で段差 ---------------------------------------------------------------------------
def track_scene(robot, h, strat, title):
    import field10_linked_track as f10
    xml, joints = build(robot, f10.stage_of("step", h))
    pose = f10.POSES.get(strat, f10.POSES["down"])

    def ctl(m, d):
        drv = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_sprocket")
        d.ctrl[drv] = (0.15 / 0.10) if d.time > 0.5 else 0.0
        if strat == "active" and d.time > 0.5:
            p, _ = pitch_yaw(d.qpos)
            a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm_pitch"); d.ctrl[a] = f10.active(d.time, {"pitch": p})["arm_pitch"]

    def info(m, d):
        p, _ = pitch_yaw(d.qpos)
        return f"nose-up {np.degrees(-p):5.1f} deg   x {d.qpos[0]:4.2f} m"
    return Scene(xml, title, init=lambda m, d: set_pose(m, d, pose), control=ctl,
                 cam=dict(distance=2.6, azimuth=90, elevation=-5, lookat=(0.2, 0, 0.35)), follow=True, info=info)


def gif_track_step():
    import field10_linked_track as f10
    s1 = track_scene(f10.ROBOT, 0.15, "back", "0.15 m step, arm folded back")
    s2 = track_scene(f10.ROBOT, 0.15, "active", "0.15 m step, active arm (ground AMBAC)")
    record("04_track_step", [s1, s2], T=14.0, fps=10, speed=2.0)


def gif_track_length():
    import field10_linked_track as f10
    s1 = track_scene(f10.ROBOT, 0.20, "active", "Track 0.8 m, 0.20 m step")
    s2 = track_scene(f10.variant("long"), 0.30, "active", "Track 1.2 m, 0.30 m step")
    record("05_track_length", [s1, s2], T=18.0, fps=10, speed=2.0)


# ---- 6. 旋回と腕の振り --------------------------------------------------------------------------
def gif_turning():
    rb = load("robots", "twin_track_arm"); xml, joints = build(rb, load("stages", "practice"))
    yaw0 = {}

    def yinfo(key):
        def f(m, d):
            _, y = pitch_yaw(d.qpos); yaw0.setdefault(key, y)
            return f"base heading {np.degrees(np.unwrap([yaw0[key], y])[1] - yaw0[key]):+6.1f} deg"
        return f

    def pivot(m, d):
        L = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_L_sprocket"); Rr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_R_sprocket")
        on = d.time > 0.5; d.ctrl[L] = 1.5 if on else 0; d.ctrl[Rr] = -1.5 if on else 0

    def swing(m, d):
        a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm_yaw")
        u = np.clip((d.time - 1.5) / 1.0, 0, 1); d.ctrl[a] = R(180) * (10 * u**3 - 15 * u**4 + 6 * u**5)
    s1 = Scene(xml, "Pivot turn: tracks +0.15 / -0.15 m/s", init=lambda m, d: set_pose(m, d, {"arm_pitch": R(-90)}), control=pivot,
               cam=dict(distance=3.0, azimuth=90, elevation=-60, lookat=(0, 0, 0.3)), info=yinfo("p"))
    s2 = Scene(xml, "Fast arm swing (1 s) with 10 kg payload", init=lambda m, d: set_pose(m, d, {"arm_pitch": 0.0}), control=swing,
               cam=dict(distance=3.4, azimuth=90, elevation=-60, lookat=(0, 0, 0.3)), info=yinfo("s"))
    record("06_turning", [s1, s2], T=6.0, fps=10, speed=1.0)


ALL = {"1": gif_space, "2": gif_pick_place, "3": gif_biped, "4": gif_track_step, "5": gif_track_length, "6": gif_turning}
if __name__ == "__main__":
    for k in (sys.argv[1:] or ALL):
        print(f"GIF {k}", flush=True); ALL[k]()
