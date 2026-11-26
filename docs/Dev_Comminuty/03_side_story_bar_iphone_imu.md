---
title: "[Side Story] One Aluminum Bar and Two iPhones: Does My Home-Made Rigid-Body Simulator Match Reality?"
tags: [iphone, phyphox, imu, python, mujoco]
published: false
---

> **This article is a draft.** The experiment has not been done yet. Theoretical predictions, the procedure and the error-budget plan are written first; measured values are left as **[not measured]**. I will fill them in and publish after measuring.

> A side story of the series. In the main articles I tested Gundam's AMBAC (changing attitude by the reaction of the limbs) with CPU-only simulation, without a robot or a GPU ([entry article](TODO)). In this side story I finally use a **real object**.

## Why a side story?

One thing has been bothering me about the simulations in the main series: **they have never been compared with reality.**

So far I have validated the simulator in two layers:

1. **Reference model**: equations written by hand (Python). The answers are known.
2. **Simulation field**: the MuJoCo physics engine, assembling machines from a parts list.

These two agreed to several decimal places (−16.0356° vs −16.0356° for the arm swing). But both stand on the same assumption: "this is how a rigid body should move". Whether that assumption matches reality can only be checked against **a third layer: a real object**.

Moreover, in the main series' calibration it turned out that **the virtual robot's own integrator (the simulator itself) produced a systematic error of +1–2 %**. The simulator can itself be a source of error. That's exactly why I want to firm up the ground first with the simplest real object whose answer is known.

I can't afford a robot. But **I do have an aluminum bar and the iPhones at hand.**

## What you need

| Item | Spec | Role |
|---|---|---|
| Aluminum angle | 20 × 20 × 2.0 mm, 1 m long, about 204 g | The rigid body |
| iPhone SE (3rd gen.) × 2 | about 144 g, 138.4 × 67.3 × 7.3 mm | IMU (accelerometer + gyro), and weights |
| phyphox | Free app; records raw sensor data and exports CSV | Recording |
| Kitchen scale | 1 g resolution | Weighing |
| Tape measure, drill, bolt | Drill a hole about 2 cm from one end as the pivot | Pendulum pivot |
| Double-sided tape, cable ties | Fixing the iPhones | |

The mass of the angle, from the cross-section $2\times20\times2-2^2 = 76\ \mathrm{mm^2}$ and density 2.70 g/cm³, is $76\times10^{-6}\times1.0\times2700 \approx 0.205$ kg, in good agreement with the catalog's ~204 g.

Here's an important observation: **one iPhone is 144 g, two are 288 g — heavier than the bar (204 g).** So the iPhones are not "sensors attached to the bar" but **most of the rigid body**. In the simulator they must go in as one rigid body assembled from the bar and two iPhones. This is the smallest real-world version of the main series' "assemble a machine from a parts list".

## What we check

Two properties with known answers that hold precisely because it is a rigid body.

### Experiment A: moment of inertia from the pendulum period

Use the hole near one end as the pivot and swing the bar gently in a vertical plane (a physical pendulum). With moment of inertia $I$ about the pivot, total mass $M$, and distance $d$ from the pivot to the center of mass, the small-amplitude period is

$$
T_0 = 2\pi\sqrt{\frac{I}{M g d}}.
$$

At larger amplitude $\theta_0$ the period lengthens slightly, $T \approx T_0\left(1 + \theta_0^2/16\right)$ (+0.19 % at 10°).

$I$ can be computed from the parts. With the pivot at $h$ from the end and the iPhone centers at $r_1, r_2$ from the pivot,

$$
I = \underbrace{\frac{m_b L^2}{12} + m_b\left(\frac{L}{2}-h\right)^2}_{\text{bar}} + \sum_{i=1,2}\left[\underbrace{\frac{m_p(a^2+b^2)}{12}}_{\text{iPhone itself}} + m_p r_i^2\right]
$$

($a, b$: the iPhone's length and width, mounted with the screen parallel to the plane of swing).

**Predictions** ($h = 0.02$ m, $g = 9.80$ m/s²):

| Configuration | $I$ [kg·m²] | $M$ [kg] | $d$ [m] | Predicted $T_0$ [s] | Measured $T$ [s] |
|---|---|---|---|---|---|
| Bar only (reference: ideal value without iPhones) | 0.0640 | 0.204 | 0.480 | 1.623 | [not measured] |
| iPhones at 0.30 m and 0.90 m | 0.1942 | 0.492 | 0.550 | 1.700 | [not measured] |
| iPhones at 0.48 m and 0.95 m | 0.2277 | 0.492 | 0.618 | 1.738 | [not measured] |

*In the actual comparison, always include the iPhone used for recording*

Changing the configuration changes the period by only about 5 %. So **I want to measure the period to 0.1 %**. Recording 20 periods with the phyphox gyro (about 100 Hz) and interpolating the zero-crossing times should get below 0.05 % per period (this, too, is an estimate until measured).

### Experiment B: when rotated, are both angular velocities equal, and is centripetal acceleration proportional to distance?

Hold the bar horizontal and rotate it about the pivot (by hand or on a turntable). For a rigid body,

- the **angular velocity $\omega$ of both iPhones is equal** (wherever they are attached), and
- the centripetal acceleration is $a_c = \omega^2 r$, **proportional to the distance $r$ from the axis**.

For example, at 0.5 rev/s ($\omega = \pi$ rad/s), the iPhone at 0.30 m sees 2.96 m/s², and the one at 0.90 m sees 8.88 m/s².

| Quantity | Prediction | Measured |
|---|---|---|
| Ratio of $\omega$ of the two phones, $\omega_2/\omega_1$ | 1.000 | [not measured] |
| Ratio of centripetal accelerations $a_{c2}/a_{c1}$ | $r_2/r_1 = 3.00$ | [not measured] |
| Slope of $a_c$ vs $\omega^2$ (phone 1) | $r_1 = 0.30$ m | [not measured] |
| Slope of $a_c$ vs $\omega^2$ (phone 2) | $r_2 = 0.90$ m | [not measured] |

The $r$ obtained from the slope of $a_c$ vs $\omega^2$ should differ slightly from the tape-measured position, because **the iPhone's accelerometer is not at the center of the body**. This offset is exactly a calibration value — "the IMU mounting position" — and I will obtain it with the main series' kinematic-fit approach, fitting $r$ and the sensor bias simultaneously.

## How to compare with the simulator

1. **Write the bar and the two iPhones into the parts list (JSON).** Masses from the scale, dimensions from the tape measure.
2. **Run the same motion in MuJoCo.** For Experiment A, release from the initial angle; for Experiment B, drive it with the measured $\omega(t)$. Use the RK4 integrator (the lesson of the main series).
3. **Place virtual IMUs inside the simulator**, recording acceleration and gyro at the same positions and sampling as the iPhones.
4. **Run the same analysis on both measurement and simulation**: period, ratio of $\omega$, slope of $a_c$ vs $\omega^2$, with the same script.
5. **Judge the difference divided by the error.** Don't decide "match / no match" by eye; use pull $= (\text{measured} - \text{predicted})/\sigma$.

TODO: comparison script (planned as `sims/bangai01_bar_imu.py`) and measured CSVs (`data/bangai01/`).

## Error budget (plan)

As in the main series, split into statistical, computational and systematic. Numbers are prior estimates, to be replaced after measuring.

| Type | Source | Estimated effect on the period |
|---|---|---|
| Statistical | Reading zero-crossing times (~100 Hz, 20 periods) | about 0.05 % |
| Systematic | Weighing (±1 g) | about 0.1 % |
| Systematic | iPhone position (±1 mm, phone at 0.90 m) | about 0.07 % (0.13 % in $I$) |
| Systematic | Pivot position and hole play (±1 mm) | about 0.1 % |
| Systematic | Amplitude correction (+0.19 % at 10°) | about 0.01 % residual after correction |
| Systematic | Pivot friction and air drag (slightly lengthen the period) | estimate from damping [not measured] |
| Systematic | Bending of the bar (rigid-body assumption) | check with the rotation experiment [not measured] |
| Systematic | Gravitational acceleration (≈9.798 m/s² near Tokyo) | use the local value, below 0.01 % |
| Computational | MuJoCo integrator and time step | RK4, compare 0.1 ms and 0.5 ms |

The angle's own cross-section inertia (the thickness effect) is about 0.02 % of the total and negligible.

## Results

[not measured]

## Discussion

[not measured]

Three things I want to see:

- **How far does the rigid-body assumption hold?** A 1 m aluminum angle should bend when spun fast. The rotation rate at which the ratio of $\omega$ starts to deviate from 1 gives the range in which it can be treated as rigid.
- **Which is larger, the simulator's error or the measurement error?** In the main series, the simulator's own integrator was a source of systematic error. Now I check that against a real object.
- **Can IMU calibration values (mounting position, bias) be obtained by kinematic fit?** This is the first step toward calibrating a real machine.

## Next: real AMBAC

Once the ground is firm with a rigid body and IMUs, the next experiment is to **sit on a swivel chair and swing this bar instead of a limb**. With one iPhone on the chest recording the body's heading and the bar's iPhone recording the "arm" motion, the main series' basic equation $\dot\theta = -\Gamma(q)\,\dot q$ and the net rotation per loop (the geometric phase) can be checked with a real object. It's also fun that the friction of the chair's axle plays the role of the main series' "friction threshold under the machine".

I can't become an astronaut or a mobile-suit pilot, but **on a swivel chair I can test AMBAC with my own body**.

## References

- phyphox (RWTH Aachen University): https://phyphox.org/
- Code and numbers of the main series: https://github.com/okamasaya/simulation_AMBAC
