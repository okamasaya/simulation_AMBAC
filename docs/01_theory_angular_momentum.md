# 01. 理論的準備：角運動量保存と AMBAC

## 1.1 AMBAC とは何か（物理的に言い直す）

AMBAC（Active Mass Balance Auto Control）は、『機動戦士ガンダム』[R1] の SF 考証として示された概念で [R42]、
「四肢を動かした反作用で姿勢を制御する」と説明される。物理学の言葉にすると次の 1 文に尽きる。

> **外トルクのない系で、内部の質量配置（形状）を変えることにより、全角運動量 L を保存したまま
> 胴体の向きを変える制御。**

推進剤を消費しないこと、反力を外部に依存しないことが本質的な利点である。
一方で、**全角運動量そのものは 1 ミリも変えられない**（回転を止める・始めることは単独ではできない）
という原理的制約がある。この「できること / できないこと」の境界を明確にするのが本リポジトリの第一目標。

## 1.2 平面自由浮遊マルチボディ系

胴体（姿勢角 θ）に n 本のリンク（関節角 q = (q₁,…,qₙ)）が付いた系を考える。
外力ゼロなので重心は等速直線運動、重心系で見れば静止とみなせる。
系重心まわりの全角運動量は

$$
L = \sum_i \left[ I_i \dot\phi_i + m_i (\mathbf r_i - \mathbf c) \times (\dot{\mathbf r}_i - \dot{\mathbf c}) \right]
$$

（φᵢ: 各剛体の絶対角、rᵢ: 各剛体重心、c: 系重心）。
回転対称性から L は θ に依存せず、速度について線形なので

$$
L = A(q)\,\dot\theta + B(q)\cdot\dot q
$$

と書ける。A(q) > 0 は「形状 q で固まった系全体の慣性モーメント」、B(q) は関節運動と姿勢の結合項。

## 1.3 AMBAC の基本式（L = 0 の場合）

静止状態から始めれば L ≡ 0 なので

$$
\boxed{\ \dot\theta = -\,\Gamma(q)\cdot\dot q,\qquad \Gamma(q) = B(q)/A(q)\ }
$$

Γ は幾何力学で **機械的接続（mechanical connection）** と呼ばれる量。重要な性質:

1. **時間に依存しない。** θ の変化量は関節空間の「経路」だけで決まり、速さには依存しない。
   ゆっくり動かしても速く動かしても回転角は同じ（必要トルクとエネルギーが変わるだけ）。
2. **スケール不変。** Γ は質量比・長さ比のみの関数。人間大でも 18 m 級でも、
   幾何学的に相似で質量分布が相似なら回転角は同じ（Sim01 で数値確認済み）。
   大型化で効くのはトルク要求・構造強度・時間スケールの方。

## 1.4 可逆的 AMBAC と非ホロノミック AMBAC

**関節 1 つ（n = 1）** の場合、θ = -∫Γ(q)dq は q だけの関数になり、
腕を戻すと姿勢も必ず元に戻る（**可逆的・ホロノミック**）。
「腕を振った分だけ一時的に向きを変える」ことしかできず、回転可能範囲も上限がある。

**関節 2 つ以上（n ≥ 2）** で、関節空間の閉ループ C を 1 周すると、形状は元に戻るのに

$$
\Delta\theta = -\oint_C \Gamma\cdot dq = -\iint_S F\,dq_1 dq_2,\qquad
F = \frac{\partial \Gamma_2}{\partial q_1} - \frac{\partial \Gamma_1}{\partial q_2}
$$

だけ姿勢が進む（ストークスの定理）。F は接続の **曲率**。これは **幾何学的位相（geometric phase）**
の一例で、落下する猫が空中で向きを変えるのと同じ原理である。
ループを繰り返せば **任意の角度まで回転できる**。これが「AMBAC で姿勢を自在に変える」ことの数学的な正体。

設計上の含意:
- 1 周あたりの回転量は「ループが囲む曲率の量」で決まる → **曲率の大きい形状領域でループを回すべき**。
- 曲率マップは機体設計（四肢の質量・長さ・取付位置）で決まる → **機体設計そのものが AMBAC 性能を決める**。

## 1.5 L ≠ 0 の場合（回転を止められるか？）

L ≠ 0 なら θ̇ = (L − B·q̇)/A(q)。内部運動では L は変えられないが、
**A(q) を大きくする（四肢を伸ばす）ことで回転速度を下げる**ことはできる（フィギュアスケートの逆）。
完全停止には外部トルク（スラスタ、リアクションホイール、環境力）が必要。
→ 今後のシミュレーション課題。

## 1.6 フィジカル AI との接点（予告）

- 姿勢制御が **非ホロノミック拘束下の経路計画問題** になる（Brockett 条件により、
  滑らかな時不変フィードバックでは原点を漸近安定化できない → 周期入力・時変制御・学習が必要）。
- 動作が「経路のみ」に依存するため、**形状空間での計画**と**トルク・時間の配分**を分離できる。
- 曲率マップは学習した方策の解釈手段（なぜその動きを選んだか）としても使える。

## 参考文献

- [R1] 『機動戦士ガンダム』（テレビアニメ, 1979 年, 日本サンライズ）。AMBAC を備えた人型機動兵器（モビルスーツ）の作品世界。※ AMBAC という用語・概念は劇中ではなく、設定考証 [R42] で示されたものとされる。
- [R42] 『ガンダムセンチュリー』（ムック, みのり書房, 1981 年）。AMBAC（Active Mass Balance Auto Control）の概念が SF 考証として最初に示された資料とされる。スタジオぬえのメンバーらが考証に関わったとされる。みのり書房は 1990 年代に活動を終えており、2000 年に樹想社から復刻版『GUNDAM CENTURY RENEWAL VERSION 宇宙翔ける戦士達』（ISBN 4-87777-028-3）が出版されている。
- [R4] R. Montgomery, "Gauge theory of the falling cat," in M. J. Enos (ed.), *Dynamics and Control of Mechanical Systems: The Falling Cat and Related Problems*, Fields Institute Communications, vol. 1, American Mathematical Society, 1993.
- [R5] J. E. Marsden, *Lectures on Mechanics*, London Mathematical Society Lecture Note Series 174, Cambridge University Press, 1992.
- [R6] A. Shapere and F. Wilczek (eds.), *Geometric Phases in Physics*, World Scientific, 1989.
- [R7] T. R. Kane and M. P. Scher, "A dynamical explanation of the falling cat phenomenon," *International Journal of Solids and Structures*, 5(7), 663–670, 1969.
- [R8] T. R. Kane and M. P. Scher, "Human self-rotation by means of limb movements," *Journal of Biomechanics*, 3(1), 39–49, 1970. doi:10.1016/0021-9290(70)90049-7 ✓
- [R9] R. W. Brockett, "Asymptotic stability and feedback stabilization," in R. W. Brockett, R. S. Millman, H. J. Sussmann (eds.), *Differential Geometric Control Theory*, Birkhäuser, pp. 181–191, 1983.
- [R10] R. M. Murray and S. S. Sastry, "Nonholonomic motion planning: steering using sinusoids," *IEEE Transactions on Automatic Control*, 38(5), 700–716, 1993.
- [R12] Y. Nakamura and R. Mukherjee, "Nonholonomic path planning of space robots via a bidirectional approach," *IEEE Transactions on Robotics and Automation*, 7(4), 500–514, 1991. doi:10.1109/70.86080 ✓
- [R13] E. Papadopoulos and S. Dubowsky, "On the nature of control algorithms for free-floating space manipulators," *IEEE Transactions on Robotics and Automation*, 7(6), 750–758, 1991.
- [R14] Y. Umetani and K. Yoshida, "Resolved motion rate control of space manipulators with generalized Jacobian matrix," *IEEE Transactions on Robotics and Automation*, 5(3), 303–314, 1989.
- [R15] D. N. Nenchev, K. Yoshida, P. Vichitkulsawat, M. Uchiyama, "Reaction null-space control of flexible structure mounted manipulator systems," *IEEE Transactions on Robotics and Automation*, 15(6), 1011–1023, 1999.

（全文献の一覧と照合状況は [references.md](references.md) を参照）
