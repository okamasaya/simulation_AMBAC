"""参考文献の一覧（docs/references.md）と、各章末の「参考文献」節を生成する.

使い方（リポジトリのルートで）:
    python tools/references.py
文献を追加・修正するときは、このファイルの R・GROUPS・USE を編集して再実行する。
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

R = {
    # A. 作品・設定・報道
    "R1": "『機動戦士ガンダム』（テレビアニメ, 1979 年, 日本サンライズ）。AMBAC を備えた人型機動兵器（モビルスーツ）の作品世界。"
          "※ AMBAC という用語・概念は劇中ではなく、設定考証 [R42] で示されたものとされる。",
    "R2": "『機動戦士ガンダム』第 42 話「宇宙要塞ア・バオア・クー」。整備兵がシャアに、脚のない機体ジオングについて言う台詞"
          "「あんなの飾りです。偉い人にはそれがわからんのですよ」。これが広まる過程で「足（脚）なんて飾りです」の形で知られるようになった。",
    "R3": "『アーマード・コア』シリーズ（フロム・ソフトウェア, 1997 年〜）。パーツを組み合わせて機体を構成する「アセンブル」の例として参照。",
    "R42": "『ガンダムセンチュリー』（ムック, みのり書房, 1981 年）。AMBAC（Active Mass Balance Auto Control）の概念が SF 考証として"
           "最初に示された資料とされる。スタジオぬえのメンバーらが考証に関わったとされる。みのり書房は 1990 年代に活動を終えており、"
           "2000 年に樹想社から復刻版『GUNDAM CENTURY RENEWAL VERSION 宇宙翔ける戦士達』（ISBN 4-87777-028-3）が出版されている。",
    "R43": "オリコンニュースの記事. https://www.oricon.co.jp/news/2385521/full/ "
           "大西卓哉さんの X への投稿 [R44] を紹介し、宇宙での姿勢制御を「ガンダムの AMBAC」と表現したことがガンダムファンの間で話題になったことを伝える記事"
           "（角運動量保存則などの物理の解説記事ではない）。※ 記事名は未照合。",
    "R44": "大西卓哉（@Astro_Onishi）の X への投稿, 2025 年 5 月 18 日. https://x.com/Astro_Onishi/status/1924053663238046048 "
           "（国際宇宙ステーションで脚を使って姿勢を制御し、AMBAC に言及したもの）※ 本リポジトリの作業環境からは閲覧できず、本文は未照合。",
    # B. 幾何力学
    "R4": "R. Montgomery, \"Gauge theory of the falling cat,\" in M. J. Enos (ed.), *Dynamics and Control of Mechanical Systems: "
          "The Falling Cat and Related Problems*, Fields Institute Communications, vol. 1, American Mathematical Society, 1993.",
    "R5": "J. E. Marsden, *Lectures on Mechanics*, London Mathematical Society Lecture Note Series 174, Cambridge University Press, 1992.",
    "R6": "A. Shapere and F. Wilczek (eds.), *Geometric Phases in Physics*, World Scientific, 1989.",
    "R7": "T. R. Kane and M. P. Scher, \"A dynamical explanation of the falling cat phenomenon,\" "
          "*International Journal of Solids and Structures*, 5(7), 663–670, 1969.",
    "R8": "T. R. Kane and M. P. Scher, \"Human self-rotation by means of limb movements,\" *Journal of Biomechanics*, 3(1), 39–49, 1970. "
          "doi:10.1016/0021-9290(70)90049-7 ✓",
    "R9": "R. W. Brockett, \"Asymptotic stability and feedback stabilization,\" in R. W. Brockett, R. S. Millman, H. J. Sussmann (eds.), "
          "*Differential Geometric Control Theory*, Birkhäuser, pp. 181–191, 1983.",
    "R10": "R. M. Murray and S. S. Sastry, \"Nonholonomic motion planning: steering using sinusoids,\" "
           "*IEEE Transactions on Automatic Control*, 38(5), 700–716, 1993.",
    "R11": "M. P. do Carmo, *Differential Geometry of Curves and Surfaces*, Prentice-Hall, 1976.（ガウス・ボンネの定理）",
    # C. 宇宙ロボット
    "R12": "Y. Nakamura and R. Mukherjee, \"Nonholonomic path planning of space robots via a bidirectional approach,\" "
           "*IEEE Transactions on Robotics and Automation*, 7(4), 500–514, 1991. doi:10.1109/70.86080 ✓",
    "R13": "E. Papadopoulos and S. Dubowsky, \"On the nature of control algorithms for free-floating space manipulators,\" "
           "*IEEE Transactions on Robotics and Automation*, 7(6), 750–758, 1991.",
    "R14": "Y. Umetani and K. Yoshida, \"Resolved motion rate control of space manipulators with generalized Jacobian matrix,\" "
           "*IEEE Transactions on Robotics and Automation*, 5(3), 303–314, 1989.",
    "R15": "D. N. Nenchev, K. Yoshida, P. Vichitkulsawat, M. Uchiyama, \"Reaction null-space control of flexible structure mounted "
           "manipulator systems,\" *IEEE Transactions on Robotics and Automation*, 15(6), 1011–1023, 1999.",
    "R16": "B. Wie, *Space Vehicle Dynamics and Control*, 2nd ed., AIAA, 2008.（リアクションホイール・CMG・スラスタによる姿勢制御）",
    "R17": "M. J. Sidi, *Spacecraft Dynamics and Control: A Practical Engineering Approach*, Cambridge University Press, 1997.",
    "R18": "M. D. Shuster, \"A survey of attitude representations,\" *Journal of the Astronautical Sciences*, 41(4), 439–517, 1993.（回転の表現と特異点）",
    # D. 学習・最適化
    "R19": "T. Salimans, J. Ho, X. Chen, S. Sidor, I. Sutskever, \"Evolution strategies as a scalable alternative to reinforcement learning,\" "
           "arXiv:1703.03864, 2017.",
    "R20": "D. P. Kingma and J. Ba, \"Adam: A method for stochastic optimization,\" *ICLR*, 2015. arXiv:1412.6980.",
    "R21": "A. Iscen, K. Caluwaerts, J. Tan, T. Zhang, E. Coumans, V. Sindhwani, V. Vanhoucke, \"Policies modulating trajectory generators,\" "
           "*Proceedings of the 2nd Conference on Robot Learning (CoRL)*, PMLR 87, 916–926, 2018. ✓",
    "R22": "A. J. Ijspeert, \"Central pattern generators for locomotion control in animals and robots: A review,\" "
           "*Neural Networks*, 21(4), 642–653, 2008.",
    "R23": "J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, P. Abbeel, \"Domain randomization for transferring deep neural networks "
           "from simulation to the real world,\" *IEEE/RSJ IROS*, 2017.",
    "R24": "X. B. Peng, P. Abbeel, S. Levine, M. van de Panne, \"DeepMimic: Example-guided deep reinforcement learning of physics-based "
           "character skills,\" *ACM Transactions on Graphics*, 37(4), 2018.（モーションキャプチャーを手本とする物理ベースの強化学習）",
    "R25": "Q. Huangfu and J. A. J. Hall, \"Parallelizing the dual revised simplex method,\" *Mathematical Programming Computation*, "
           "10(1), 119–142, 2018.（HiGHS: scipy の linprog で使用）",
    "R26": "D. Kraft, \"A software package for sequential quadratic programming,\" Technical Report DFVLR-FB 88-28, DLR, 1988."
           "（SLSQP: scipy の minimize で使用）",
    "R27": "T. Flash and N. Hogan, \"The coordination of arm movements: an experimentally confirmed mathematical model,\" "
           "*Journal of Neuroscience*, 5(7), 1688–1703, 1985.（最小躍度軌道）",
    # E. 地上
    "R28": "M. Vukobratović and B. Borovac, \"Zero-moment point — thirty five years of its life,\" "
           "*International Journal of Humanoid Robotics*, 1(1), 157–173, 2004.",
    "R29": "R. McN. Alexander and A. S. Jayes, \"A dynamic similarity hypothesis for the gaits of quadrupedal mammals,\" "
           "*Journal of Zoology*, 201(1), 135–152, 1983.（フルード数による相似則）",
    "R30": "D. Karnopp, \"Computer simulation of stick-slip friction in mechanical dynamic systems,\" "
           "*Journal of Dynamic Systems, Measurement, and Control*, 107(1), 100–103, 1985.",
    "R31": "F. B. Horak and L. M. Nashner, \"Central programming of postural movements: adaptation to altered support-surface "
           "configurations,\" *Journal of Neurophysiology*, 55(6), 1369–1381, 1986.（足首の戦略・腰の戦略）✓",
    "R32": "J. Y. Wong, *Theory of Ground Vehicles*, 4th ed., Wiley, 2008.（履帯車両の走行・段差乗り越え）",
    "R33": "G. Galilei, *Discorsi e dimostrazioni matematiche intorno a due nuove scienze*（新科学対話）, 1638.（二乗三乗則）",
    # F. 肩
    "R34": "E. A. Codman, *The Shoulder*, Thomas Todd, Boston, 1934.（コッドマンのパラドックス）",
    "R35": "J. C. Politti, G. Goroso, M. E. Valentinuzzi, O. Bravo, \"Codman's paradox of the arm rotations is not a paradox: "
           "mathematical validation,\" *Medical Engineering & Physics*, 20(4), 257–260, 1998. ✓（その後も解釈をめぐる議論がある）",
    "R36": "G. Wu et al., \"ISB recommendation on definitions of joint coordinate systems of various joints for the reporting of human "
           "joint motion — Part II: shoulder, elbow, wrist and hand,\" *Journal of Biomechanics*, 38(5), 981–992, 2005.（肩の角度の測り方の標準）",
    "R37": "V. T. Inman, J. B. Saunders, L. C. Abbott, \"Observations on the function of the shoulder joint,\" "
           "*Journal of Bone and Joint Surgery*, 26(1), 1–30, 1944.（肩甲上腕リズム）",
    "R38": "American Academy of Orthopaedic Surgeons, *Joint Motion: Method of Measuring and Recording*, AAOS, 1965.（関節可動域の標準値）",
    # G. ソフトウェア
    "R39": "C. R. Harris et al., \"Array programming with NumPy,\" *Nature*, 585, 357–362, 2020.",
    "R40": "P. Virtanen et al., \"SciPy 1.0: fundamental algorithms for scientific computing in Python,\" *Nature Methods*, 17, 261–272, 2020.",
    "R41": "J. D. Hunter, \"Matplotlib: A 2D graphics environment,\" *Computing in Science & Engineering*, 9(3), 90–95, 2007.",
}

GROUPS = [
    ("A. 出発点となった作品・設定・報道", ["R1", "R42", "R2", "R3", "R43", "R44"]),
    ("B. 幾何力学・幾何学的位相・非ホロノミック系", ["R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"]),
    ("C. 宇宙ロボット・人工衛星の姿勢制御", ["R12", "R13", "R14", "R15", "R16", "R17", "R18"]),
    ("D. 学習・最適化", ["R19", "R20", "R21", "R22", "R23", "R24", "R25", "R26", "R27"]),
    ("E. 地上での安定性・移動", ["R28", "R29", "R30", "R31", "R32", "R33"]),
    ("F. 肩・人体工学", ["R34", "R35", "R36", "R37", "R38"]),
    ("G. ソフトウェア", ["R39", "R40", "R41"]),
]

USE = {
    "00_purpose.md": ["R1", "R42", "R2", "R3", "R8", "R43", "R44", "R24", "R31"],
    "01_theory_angular_momentum.md": ["R1", "R42", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R12", "R13", "R14", "R15"],
    "02_sim_results_planar.md": ["R4", "R7", "R8", "R27"],
    "03_rw_vs_ambac.md": ["R5", "R15", "R16", "R17"],
    "04_dynamics_scaling.md": ["R16", "R33"],
    "05_physical_ai.md": ["R9", "R10", "R19", "R20", "R21", "R22", "R23", "R25"],
    "06_ground_tank.md": ["R26", "R27", "R28", "R29", "R30", "R33"],
    "07_spatial_3d.md": ["R8", "R18", "R26", "R28"],
    "08_shoulder_human.md": ["R11", "R24", "R34", "R35", "R36", "R37", "R38"],
    "09_legs.md": ["R1", "R2", "R3", "R8", "R43", "R44", "R16", "R28", "R31"],
    "10_steps.md": ["R29", "R31", "R32", "R33"],
}

NOTE = ("✓ 印は書誌情報（巻・号・ページ等）をウェブで照合済み。それ以外は一般に流通している書誌情報、または作品に詳しい方からの情報に基づくため、"
        "引用する際は原典で確認すること。")


def main():
    out = ["# 参考文献一覧", "",
           "本リポジトリの各章で参照した資料の一覧。各章の末尾にも、その章で参照したものを再掲している。",
           "このファイルと各章の参考文献節は `tools/references.py` で生成している。", "", NOTE, ""]
    for g, ids in GROUPS:
        out += [f"## {g}", ""] + [f"- **[{i}]** {R[i]}" for i in ids] + [""]
    out += ["## 章と参考文献の対応", "", "| 章 | 参考文献 |", "|---|---|"]
    out += [f"| [{f[:2]}]({f}) | {', '.join(ids)} |" for f, ids in USE.items()] + [""]
    (DOCS / "references.md").write_text("\n".join(out))
    for f, ids in USE.items():
        p = DOCS / f
        s = re.sub(r"\n## 参考文献\n[\s\S]*$", "\n", p.read_text())
        s = s.rstrip() + "\n\n## 参考文献\n\n" + "\n".join(f"- [{i}] {R[i]}" for i in ids) + \
            "\n\n（全文献の一覧と照合状況は [references.md](references.md) を参照）\n"
        p.write_text(s)


def to_latex(text):
    """参考文献の文字列を LaTeX に変換する（斜体・URL・特殊文字）"""
    urls = []
    def keep(m):
        urls.append(m.group(0)); return f"@@URL{len(urls) - 1}@@"
    t = re.sub(r"https?://\S+", keep, text)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")):
        t = t.replace(a, b)
    t = re.sub(r"\*([^*]+)\*", r"\\textit{\1}", t)
    t = re.sub(r"\[(R\d+)\]", r"\\cite{\1}", t)
    t = t.replace("✓", r"$\\checkmark$")
    for i, u in enumerate(urls):
        t = t.replace(f"@@URL{i}@@", r"\url{" + u + "}")
    return t


def write_latex(path=ROOT / "report" / "references.tex"):
    """報告書用の thebibliography（分野の順に並べる）"""
    lines = [r"\begin{thebibliography}{R44}", r"\small"]
    for g, ids in GROUPS:
        lines.append(r"\item[] \textbf{" + g + "}")
        lines += [r"\bibitem[" + i + "]{" + i + "} " + to_latex(R[i]) for i in ids]
    lines.append(r"\end{thebibliography}")
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
    write_latex()
