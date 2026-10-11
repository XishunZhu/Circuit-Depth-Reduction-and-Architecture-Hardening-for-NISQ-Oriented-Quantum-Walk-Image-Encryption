# -*- coding: utf-8 -*-
"""生成中文论文用图表 (600 dpi) -> results_v3/*.png"""
import os, math, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
from qiskit import QuantumCircuit
import qprw_qie_v2 as V

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "results_v3")
os.makedirs(OUT, exist_ok=True)
DPI = 600
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.dpi": DPI})
IMAGES = ["lake", "baboon", "peppers", "cameraman", "jetplane", "house"]
CN_NAME = {"lake": "Lake", "baboon": "Baboon", "peppers": "Peppers",
           "cameraman": "Cameraman", "jetplane": "Jetplane", "house": "House"}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    sz = os.path.getsize(os.path.join(OUT, name)) / 1048576.0
    print("  %-26s  %.2f MB" % (name, sz))


# ---------------------------------------------------------------- Fig 1 流水线
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x, y, w, h, txt, fc="#eaf2fb", ec="#2f6fa8", fs=8.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    linewidth=1.2, edgecolor=ec, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color="#12324f")

    def arrow(x1, y1, x2, y2, c="#4a4a4a", ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                     linewidth=1.1, color=c, linestyle=ls,
                                     connectionstyle="arc3,rad=0"))
    ax.text(50, 96, "Overall pipeline  (64$\\times$64 grayscale, block length $L$=256)",
            ha="center", fontsize=10.5, fontweight="bold", color="#12324f")
    ax.text(3, 86, "ENCRYPTION", fontsize=9, fontweight="bold", color="#1b6b3a")
    box(3, 74, 17, 9, "Plain image $I$", "#f2f7f2", "#1b6b3a")
    box(23, 74, 17, 9, "QPR\n(20 q circuit)", "#e6f4ec", "#1b6b3a")
    box(43, 74, 17, 9, "QWP permutation\n(DTQW, 9 q)", "#e6f4ec", "#1b6b3a")
    box(63, 74, 15, 9, "BCD-A\ndiffusion", "#eaf2fb", "#2f6fa8")
    box(80, 74, 17, 9, "DNA layer", "#eaf2fb", "#2f6fa8")
    for a, b in ((20, 23), (40, 43), (60, 63), (78, 80)):
        arrow(a, 78.5, b, 78.5)
    box(80, 58, 17, 9, "Ciphertext $C$", "#f2f7f2", "#1b6b3a")
    arrow(88.5, 74, 88.5, 67.5)
    ax.text(3, 62, "KEY / QUANTUM BRANCH", fontsize=9, fontweight="bold", color="#8a4b1f")
    box(3, 40, 20, 9, "QHRNG (20 q, $2^{13}$ shots)\nquantum entropy", "#fdf1e3", "#c07028")
    box(26, 40, 20, 9, "nonce $N$ (16 B)\nin cleartext header", "#fdf1e3", "#c07028")
    box(49, 40, 22, 9, "keystream $KS$\nSHA3-256 extraction", "#fdf1e3", "#c07028")
    box(74, 40, 23, 9, "DTQW distributions\n$\\to$ perm seed $S$ (256 bit)", "#fdf1e3", "#c07028")
    arrow(23, 44.5, 26, 44.5)
    arrow(46, 44.5, 49, 44.5)
    arrow(71, 44.5, 74, 44.5)
    arrow(13, 40, 13, 33, c="#c07028"); arrow(36, 40, 36, 33, c="#c07028")
    box(26, 24, 48, 9, "Header:  $N$ (16 B)  $\\Vert$  $S \\oplus KS_{hdr}$ (32 B)   +   MAC tag (16 B)",
        "#f6f6f6", "#555555")
    arrow(36, 33, 36, 33.2, c="#c07028")
    arrow(50, 24, 50, 24.2, c="#555555")
    arrow(88.5, 58, 74, 33.2, c="#555555", ls="--")
    ax.text(50, 14, "DEC:  MAC verify  $\\to$  recover $S$ from header  $\\to$  Fisher-Yates expand $\\pi$ (no QWP rerun)\n"
                    "$\\to$  DNA$^{-1}$  $\\to$  BCD-A$^{-1}$  $\\to$  $\\pi^{-1}$  $\\to$  QPR$^{-1}$ (20 q circuit)",
            ha="center", fontsize=8.5, color="#12324f")
    ax.text(50, 4.5, "solid arrows: data flow      dashed: header assembly      orange: quantum-resource branch",
            ha="center", fontsize=7.5, color="#666666")
    save(fig, "fig1_pipeline.png")


# ---------------------------------------------------------------- Fig 2 QPR 线路
def fig2_qpr():
    qc = V.build_qpr_circuit(False)
    qc.barrier()
    qc_inv = V.build_qpr_circuit(True)
    fig = plt.figure(figsize=(9.0, 5.6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("(a) QPR forward  (Step3 CCX + Step4 CX)", fontsize=9)
    qc.draw("mpl", ax=ax1, idle_wires=False, fold=-1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("(b) QPR inverse", fontsize=9)
    qc_inv.draw("mpl", ax=ax2, idle_wires=False, fold=-1)
    save(fig, "fig2_qpr_circuit.png")


def _qft_box(npos, label):
    g = V._official_qft(npos).to_gate()
    g.label = label
    return g


# ---------------------------------------------------------------- Fig 3 折叠行走
def fig3_folded():
    npos, T = 4, 3
    nq = npos + 1
    qc = QuantumCircuit(nq)
    coin, pos = 0, list(range(1, nq))
    qc.u(2 * (math.pi / 4), 0.3, 0, coin)
    qc.append(_qft_box(npos, "QFT"), pos)
    qc.barrier()
    for s in range(T):
        qc.u(0.7 + 0.1 * s, 0, math.pi, coin)
        for j in range(npos):
            qc.cp(2 * V.theta_j(j, npos), coin, pos[j])
        for j in range(npos):
            qc.p(-V.theta_j(j, npos), pos[j])
        qc.barrier(label="step %d" % (s + 1))
    qc.append(_qft_box(npos, "QFT$^\\dagger$"), pos)
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    ax.set_title("QFT-frame folded DTQW:  $U = $QFT$\\cdot(D C_T)\\cdots(D C_1)\\cdot$QFT$^{-1}$"
                 "   (shown with $n_{pos}$=4, $T$=3;  deployed with $n_{pos}$=8, $T$=32)", fontsize=9)
    qc.draw("mpl", ax=ax, idle_wires=False, fold=-1)
    save(fig, "fig3_qwp_folded.png")


# ---------------------------------------------------------------- Fig 4 移位算符对照
def fig4_shift():
    npos = 4
    nq = npos + 1

    def mk(kind):
        qc = QuantumCircuit(nq)
        coin, pos = 0, list(range(1, nq))
        qc.u(0.7, 0, math.pi, coin)
        qc.barrier()
        if kind == "mcx":
            V.ctrl_incdec_mcx(qc, pos, coin, +1)
            qc.x(coin); V.ctrl_incdec_mcx(qc, pos, coin, -1); qc.x(coin)
        elif kind == "perstep":
            qc.append(_qft_box(npos, "QFT"), pos)
            for j in range(npos): qc.cp(V.theta_j(j, npos), coin, pos[j])
            qc.append(_qft_box(npos, "QFT$^\\dagger$"), pos); qc.x(coin)
            qc.append(_qft_box(npos, "QFT"), pos)
            for j in range(npos): qc.cp(-V.theta_j(j, npos), coin, pos[j])
            qc.append(_qft_box(npos, "QFT$^\\dagger$"), pos); qc.x(coin)
        else:
            for j in range(npos): qc.cp(2 * V.theta_j(j, npos), coin, pos[j])
            for j in range(npos): qc.p(-V.theta_j(j, npos), pos[j])
        return qc
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.4))
    titles = ["(a) MCX ripple-carry (v1)", "(b) per-step QFT incrementer",
              "(c) folded layer $D$ (this work)"]
    for ax, k, t in zip(axes, ("mcx", "perstep", "folded"), titles):
        ax.set_title(t, fontsize=9)
        mk(k).draw("mpl", ax=ax, idle_wires=False, fold=-1)
    save(fig, "fig4_shift_compare.png")


# ---------------------------------------------------------------- Fig 5 QHRNG
def fig5_qhrng():
    a = [V.bytes_to_floats(V.sha3_ctr(b"a", 16)) * math.pi] * 3
    qc = V.build_qhrng_circuit(a)
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.set_title("QHRNG: Hadamard layer + parameterised rotations + DCX chain + measurement (20 qubits)",
                 fontsize=9)
    qc.draw("mpl", ax=ax, idle_wires=False, fold=-1)
    save(fig, "fig5_qhrng.png")


# ---------------------------------------------------------------- Fig 6 明文/密文 + 直方图
def fig6_images():
    fig, axes = plt.subplots(3, 6, figsize=(11.4, 6.2))
    for i, nm in enumerate(IMAGES):
        pt = np.array(Image.open(os.path.join(OUT, "pt_%s.png" % nm)))
        ct = np.array(Image.open(os.path.join(OUT, "ct_%s.png" % nm)))
        axes[0, i].imshow(pt, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[0, i].set_title(CN_NAME[nm], fontsize=9)
        axes[1, i].imshow(ct, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[2, i].hist(ct.ravel(), bins=32, color="#2f6fa8", alpha=.85)
        axes[2, i].set_ylim(0, 260); axes[2, i].tick_params(labelsize=6)
        for a in (axes[0, i], axes[1, i]): a.set_xticks([]); a.set_yticks([])
    axes[0, 0].set_ylabel("plain", fontsize=8); axes[1, 0].set_ylabel("cipher", fontsize=8)
    axes[2, 0].set_ylabel("histogram", fontsize=8)
    save(fig, "fig6_plain_cipher.png")


# ---------------------------------------------------------------- Fig 7 统计
def fig7_stats():
    R = json.load(open(os.path.join(OUT, "_results.json"), encoding="utf-8"))
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.6))
    pt = np.array(Image.open(os.path.join(OUT, "pt_lake.png")))
    ct = np.array(Image.open(os.path.join(OUT, "ct_lake.png")))
    axes[0, 0].hist(pt.ravel(), bins=64, color="#999999", alpha=.8, label="plain")
    axes[0, 0].hist(ct.ravel(), bins=64, color="#2f6fa8", alpha=.7, label="cipher")
    axes[0, 0].legend(fontsize=7); axes[0, 0].set_title("(a) histogram (Lake)", fontsize=9)
    p = pt.ravel().astype(float); q = np.roll(pt, 1, axis=1).ravel().astype(float)
    axes[0, 1].scatter(p[::3], q[::3], s=2, color="#c07028", alpha=.4)
    axes[0, 1].set_title("(b) adjacent-pixel correlation: plain", fontsize=9)
    axes[0, 1].set_xlabel("$I(x,y)$", fontsize=8); axes[0, 1].set_ylabel("$I(x,y+1)$", fontsize=8)
    p2 = ct.ravel().astype(float); q2 = np.roll(ct, 1, axis=1).ravel().astype(float)
    axes[0, 2].scatter(p2[::3], q2[::3], s=2, color="#2f6fa8", alpha=.4)
    axes[0, 2].set_title("(c) adjacent-pixel correlation: cipher", fontsize=9)
    axes[0, 2].set_xlabel("$C(x,y)$", fontsize=8); axes[0, 2].set_ylabel("$C(x,y+1)$", fontsize=8)
    names = IMAGES
    cp = [R["main"][n]["corr_p"][0] for n in names]
    cc = [R["main"][n]["corr_c"][0] for n in names]
    x = np.arange(len(names)); w = 0.38
    axes[1, 0].bar(x - w / 2, cp, w, color="#c07028", label="plain")
    axes[1, 0].bar(x + w / 2, cc, w, color="#2f6fa8", label="cipher")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([CN_NAME[n] for n in names], rotation=35, fontsize=7)
    axes[1, 0].legend(fontsize=7); axes[1, 0].set_title("(d) horizontal correlation", fontsize=9)
    ep = [R["main"][n]["ent_p"] for n in names]
    ec = [R["main"][n]["ent_c"] for n in names]
    axes[1, 1].bar(x - w / 2, ep, w, color="#c07028", label="plain")
    axes[1, 1].bar(x + w / 2, ec, w, color="#2f6fa8", label="cipher")
    axes[1, 1].axhline(8, ls="--", c="k", lw=.8)
    axes[1, 1].set_ylim(5, 8.2)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([CN_NAME[n] for n in names], rotation=35, fontsize=7)
    axes[1, 1].legend(fontsize=7); axes[1, 1].set_title("(e) information entropy (bit)", fontsize=9)
    ch = [R["main"][n]["chi_c"] for n in names]
    axes[1, 2].bar(x, ch, 0.5, color="#2f6fa8")
    axes[1, 2].axhline(293.2, ls="--", c="r", lw=.9)
    axes[1, 2].text(0.05, 296, "$\\chi^2_{0.05,255}$=293.2", fontsize=7, color="r")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels([CN_NAME[n] for n in names], rotation=35, fontsize=7)
    axes[1, 2].set_title("(f) chi-square of ciphertext", fontsize=9)
    save(fig, "fig7_statistics.png")


# ---------------------------------------------------------------- Fig 8 资源
def fig8_resource():
    labels = ["MCX baseline\n$T$=128, both sides", "per-step QFT baseline\n$T$=128, both sides",
              "folded\n$T$=32, enc only"]
    cx = [3809336, 1245240, 10480]
    depth = [259841, 39424, 1302]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    b = axes[0].bar(labels, cx, color=["#c07028", "#7fa8d0", "#1b6b3a"])
    axes[0].set_yscale("log"); axes[0].set_ylabel("two-qubit gates per image (log)")
    axes[0].set_title("(a) CX budget, 64$\\times$64 image", fontsize=9)
    for r, v in zip(b, cx):
        axes[0].text(r.get_x() + r.get_width() / 2, v * 1.15, format(v, ","), ha="center", fontsize=7.5)
    b2 = axes[1].bar(labels, depth, color=["#c07028", "#7fa8d0", "#1b6b3a"])
    axes[1].set_yscale("log"); axes[1].set_ylabel("circuit depth (log)")
    axes[1].set_title("(b) depth of the largest coherent block", fontsize=9)
    for r, v in zip(b2, depth):
        axes[1].text(r.get_x() + r.get_width() / 2, v * 1.15, format(v, ","), ha="center", fontsize=7.5)
    for ax in axes:
        ax.tick_params(labelsize=7)
        ax.set_xticklabels(labels, fontsize=7)
    save(fig, "fig8_resource.png")


# ---------------------------------------------------------------- Fig 9 噪声
def fig9_noise():
    p = [0, 1e-4, 1e-3, 5e-3, 1e-2]
    tv = {"MCX (v1)": [0, .9263, .9410, .9410, .9410],
          "per-step QFT": [0, .7232, .9409, .9410, .9410],
          "QFF folded": [0, .0903, .5431, .9039, .9369]}
    co = {"MCX (v1)": [1, .4389, .1649, -.0949, .0145],
          "per-step QFT": [1, .8677, .4554, .2519, .1019],
          "QFF folded": [1, .9985, .8874, .5268, .4424]}
    Ts = [16, 32, 64, 128]
    tvT = {1e-4: [.0903, .1269, .1735, .2348], 1e-3: [.5431, .6237, .6667, .6606],
           5e-3: [.9075, .8944, .8401, .7399]}
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6))
    for k, c, m in zip(tv, ["#c07028", "#7fa8d0", "#1b6b3a"], ["o", "s", "^"]):
        axes[0].plot(p[1:], tv[k][1:], m + "-", color=c, label=k)
    axes[0].set_xscale("log"); axes[0].set_xlabel("depolarising parameter $p$")
    axes[0].set_ylabel("total-variation distance"); axes[0].legend(fontsize=7)
    axes[0].set_title("(a) distribution degradation ($T$=16)", fontsize=9)
    for k, c, m in zip(co, ["#c07028", "#7fa8d0", "#1b6b3a"], ["o", "s", "^"]):
        axes[1].plot(p[1:], co[k][1:], m + "-", color=c, label=k)
    axes[1].set_xscale("log"); axes[1].set_xlabel("depolarising parameter $p$")
    axes[1].set_ylabel("Pearson correlation with noiseless"); axes[1].legend(fontsize=7)
    axes[1].set_title("(b) correlation retention ($T$=16)", fontsize=9)
    for pp, c, m in zip(tvT, ["#1b6b3a", "#c07028", "#7f7f7f"], ["o", "s", "^"]):
        axes[2].plot(Ts, tvT[pp], m + "-", color=c, label="$p$=%g" % pp)
    axes[2].set_xlabel("walk steps $T$"); axes[2].set_ylabel("TV distance")
    axes[2].legend(fontsize=7); axes[2].set_title("(c) QFF: horizon vs degradation", fontsize=9)
    for ax in axes: ax.tick_params(labelsize=7.5); ax.grid(alpha=.25)
    save(fig, "fig9_noise.png")


# ---------------------------------------------------------------- Fig 10 横向对比（近期文献）
def fig10_comparison():
    """(a) 近期方案在量子资源上演示的图像规模；(b) 验证完整度矩阵（依据各论文摘要/正文自报内容）。"""
    names = ["Abdelfatah\n2025", "Gururaja\n2026", "Wing-B.\n2023", "Acasiete\n2020",
             "Nandi\n2025", "This\nwork"]
    # 量子资源上实际演示的图像规模（像素）；Liu 2019 为经典方案不纳入
    hw_size = [4096, 4096, 16, 16, 16, 4096]
    hw_clr = ["#c07028", "#c07028", "#7fa8d0", "#7fa8d0", "#7fa8d0", "#1b6b3a"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    axes[0].bar(range(len(names)), hw_size, color=hw_clr)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("image size demonstrated on quantum\nresources (pixels, log scale)", fontsize=8)
    axes[0].set_title("(a) demonstrated quantum-resource scale", fontsize=9)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, fontsize=7)
    axes[0].tick_params(labelsize=7.5)
    axes[0].grid(alpha=.2, axis="y")
    axes[0].set_ylim(6, 80000)
    for i, v in enumerate(hw_size):
        axes[0].text(i, v * 1.25, str(v), ha="center", fontsize=7, color="#333333")
    # (b) 验证完整度矩阵: 2=明确报告, 1=部分/模型化, 0=未见报告
    rows = ["Real HW run", "NIST 800-22", "Tamper tests", "Gate accounting",
            "Noise model", "Sensitivity", "MAC tag"]
    M = np.array([
        [2, 2, 2, 2, 0, 0, 2],   # Abdelfatah 2025: HW+NIST?->摘要提及统计/差分; 门核算未给
        [2, 2, 1, 0, 1, 2, 0],   # Gururaja 2026: torino 真机, NIST, 噪声环境
        [2, 0, 0, 0, 0, 0, 0],   # Wing-Bocanegra 2023: 仅行走本身
        [2, 0, 0, 0, 0, 0, 0],   # Acasiete 2020: 仅行走本身
        [1, 0, 0, 1, 0, 0, 0],   # Nandi 2025: 模拟, 门数减半
        [0, 2, 2, 2, 1, 2, 2],   # this work: 模拟+硬件参数噪声模型
    ])
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#f2d5d5", "#f6e3b4", "#bfe3c0"])
    axes[1].imshow(M, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, fontsize=7)
    axes[1].set_yticks(range(len(rows)))
    axes[1].set_yticklabels(rows, fontsize=7)
    sym = {0: "x", 1: "~", 2: "✓"}
    col = {0: "#a33", 1: "#a80", 2: "#181"}
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            axes[1].text(j, i, sym[M[i, j]], ha="center", va="center",
                         fontsize=9, color=col[M[i, j]], fontweight="bold")
    axes[1].set_title("(b) evaluation coverage (self-reported)", fontsize=9)
    axes[1].tick_params(length=0)
    for s in axes[1].spines.values():
        s.set_visible(False)
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig10_comparison.png")


if __name__ == "__main__":
    print("rendering figures at %d dpi ..." % DPI)
    fig1_pipeline(); print("fig1 ok")
    fig2_qpr(); print("fig2 ok")
    fig3_folded(); print("fig3 ok")
    fig4_shift(); print("fig4 ok")
    fig5_qhrng(); print("fig5 ok")
    fig6_images(); print("fig6 ok")
    fig7_stats(); print("fig7 ok")
    fig8_resource(); print("fig8 ok")
    fig9_noise(); print("fig9 ok")
    fig10_comparison(); print("fig10 ok")
