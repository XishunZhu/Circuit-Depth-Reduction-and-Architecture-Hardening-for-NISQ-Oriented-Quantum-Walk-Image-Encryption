# -*- coding: utf-8 -*-
r"""
绘制 QPRW-QIE 优化版的量子电路结构图
====================================
产出（均在 results_opt/）：
  fig1_pipeline.png        加/解密系统级流水线框图
  fig2_qpr_encrypt.png     QPR 加密量子线路结构
  fig3_qpr_decrypt.png     QPR 解密（逆）量子线路结构
  fig4_qwp_step.png        QWP 单步 DTQW 量子线路（QFT 循环增量器）
  fig5_shift_compare.png   移位算符：MCX 涟波(原) vs QFT 增量器(优化)
  fig6_qhrng.png           QHRNG 混合量子随机数发生器线路
"""
import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from qiskit import QuantumCircuit
from qiskit.circuit import Gate

import qprw_qie_opt as Q

BASE = Q.BASE
OUT = os.path.join(BASE, "results_opt")
os.makedirs(OUT, exist_ok=True)

# ---------- 通用：把子线路包成可画成方块的 Gate ----------
def as_gate(sub, name, nq):
    g = Gate(name, nq, [])
    g.definition = sub
    return g

def _upos(npos):
    """构造 npos 个位置的增量子线路（9 -> npos+1 qubit 通用版本）。"""
    pos = list(range(1, npos + 1))
    sub = QuantumCircuit(npos + 1)
    Q.ctrl_incdec_qft(sub, pos, 0, +1)
    return sub

# =====================================================================
# Fig 1: 系统级流水线
# =====================================================================
def draw_pipeline():
    fig, ax = plt.subplots(figsize=(15.5, 6.8))
    ax.set_xlim(0, 120); ax.set_ylim(0, 48); ax.axis("off")
    ax.set_facecolor("#ffffff")

    def box(x, y, w, h, text, fc, tc="white", fs=9.0, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.0",
                                    linewidth=1.4, edgecolor="#37474f", facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc,
                fontsize=fs, fontweight="bold" if bold else "normal",
                fontfamily="DejaVu Sans")

    def arrow(x1, y1, x2, y2, color="#455a64", lw=1.6, style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
                                     linewidth=lw, color=color, linestyle=ls, shrinkA=0, shrinkB=0))

    QM = "#1e88e5"; CL = "#78909c"; IO = "#43a047"; ACK = "#8e24aa"
    ax.text(60, 45.0, "QPRW-QIE  Encryption  /  Decryption  Pipeline   (64 x 64 grayscale)",
            ha="center", fontsize=14, fontweight="bold")

    # ---- 加密主行 ----
    y = 33.0
    box(1, y, 11, 8, "Plain\nimage\n$I$", IO, fs=9, bold=True)
    arrow(12.4, y + 4, 15.4, y + 4)
    box(15.5, y, 17, 8, "QPR  qubit-plane rotor\nquantum, 20 qubits\nCCX x8 + CX x8", QM, fs=8.4, bold=True)
    arrow(32.9, y + 4, 35.9, y + 4)
    box(36, y, 17, 8, "QWP  walk permutation\nquantum, 9 qubits\nring-256, t = 128", QM, fs=8.4, bold=True)
    arrow(53.4, y + 4, 56.4, y + 4)
    box(56.5, y, 17, 8, "BCD  bi-directional\nchain diffusion\n(classical)", CL, fs=8.4)
    arrow(73.9, y + 4, 76.9, y + 4)
    box(77, y, 17, 8, "DNA  dynamic coding\n8 rules, LUT\n(classical)", CL, fs=8.4)
    arrow(94.4, y + 4, 97.4, y + 4)
    box(97.5, y, 11, 8, "Cipher\n$C$", "#c62828", fs=9, bold=True)

    ax.text(24, y + 9.4, "Step1 shift + Step2 plane rotation\n(compile-time relabel -> 0 gates)",
            ha="center", fontsize=7.6, color="#5d4037", style="italic")
    ax.text(44.5, y + 9.4, "time-averaged distribution\n-> perturbed argsort permutation",
            ha="center", fontsize=7.6, color="#5d4037", style="italic")

    # ---- 解密 ----
    y2 = 13.0
    box(97.5, y2, 11, 8, "Cipher\n$C$", "#c62828", fs=9, bold=True)
    arrow(97.4, y2 + 4, 94.4, y2 + 4, color="#8e24aa")
    box(77, y2, 17, 8, "DNA$^{-1}$\nself-inverse", CL, fs=8.4)
    arrow(76.9, y2 + 4, 73.9, y2 + 4, color="#8e24aa")
    box(56.5, y2, 17, 8, "BCD$^{-1}$\nreverse chain", CL, fs=8.4)
    arrow(56.4, y2 + 4, 53.4, y2 + 4, color="#8e24aa")
    box(36, y2, 17, 8, "QWP$^{-1}$\ninverse permutation", ACK, fs=8.4)
    arrow(35.9, y2 + 4, 32.9, y2 + 4, color="#8e24aa")
    box(15.5, y2, 17, 8, "QPR$^{-1}$  (quantum)\nCX x8 + CCX x8", QM, fs=8.4, bold=True)
    arrow(15.4, y2 + 4, 12.4, y2 + 4, color="#8e24aa")
    box(1, y2, 11, 8, "Recovered\n$I$", IO, fs=9, bold=True)

    # 密钥支路
    box(36, 24.0, 40, 6.0, "QHRNG  hybrid quantum RNG\n20 qubits, real measurement -> SHA3 extractor",
        "#00695c", fs=8.4)
    arrow(56, 24.0, 56, y2 + 8.1, color="#00695c", lw=1.2, ls="--")
    arrow(56, 30.1, 56, y + 8.1, color="#00695c", lw=1.2, ls="--")
    arrow(76.1, 27, 81, 27, color="#00695c", lw=1.2, ls="--")
    arrow(75.9, 27, 71, 27, color="#00695c", lw=1.2, ls="--")

    box(1, 24.0, 32, 6.0, "Key-dependent:  rho, dy[b], dx[b], phi0, iv0, iv1\n"
        "plaintext-bound:  H = SHA3-256(I)  (semantic security)",
        "#ef6c00", fs=7.8)
    arrow(17, 24.0, 17, y2 + 8.1, color="#ef6c00", lw=1.2, ls="--")
    arrow(24, 30.1, 24, y + 8.1, color="#ef6c00", lw=1.2, ls="--")

    p = os.path.join(OUT, "fig1_pipeline.png")
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p

# =====================================================================
# Fig 2/3: QPR 加密 / 解密线路
# =====================================================================
NPOS_ILL = 3          # 画 8 像素的示意结构，与 64x64 结构完全相同

def build_qpr_drawable(inverse, npos=NPOS_ILL):
    nq = npos + Q.NPLANE
    qc = QuantumCircuit(nq, name="QPR-Enc" if not inverse else "QPR-Dec")
    pl = [npos + b for b in range(Q.NPLANE)]
    if inverse:
        qc.barrier(); 
        for b in range(Q.NPLANE - 1, -1, -1):
            qc.cx(pl[(b - 1) % Q.NPLANE], pl[b])
        qc.barrier()
        for b in range(Q.NPLANE - 1, -1, -1):
            qc.ccx(pl[(b + 1) % Q.NPLANE], pl[(b + 4) % Q.NPLANE], pl[b])
        qc.barrier()
    else:
        qc.barrier()
        for b in range(Q.NPLANE):
            qc.ccx(pl[(b + 1) % Q.NPLANE], pl[(b + 4) % Q.NPLANE], pl[b])
        qc.barrier()
        for b in range(Q.NPLANE):
            qc.cx(pl[(b - 1) % Q.NPLANE], pl[b])
        qc.barrier()
    return qc

def draw_qpr(inverse):
    qc = build_qpr_drawable(inverse)
    title = ("QPR  decryption  (inverse):  Step4$^{-1}$ -> Step3$^{-1}$ -> Step2$^{-1}$"
             if inverse else
             "QPR  encryption:  Step3 Toffoli injection  ->  Step4 CNOT plane chain")
    fig = qc.draw("mpl", fold=-1, style={"backgroundcolor": "#FFFFFF"},
                  plot_barriers=True, idle_wires=False)
    ax = fig.gca()
    ax.set_title(title + "\n(structure identical for 64x64: npos=12 -> 20 qubits; "
                         "Step2 plane rotation costs 0 gates after compile-time relabel)",
                 fontsize=9.5)
    name = "fig3_qpr_decrypt.png" if inverse else "fig2_qpr_encrypt.png"
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p

# =====================================================================
# Fig 4: QWP 单步 DTQW
# =====================================================================
def draw_qwp_step():
    npos = 8
    nq = npos + 1
    coin, pos = 0, list(range(1, nq))

    sub_inc = QuantumCircuit(nq)
    Q.ctrl_incdec_qft(sub_inc, pos, coin, +1)
    sub_dec = QuantumCircuit(nq)
    Q.ctrl_incdec_qft(sub_dec, pos, coin, -1)
    G_INC = as_gate(sub_inc, "QFT-Inc\n(+1 mod 2^n)", nq)
    G_DEC = as_gate(sub_dec, "QFT-Dec\n(-1 mod 2^n)", nq)

    qc = QuantumCircuit(nq, name="DTQW-step")
    qc.u(2.0 * 0.7, 0.0, math.pi, coin)
    qc.append(G_INC, range(nq))
    qc.append(Gate("X", 1, []), [coin])
    qc.append(G_DEC, range(nq))
    qc.append(Gate("X", 1, []), [coin])
    qc.barrier()

    fig = qc.draw("mpl", fold=-1, style={"backgroundcolor": "#FFFFFF"}, idle_wires=False)
    ax = fig.gca()
    ax.set_title("QWP  one step:  coin U(2*theta,0,pi)  +  shift S\n"
                 "S :  coin=1 -> x+1 (mod 256);  coin=0 -> x-1 (mod 256)\n"
                 "incrementer = QFT . controlled-phase . QFT^-1   (no carry ancilla)",
                 fontsize=9.5)

    # 下方补一个 QFT 内部的标量结构说明
    fig2, a2 = plt.subplots(figsize=(11, 2.6))
    a2.axis("off"); a2.set_xlim(0, 100); a2.set_ylim(0, 10)
    pieces = [("QFT", "#1e88e5"), ("P(2*pi*2^j/N)", "#43a047"), ("QFT^-1", "#8e24aa")]
    x = 6
    for t, c in pieces:
        w = 12 if t.startswith("QFT") else 24
        a2.add_patch(FancyBboxPatch((x, 3), w, 4, boxstyle="round,pad=0.3,rounding_size=0.6",
                                    linewidth=1.3, edgecolor="#37474f", facecolor=c))
        a2.text(x + w / 2, 5, t, ha="center", va="center", color="white", fontsize=10,
                fontweight="bold")
        x += w + 3
        if x < 90:
            a2.annotate("", xy=(x - 2.6, 5), xytext=(x - 4.6, 5),
                        arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#37474f"))
    a2.text(6, 1.2, "cyclic +1 without carry qubit:  QFT diagonalises the circulant shift; "
                    "the enforced phase kick e^{2*pi*i*k/N} shifts the Fourier index k -> k+1",
            fontsize=8.5, color="#37474f")
    p2 = os.path.join(OUT, "fig4b_qwp_shift_detail.png")
    fig2.savefig(p2, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig2)

    p = os.path.join(OUT, "fig4_qwp_step.png")
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p, p2

# =====================================================================
# Fig 5: 移位算符两种实现的线路对比
# =====================================================================
def draw_shift_compare():
    npos = 4
    nq = npos + 1
    pos = list(range(1, nq))
    a = QuantumCircuit(nq)
    Q.ctrl_incdec_mcx(a, pos, 0, +1)
    b = QuantumCircuit(nq)
    Q.ctrl_incdec_qft(b, pos, 0, +1)
    fa = a.draw("mpl", fold=-1, style={"backgroundcolor": "#FFFFFF"}, idle_wires=False)
    fb = b.draw("mpl", fold=-1, style={"backgroundcolor": "#FFFFFF"}, idle_wires=False)
    pa = os.path.join(OUT, "fig5a_shift_mcx.png")
    pb = os.path.join(OUT, "fig5b_shift_qft.png")
    fa.gca().set_title("Baseline: multi-controlled-X ripple carry  (npos=4 illustration)",
                       fontsize=9.5)
    fb.gca().set_title("Optimised: QFT cyclic incrementer  (npos=4 illustration)",
                       fontsize=9.5)
    fa.savefig(pa, dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fa)
    fb.savefig(pb, dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fb)
    return pa, pb

# =====================================================================
# Fig 6: QHRNG
# =====================================================================
def draw_qhrng(n=8):
    ang = [np.linspace(0.3, 1.2, 8) * math.pi for _ in range(3)]
    from qiskit import QuantumCircuit as QC
    qc = QC(n, n, name="QHRNG")
    qc.h(range(n))
    qc.barrier()
    for j in range(n):
        qc.ry(float(ang[0][j]), j)
        qc.rz(float(ang[1][j]), j)
    qc.barrier()
    for j in range(0, n - 1, 2):
        qc.cx(j, j + 1); qc.cx(j + 1, j)
    for j in range(1, n - 1, 2):
        qc.cx(j, j + 1); qc.cx(j + 1, j)
    qc.barrier()
    for j in range(n):
        qc.ry(float(ang[2][j]), j)
    qc.barrier()
    qc.measure(range(n), range(n))
    fig = qc.draw("mpl", fold=-1, style={"backgroundcolor": "#FFFFFF"}, idle_wires=False)
    fig.gca().set_title("QHRNG:  H layer (QTRNG)  +  RY/RZ (parameterised)  +  DCX chain (QPRNG)\n"
                        "measurement outcomes -> SHA3-256 extractor -> keystream   "
                        "(production uses %d qubits)" % Q._QRNG_N, fontsize=9.5)
    p = os.path.join(OUT, "fig6_qhrng.png")
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p

# =====================================================================
def main():
    made = []
    made.append(draw_pipeline())
    made.append(draw_qpr(False))
    made.append(draw_qpr(True))
    a, b = draw_qwp_step(); made += [a, b]
    c, d = draw_shift_compare(); made += [c, d]
    made.append(draw_qhrng())
    print("generated:")
    for p in made:
        print("   %s  (%d bytes)" % (os.path.basename(p), os.path.getsize(p)))

if __name__ == "__main__":
    main()
