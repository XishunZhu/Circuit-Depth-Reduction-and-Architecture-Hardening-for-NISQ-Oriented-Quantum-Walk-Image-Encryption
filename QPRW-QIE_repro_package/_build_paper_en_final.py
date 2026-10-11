# -*- coding: utf-8 -*-
r"""Translate the anonymized Chinese paper into an English manuscript (Times New Roman, 600 dpi figures)."""
import os
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

BASE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(BASE, "results_v3")
PAPERS = os.path.join(BASE, "paper")
os.makedirs(PAPERS, exist_ok=True)

doc = Document()
st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(10.5)
st.paragraph_format.space_after = Pt(4)
st.paragraph_format.line_spacing = 1.25


def _font(run, cn="Times New Roman", en="Times New Roman", size=10.5, bold=False, italic=False, color=None):
    run.font.name = en
    run._element.rPr.rFonts.set(qn('w:eastAsia'), cn)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)


def P(text, size=10.5, bold=False, italic=False, align="j", cn="Times New Roman", space_after=5,
      indent=True, color=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    _font(r, cn=cn, size=size, bold=bold, italic=italic, color=color)
    p.alignment = {"c": WD_ALIGN_PARAGRAPH.CENTER, "j": WD_ALIGN_PARAGRAPH.JUSTIFY,
                   "l": WD_ALIGN_PARAGRAPH.LEFT}[align]
    if indent and align == "j":
        p.paragraph_format.first_line_indent = Pt(size * 2)
    p.paragraph_format.space_after = Pt(space_after)
    return p


def H(text, level=1):
    h = doc.add_heading("", level=level)
    r = h.add_run(text)
    _font(r, cn="Times New Roman", en="Times New Roman",
          size={1: 14, 2: 12, 3: 11}[level], bold=True, color=(0x12, 0x32, 0x4F))
    h.paragraph_format.space_before = Pt(10 if level == 1 else 6)
    h.paragraph_format.space_after = Pt(5)
    return h


def EQ(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    _font(r, cn="Times New Roman", en="Cambria Math", size=10.5, italic=True)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(6)
    return p


def TBL(caption, header, rows, widths=None, fontsize=8.5):
    c = doc.add_paragraph()
    rr = c.add_run(caption)
    _font(rr, cn="Times New Roman", size=9, bold=True)
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.space_after = Pt(2)
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(h))
        _font(run, cn="Times New Roman", size=fontsize, bold=True)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(v))
            _font(run, cn="Times New Roman", size=fontsize)
            if i > 0:
                cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def FIG(name, caption, width=6.0):
    doc.add_picture(os.path.join(OUTD, name), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    r = p.add_run(caption)
    _font(r, cn="Times New Roman", size=9)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)


def STEP(label, text, size=10.5):
    """Body paragraph with a bold step label (hanging indent), for encryption/decryption steps."""
    p = doc.add_paragraph()
    r1 = p.add_run(label)
    _font(r1, cn="Times New Roman", size=size, bold=True)
    r2 = p.add_run(text)
    _font(r2, cn="Times New Roman", size=size)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Pt(21)
    p.paragraph_format.first_line_indent = Pt(-21)
    p.paragraph_format.space_after = Pt(3)
    return p


# ======================================================================
# Title and Abstract
# ======================================================================
P("Circuit Depth Reduction and Architecture Hardening for NISQ-Oriented Quantum-Walk Image Encryption",
  size=16, bold=True, align="c", cn="Times New Roman", indent=False, space_after=10)
P("Abstract", size=12, bold=True, align="c", cn="Times New Roman", indent=False, space_after=4)
P("To address the problem that the circuit cost of quantum-walk image encryption schemes exceeds the coherence "
  "budget of noisy intermediate-scale quantum (NISQ) devices, two structural overheads are identified - the "
  "multi-controlled ripple-carry adder and the duplicated execution of the walk at both ends. Accordingly, four "
  "depth-reduction mechanisms are proposed: the QFT-controlled incrementer, Fourier-frame folding, +/-1 direction "
  "fusion and compile-time bit-plane relabeling, and a permutation-seed encapsulation mechanism confines the walk "
  "to a single execution on the encryption side; in parallel, the differential-cancellation flaw of the baseline "
  "XOR diffusion chain is repaired by a modular-addition and S-box bidirectional chain. The two-qubit gate count "
  "for a single 64x64 image is reduced from 3,809,336 to 10,480, and the tolerable upper bound on the two-qubit "
  "error rate is relaxed from 8.4e-6 to 1.5e-3. Byte-wise lossless decryption is achieved on six standard test "
  "images, with ciphertext entropy reaching 7.9637, NPCR/UACI of 99.90%/33.59% under single-bit plaintext "
  "perturbation, and 2.03 Mbit of ciphertext passing the NIST SP 800-22 test suite at 98.8%.",
  size=10, indent=False)
P("Keywords: quantum image encryption; discrete-time quantum walk; NISQ; circuit depth reduction; differential diffusion",
  size=10, italic=True, indent=False, space_after=12)

# ======================================================================
H("1 Introduction", 1)
P("Research on quantum image encryption has advanced along five threads: quantum image representation, "
  "permutation transforms, diffusion and coding, key randomness sources, and the circuit implementation of "
  "quantum walks themselves. The following reviews representative works along each thread, clarifying what "
  "problem each solves and what remains unsolved, so as to delimit the starting point of this work.")
P("Le et al. proposed FRQI, which encodes grayscale information in amplitudes and represents 2^n pixels with "
  "n+1 qubits [3], solving the basic question of how a whole image is loaded into a single quantum state. Its "
  "remaining issue is that the grayscale is encoded in probability amplitudes and can only be estimated from "
  "many repeated measurements, so a single measurement cannot deterministically recover a pixel value. To remove "
  "this uncertainty, Zhang et al. proposed NEQR [2], which writes the grayscale as the basis-state bit string of "
  "the color register, so that a pixel value is read out deterministically with a single measurement, and grayscale "
  "transforms can be performed directly by reversible circuits acting on the register; NEQR has therefore become the "
  "representation basis for most subsequent encryption schemes. The cost of NEQR is that the preparation circuit "
  "grows rapidly with the number of address qubits. To compress storage and preparation overhead, Chen et al. "
  "proposed the HSI-color-space-based QIRHSI representation [30], requiring only 10 qubits on the hue and "
  "saturation channels, with a reported preparation optimization ratio of 55.87%-95.72% on an 8x8 sample image; "
  "however, this optimization was verified only on extremely small samples and did not examine the circuit scale "
  "and coherence depth of a complete encryption pipeline.")
P("The permutation stage breaks spatial correlation among pixels. Jiang et al. gave quantum-circuit implementations "
  "of Arnold and Fibonacci permutations [4], decomposing the classical transforms into CNOT and Toffoli networks "
  "using ordinary adders and modular adders, with complexity growing linearly in the operand bit width. This work "
  "solved the problem of how classical permutation is realized on reversible circuits, but leaves the permutation "
  "family itself public, with the key being only the iteration count, and never discusses feasibility under noisy "
  "execution. To make the permutation strongly correlated with the master key, Abd-El-Atty et al. constructed the "
  "probability distribution of a one-particle quantum walk on a cycle as a quantum key image and applied CNOT to an "
  "NEQR image for encryption [5], introducing for the first time the high sensitivity of quantum walks to coin "
  "parameters into key generation. Pawela then pointed out two tensions in this route [16]: first, gates acting "
  "transversely on two independently addressed quantum images do not automatically equal per-pixel XOR; second, "
  "once the quantum key register, already entangled with the data during evolution, is measured or discarded, the "
  "information needed for decryption is lost. The same work also gives a resource estimate: any NEQR preparation "
  "cost dominates the walk layer of complexity O(q^2+qr), implying that a 'low-depth walk' does not equal a "
  "low-depth pipeline. Along this route, Li and Tang realized color-to-grayscale reversible conversion and "
  "encryption with true random numbers and a two-dimensional quantum walk [17]; Alblehai et al. constructed a "
  "medical-image cipher with cascaded quantum walks combined with a Chebyshev map [23]; Cui et al. introduced a "
  "two-dimensional quantum walk into optical color-image encryption [27]; and Abd El-Latif et al. initialized "
  "parameters with a four-dimensional hyperchaotic system, dynamically updated walk parameters by mid-ciphertext "
  "feedback, and superimposed dynamic DNA operations [6]. These works jointly improved statistical metrics, but "
  "left two unresolved problems: cascading and feedback further inflate the circuit scale; and the heavy mixing of "
  "classical components makes it hard to separately quantify the actual security contribution of the quantum "
  "components.")
P("The confusion-diffusion two-stage structure established by Fridrich provides a unified skeleton for the above "
  "schemes [18]. At the bit level, Liu et al. proposed intra- and inter-bit two-level permutation [14]: intra-bit "
  "permutation is realized by chaotic-sequence sorting, and inter-bit permutation by qubit XOR between two selected "
  "bit planes, solving the problem that pixel-position permutation alone does not change the grayscale histogram. "
  "Its limitation is that sorting in a real reversible circuit must be borne by a large comparator network, whereas "
  "the paper gives only an abstract circuit description without gate-level accounting. At the byte level, Chai et al. "
  "used dynamic DNA coding coupled with chaos [15], so that the base-operation rule is selected pixel by pixel, "
  "solving the problem that fixed rules can be broken by statistical enumeration attacks; but this scheme is "
  "classical and lacks quantum randomness. Notably, Kumari and Gupta's horizontal comparison of chaotic and "
  "quantum-chaotic schemes [22] shows that the two classes differ little on entropy, correlation, NPCR and other "
  "metrics, indicating that classical statistical metrics alone cannot tell whether the 'quantum' component truly "
  "plays a role - a gap at the level of evaluation methodology. Wu et al. established a statistical model of ideal "
  "ciphertext for NPCR and UACI with critical values [26], turning these two metrics from the empirical 'higher is "
  "better' into a decidable hypothesis test.")
P("For key entropy sources, Gururaja and Pravinkumar built a hybrid source from a quantum true random-number "
  "generator and a pseudo-random generator to produce the key stream and performed measurement forensics on a real "
  "quantum backend [7]; Sridharan et al. further used QTRNG together with a 6D hyperchaotic key and a multi-layer "
  "DNA structure [8]. These two works solved the true-randomness problem of keys, but leave the key used only for "
  "stream generation while the encryption structure itself remains classical, so the introduction of quantum "
  "resources does not change the security argument of the cryptosystem. NIST SP 800-90B specifies the entropy-"
  "evaluation norm for random bit sources [21], providing a test basis for the above claims but is rarely cited. "
  "Shrestha et al. generated a shared key with the BB84 protocol, enhanced it with a Henon map and derived a "
  "permutation sequence from a Logistic map, comparing medical-image reconstruction quality under four noise "
  "channels - amplitude damping, depolarization, bit flip and phase damping [25], filling the previous gap of "
  "ignoring quantum-noise effects; their remaining issue is that the global permutation circuit's gate count was "
  "not accounted for, and the permutation still has to be rebuilt at both ends in the same way. Abdelfatah et al. "
  "performed real-device execution on an IBM quantum processor, implementing a one-time pad with a CNOT chain and "
  "cipher chaining feedback [9], but left no layer-wise gate-count and depth accounting, nor an evaluation of "
  "circuit degradation at larger scales.")
P("The conditional-shift operator is diagonalized by the quantum Fourier transform, avoiding multi-controlled "
  "ripple-carry cascades, tracing back to the work of Douglas and Wang [29]. Acasiete et al. implemented walks on "
  "cycles, lattices and complete graphs on IBM devices [10], at an experimental scale of 16 nodes and depth of "
  "about one hundred layers, with measurements showing that the trustworthily executable number of steps is limited "
  "by the number of multi-controlled gates. Wing-Bocanegra and Venegas-Andraca implemented discrete-time quantum "
  "walks on Falcon devices via the shunt-decomposition method, systematically comparing implementation overhead and "
  "pointing out that the number of multi-controlled gates directly determines experimental accuracy [11]. Nandi et "
  "al. introduced mid-circuit qubit (qudit) techniques to decompose multi-qubit gates, halving the gate count and "
  "depth of the increment-decrement implementation without ancilla qubits [12]; Saini et al. used Espresso-algorithm "
  "synthesis with one ancilla qubit to reduce the quantum cost of the NEQRX scheme by about 50% [13]. The common "
  "contribution of these works is to push the cost of a single walk step down to a device-acceptable level, while "
  "the common remaining problem has two parts: first, the optimization targets the single-step structure and "
  "single-step cascades, yet in a T-step evolution each step still pays the cost of two Fourier transforms, so the "
  "cost still accumulates linearly with the number of steps; second, the optimization stops at the walk primitive "
  "and is never placed in a complete protocol - when the same walk circuit must be executed once at each of the "
  "encryption and decryption ends, the physical meaning of the above performance changes fundamentally.")
P("Kaur and Kumar systematically organized the spectrum of classical image-encryption techniques [1]; Zhang et al. "
  "reviewed the development of chaotic image security and noted the lack of unified evaluation standards [28]; Kumar "
  "et al. gave a more recent survey on quantum image encryption [24], classifying by quantum image representation, "
  "key-generation mechanism, permutation strategy, quantum transform and classical-quantum hybrid route, and "
  "tabulating security and performance metrics. At present, however, the vast majority of quantum image-encryption "
  "schemes do not disclose the two-qubit gate count and coherence depth of the encryption circuit, while the NEQR "
  "preparation depth grows superlinearly with resolution and is precisely the first component to fail on NISQ "
  "devices. Moreover, only a tiny fraction of works have been verified on real quantum hardware or noisy models; "
  "most conclusions come from noiseless state-vector simulation, and security checks are limited to classical "
  "statistical metrics [24,28]. Many architectural assumptions remain unexamined: practices such as rebuilding "
  "permutation parameters at both ends in the same way and key derivation depending on a plaintext digest are "
  "widely inherited, and structural flaws are exposed once real noise is introduced or the same plaintext is "
  "encrypted repeatedly. The NISQ-era constraint defined by Preskill [19] implies that what determines the "
  "feasibility of a scheme is not the asymptotic complexity of the representation method, but the depth and "
  "two-qubit gate count of the largest coherent block that must be executed error-free. Around these problems, "
  "this work locates bottlenecks by resource audit, compresses the coherence budget by circuit-level depth "
  "reduction, eliminates hidden assumptions by architecture-level hardening, and closes the whole paper with a "
  "verifiable complete evidence chain.")
P("The main contributions of this work are as follows:")
P("(1) Resource audit and bottleneck localization. A layer-wise accounting of the baseline scheme on the hardware "
  "basis {rz, sx, x, cx} locates two structural overheads - the multi-controlled ripple-carry position-shift "
  "operator and the dual-end repeated execution of the walk - and quantifies a per-image budget of 3,809,336 "
  "two-qubit gates and 259,841 layers of coherence depth.")
P("(2) Circuit-level depth reduction. The QFT frame folding (QFF) and +/-1 direction-fusion mechanisms, together "
  "with the QFT-controlled incrementer and compile-time bit-plane relabeling, reduce the walk to 648 two-qubit gates "
  "per block and a total per-image budget of 10,480 two-qubit gates (363.5x compression), with the maximum "
  "coherent-block depth reduced to 1302 layers (about 200x).")
P("(3) Architecture-level hardening. A Permutation-Seed Encapsulation (PSE) mechanism removes the walk entirely "
  "from the decryption path; a randomized quantum nonce and a message authentication code decouple key derivation "
  "from the plaintext digest and provide integrity protection; and the differential-cancellation flaw of the XOR "
  "bidirectional diffusion chain is identified and fixed.")

# ======================================================================
H("2 Preliminaries", 1)
H("2.1 NEQR Quantum Image Representation", 2)
P("NEQR encodes a 2^n-pixel, 8-bit grayscale image as the quantum state [2]:")
EQ("|I> = (1/2^n) Sum_{i=0}^{2^n-1} |C_i> (x) |i> ,")
P("where |C_i> = |c^7_i>...|c^0_i> is the 8-qubit color register of the i-th pixel and |i> is the position "
  "register. Unlike the amplitude encoding of FRQI, NEQR uses basis-state exact encoding of color information, so "
  "pixel values are recovered deterministically after measurement, and color transforms can be implemented directly "
  "by reversible circuits acting on the color register, making it more suitable for cryptographic operations [2,25]. "
  "The experiment object of this work is a 64x64 grayscale image with a 12-qubit position register and an 8-qubit "
  "color register.")
H("2.2 Discrete-Time Quantum Walk on a Cycle", 2)
P("Consider a cycle of 2^m positions (here m=8, i.e., blocks of L=256 pixels). The walker is described by one coin "
  "qubit and m position qubits. Each step first applies a coin operator C and then a conditional-shift operator S, "
  "which cyclically adds (subtracts) 1 to (from) the position register under coin state |0> (|1>):")
EQ("S = |0><0| (x) Sum_k |k+1><k| + |1><1| (x) Sum_k |k-1><k| .")
P("The T-step evolution operator is U^T = (S C_T)...(S C_1), where the coin angle and initial phase of each step "
  "are determined by the key stream, giving the walk distribution key dependence. The measured position distribution, "
  "after hashing, serves as the permutation seed; the high sensitivity of the walk distribution to key and plaintext "
  "constitutes the security source of the permutation layer [5,10].")
H("2.3 Confusion-Diffusion Paradigm and Dynamic DNA Coding", 2)
P("The confusion-diffusion two-stage structure is the universal skeleton of modern image ciphers [18]: the "
  "confusion layer destroys the association between pixel positions and ciphertext, and the diffusion layer propagates "
  "local plaintext perturbation globally. DNA coding maps bytes to base pairs and performs XOR under a rule table, "
  "and is a commonly used diffusion means in image encryption [15]. This work superimposes quantum-walk permutation "
  "and a quantum random source on the classical skeleton, and gives a differential-analysis improvement for the "
  "diffusion layer.")

# ======================================================================
H("3 Baseline Scheme and Resource-Bottleneck Analysis", 1)
H("3.1 Baseline Scheme Flow", 2)
P("To give the subsequent resource accounting and optimization a concrete object, this section gives the complete "
  "processing flow of the scheme: first the notation and data partition are fixed, then the four layers of the "
  "encryption side, key derivation and ciphertext encapsulation are explained step by step, then the inverse "
  "operation of the decryption side is given, and finally the reversibility of the whole chain is proved. The flow "
  "is described below at the T=32 implementation caliber; the unoptimized T=128 version appears only as a reference "
  "in the resource audit (Table 1). The overall structure is shown in Fig. 1.")
FIG("fig_1.png",
    "Fig. 1 Overall pipeline (64x64 grayscale, block length L=256). Solid lines are data flow, dashed lines are "
    "ciphertext-header assembly, orange is the quantum-resource branch; on the decryption side the permutation is "
    "expanded directly from the seed after MAC verification, and the quantum walk is not re-run.", 6.1)
H("3.1.1 Notation and Data Partition", 3)
P("Let the plaintext image of H=W=64, 8-bit grayscale be row-expanded into a byte vector P=(p_0,...,p_{N-1}) of "
  "length N=H*W=4096. The whole pipeline forms a 20-qubit NEQR state |I> with a 12-qubit position register and an "
  "8-qubit color register. Expand P into 8 bit planes P^0,...,P^7 (P^b is the HxW binary matrix formed by the b-th "
  "bit, b=0 is the least significant bit). In the permutation stage P is further divided into B=N/L=16 blocks of "
  "length L=256, the i-th block denoted P^{(i)}. The master key is denoted MK, the random number is denoted nonce, "
  "the operator XOR denotes bitwise exclusive-or and (plus) denotes modular addition by 256. In the following steps the "
  "nominal parameters are T=32 (walk steps), L=256 (cycle length), coin-angle interval theta in [0.1pi,0.4pi].")
H("3.1.2 Encryption Flow", 3)
P("Encryption serially concatenates four layers: quantum bit-plane rotor (QPR), quantum-walk permutation (QWP), "
  "bidirectional-chain diffusion (BCD) and dynamic DNA coding; a quantum-resource branch (nonce generation and walk "
  "evolution) is interleaved. The input, operation and output of each step are as follows.")
STEP("Step E1 (random-number generation): ",
     "If the caller does not explicitly supply a nonce, a 20-qubit quantum hybrid random-number generator (QHRNG) "
     "measures the seed seed=MK||'|nonce|'||ctr once, and the first 16 bytes are taken as the nonce. The random "
     "nonce makes two encryptions of the same plaintext under the same master key differ, which is the prerequisite "
     "for subsequently decoupling the key from the plaintext.")
STEP("Step E2 (QWP angle derivation): ",
     "For the i-th block (i=0,...,B-1) compute seed_i=MK||'|qwp|'||nonce||BE_32(i), expand 4096 bytes in CTR mode "
     "of SHA3-256 and normalize to a sequence f; take theta_s=theta_lo+(theta_hi-theta_lo)*f_s (s=1,...,T) as the "
     "coin angle of each step, and phi=2*pi*f_{T+1} as the initial phase. The angles are determined entirely by "
     "(MK,nonce,i) and are independent of the plaintext, so they can be reproduced independently at the decryption end.")
STEP("Step E3 (bit-plane rotor QPR): ",
     "On the 20-qubit NEQR register, in order: (1) bit-plane cyclic shift; (2) plane rotation; (3) Toffoli and "
     "injection plane_b <- plane_b XOR (plane_{b+1} AND plane_{b+4}); (4) cross-plane CNOT chain "
     "plane_b <- plane_b XOR plane_{b-1}. All four are reversible gates, forming the first perturbation of the "
     "confusion layer.")
STEP("Step E4 (walk evolution): ",
     "On the 9-qubit register of the i-th block (1 coin bit + 8 position bits), the initial state is taken as "
     "(cos(pi/8)|0> + e^{i*phi} sin(pi/8)|1>) (x) |0>, and U^T = QFT^dagger * (D * C_T)...(D * C_1) * QFT is applied, "
     "where C_s is the s-th step coin rotation and D is the diagonalized conditional-shift layer. The whole circuit "
     "is loaded once and evolves to the end, with no measurement inserted in the middle.")
STEP("Step E5 (measurement and distribution extraction): ",
     "After evolution, read the position-register probability distribution d_i=(d_i(0),...,d_i(255)). Under the "
     "error-free model it is taken from the state-vector modulus squared; under the noisy model it is given by the "
     "diagonal of the density matrix. This distribution is highly sensitive to (MK,nonce) and is the entropy source "
     "of the permutation.")
STEP("Step E6 (quantization, seed encapsulation and permutation): ",
     "Quantize each block distribution into a 32-bit unsigned integer q_i(k)=round(d_i(k)*2^32), feed the blocks in "
     "order into SHA3-256 to output a 32-byte permutation seed S, compressing 16*256 reals into a single 256-bit "
     "digest. Then use k_i=SHA3-CTR(S||BE_32(i)||'|fy|',4L) as the random source to perform a Fisher-Yates shuffle "
     "on the i-th block, giving the permutation pi_i, and permute block-wise s_2^{(i)}[j]=s_1^{(i)}[pi_i(j)]. Since S "
     "is encapsulated in the ciphertext header, this permutation can be reproduced at the decryption end from the "
     "same S without re-running the quantum walk.")
STEP("Step E7 (bidirectional-chain diffusion BCD): ",
     "Apply a forward chain a[i]=(S_inv[a[i-1]]+(ks1[i]+a[i-1])) mod 256 and then a reverse chain "
     "b[i]=(S_inv[b[i]]+(ks2[i]+b[i])) mod 256 (S_inv is the AES inverse S-box, ks1/ks2 are key streams) to the "
     "permuted byte sequence, completing modular-addition-S-box bidirectional diffusion; its inverse is exactly "
     "invertible byte by byte under known key streams and initial values.")
STEP("Step E8 (dynamic DNA coding): ",
     "XOR-code the diffusion result under a pixel-wise selected DNA rule to obtain the ciphertext payload. The key "
     "streams ks1/ks2 are expanded from SHA3-256(MK||nonce) in CTR mode.")
P("After encryption the ciphertext header is assembled: the first 16 bytes are the nonce, then 16 bytes are the "
  "message-authentication-code tag=SHA3-256(MK||nonce||ciphertext)[:16], then a 32-byte permutation seed S; the "
  "ciphertext payload is the DNA-coded byte stream. The ciphertext header totals 64 bytes (1.56% of the payload), "
  "allowing the decryption end to reconstruct all key-dependent structures without re-running the walk.")
H("3.1.3 Decryption Flow", 3)
P("Decryption is the strict inverse order of encryption, and because the ciphertext header already carries the nonce "
  "and seed, no repeated quantum-resource branch is needed:")
STEP("Step D1 (parse and verify): ",
     "Read the nonce, tag and seed S from the ciphertext header; recompute tag'=SHA3-256(MK||nonce||ciphertext)[:16] "
     "with the same formula; if tag' != tag the ciphertext is judged tampered and aborts, providing integrity "
     "protection.")
STEP("Step D2 (inverse DNA coding): ",
     "Apply the inverse XOR under the same pixel-wise selected DNA rule as encryption to recover the byte stream "
     "before modular-addition-S-box diffusion.")
STEP("Step D3 (inverse bidirectional-chain diffusion): ",
     "Apply the inverse of the reverse chain in reverse cipher-stream order prev=(a[i]-ks2[i]-(i==0?iv1:prev)) mod "
     "256, then the inverse of the forward chain prev=(S[a[i]]-ks1[i]-(i==0?iv0:prev)) mod 256, where iv0/iv1 are "
     "fixed by key derivation; this inverse is byte-wise mutually inverse with E7.")
STEP("Step D4 (inverse bit-plane rotor QPR): ",
     "Execute in reverse order of E3: first undo the CNOT chain, then undo the Toffoli injection, then reverse the "
     "plane rotation, finally reverse the bit-plane cyclic shift, to recover the permuted byte sequence.")
STEP("Step D5 (inverse permutation expansion): ",
     "Rebuild the permutation pi_i from the seed S in the ciphertext header through the same Fisher-Yates process "
     "as E6, and apply the inverse permutation r^{(i)}[pi_i(j)]=s^{(i)}[j] to recover the original block order.")
STEP("Step D6 (quantum walk need not be re-run): ",
     "Because the permutation is already deterministically reproduced at the decryption end through the seed S, the "
     "quantum-walk circuit is executed only once at the encryption end, and decryption correctness is completely "
     "decoupled from hardware noise - the core benefit of the architecture hardening in Section 5.1.")
H("3.1.4 Reversibility Argument", 3)
P("The reversibility of the whole chain is guaranteed by four pairs of mutually inverse maps. The four items of QPR "
  "are all reversible gates (Toffoli and its inverse, CNOT and itself, cyclic shift and rotation are all "
  "permutations), so E3/D4 are inverses. The QWP permutation is given by a Fisher-Yates shuffle, whose inverse is "
  "obtained by rebuilding pi_i from the same random source S and applying pi_i^{-1}, consistent with E6/D5. The BCD "
  "bidirectional chain is a cascade of modular-256 addition and the AES S-box, which is exactly invertible byte by "
  "byte under known key streams and initial values (E7/D3). DNA coding is base XOR, whose inverse is XOR under the "
  "same rule (E8/D2). Concatenating the four inverse maps, for any plaintext, decryption(encryption(P))=P holds "
  "strictly in the deterministic (noiseless) model; under the noisy model, because the permutation is encapsulated "
  "in the ciphertext header and the walk no longer participates in decryption, decryption depends only on classical "
  "reversible operations, so noise does not accumulate and does not destroy decryptability. This conclusion is "
  "verified in Chapter 6 experiments by 6/6 byte-wise lossless decryption of images.")
H("3.2 Resource Audit", 2)
P("On the hardware basis {rz, sx, x, cx} and compilation optimization level 0 (structural caliber, to avoid the "
  "optimizer hiding structural cost), the circuits of each baseline stage are accounted for, and the results are "
  "shown in Table 1. The conditional-shift operator is the source of all overhead: on n=8 position qubits, a "
  "controlled cyclic add/subtract implemented by multi-controlled ripple carry needs 902 CNOTs, 1,804 per step for "
  "both directions; T=128 steps, 16 blocks, and the walk must be executed once at each of the encryption and "
  "decryption ends, so the walk alone consumes 3,805,568 two-qubit gates, and with the QPR layer the per-image "
  "total is 3,809,336. At the same time, the maximum coherence depth of the walk block reaches 259,841 layers - "
  "under the current superconducting-device two-qubit error rate of about 1e-3, the expected number of errors at "
  "this depth exceeds 200, and the circuit output degenerates completely into noise.")
TBL("Table 1 Circuit resource accounting of the baseline and improved schemes (hardware basis {rz, sx, x, cx}, optimization level 0)",
    ["Circuit", "Depth", "CNOT", "Total gates"],
    [["QHRNG (20 qubits)", "20", "38", "230"],
     ["QWP, multi-controlled ripple carry, T=128/block (v1)", "259,841", "119,040", "386,949"],
     ["QWP, stepwise QFT incrementer, T=128/block", "39,424", "38,912", "101,253"],
     ["QWP, QFT frame folding, T=32/block (v2)", "1,302", "648", "2,053"],
     ["QPR forward/inverse (20 qubits)", "113", "56", "160"]],
    widths=[3.1, 1.0, 1.0, 1.0])
P("It should be emphasized that the above table is at the single-block caliber. The walk-block circuit must be "
  "repeated for 16 blocks, so the total two-qubit gate count scales linearly with the number of blocks; whereas the "
  "depth caliber reflects the longest chain that must be executed error-free consecutively, and is the first "
  "criterion of NISQ feasibility. This audit conclusion makes the optimization goal explicit: lower the single-block "
  "depth and reduce the number of walk executions.", indent=True)
H("3.3 Architectural Defect Analysis", 2)
P("Besides circuit overhead, the resource audit also exposes three architectural defects of the baseline scheme. "
  "First, dual-end repeated execution of the walk: the permutation is derived from the walk distribution, so the "
  "decryption end must re-run the walk circuit with the same parameters to rebuild the same permutation; but the "
  "measured distribution of real hardware differs run by run under noise, causing the decryption permutation to "
  "disagree with the encryption permutation - noise not only lowers fidelity but directly destroys decryptability. "
  "Second, key derivation depends on the plaintext digest: the key stream is derived jointly from the master key and "
  "SHA3-256(plaintext), meaning the same plaintext under the same master key always yields the same ciphertext, so "
  "the scheme lacks semantic security, and plaintext-digest participation in key derivation is not standard practice "
  "in the provable-security framework. Third, the diffusion layer uses an XOR bidirectional chain, and the "
  "differential analysis in Section 5.3 shows a structural truncation in its diffusion propagation.")

# ======================================================================
H("4 Circuit-Level Depth-Reduction Optimization", 1)
P("This chapter gives four circuit-level mechanisms. The first three act on the walk-block circuit and the fourth "
  "on the QPR layer; their common idea is to transform the multi-qubit controlled structure that grows linearly with "
  "the number of steps into an equivalent form that is independent of the number of steps or requires only "
  "single-qubit gates.")
H("4.1 QFT-Controlled Incrementer (OPT-1)", 2)
P("The cyclic-shift operator on a cycle is a circulant matrix and can be diagonalized simultaneously with the "
  "quantum Fourier transform [29]. In the Fourier basis, the controlled +/-1 cyclic shift decomposes into m "
  "controlled-phase gates and a pair of QFTs:")
EQ("S_{+/-} = QFT^dagger * diag(e^{+/-i*phi(k)}) * QFT ,   phi(k) = 2*pi*k/2^m .")
P("Compared with the multi-controlled ripple-carry cascade, this implementation reduces the per-direction cost from "
  "902 CNOTs to m-1=7 controlled-phase gates (the highest phase 2*phi(2^{m-1})=2*pi can be dropped) plus a pair of "
  "QFTs. After calibration with the official QFT library, exhaustive verification over all shift amounts at scales "
  "2^3, 2^4, 2^6 holds exactly. The T=128 walk-block cost drops from 119,040 to 38,912 CNOTs (Table 1), with depth "
  "reduced to 39,424 layers.")
H("4.2 QFT Frame Folding (OPT-2)", 2)
P("In the stepwise incrementer implementation, a T-step walk contains 2T Fourier transforms. Noting that the coin "
  "operator acts only on the coin qubit and commutes with the Fourier transform acting on the position register, the "
  "QFT^dagger * QFT between adjacent steps cancels exactly:")
P("Proposition 1 (Frame folding). For any step order and coin-angle sequence, U^T = (S C_T)...(S C_1) satisfies "
  "U^T = QFT * (D C_T)...(D C_1) * QFT^dagger, where D is the diagonal operator in the Fourier basis, formed by the "
  "composite of a per-step unconditional phase layer and a controlled-phase layer.", indent=False)
P("Proof sketch: induction on T. For a single step S_{+/-} = QFT^dagger D QFT is the definition; assuming the first "
  "T-1 steps are folded, the QFT introduced by the T-th step's conditional shift is adjacent to the QFT^dagger at "
  "the end of the previous step and cancels, and the coin operator C_T can move freely through the diagonal operator "
  "on the position register to the outermost level. QED.", indent=False)
P("The folded walk circuit is shown in Fig. 2: the whole circuit contains only one pair of QFTs and T single-step "
  "layers. At T=32, the per-block cost is further reduced to 648 CNOTs and depth 1302 layers. To verify equivalence, "
  "the output distribution of the full T=32 walk is compared between the folded and stepwise implementations by "
  "state-vector simulation, with total-variation distance TV < 3e-14 over the full space, i.e., the two are "
  "identical to double precision.")
FIG("fig_2.png",
    "Fig. 2 Walk circuit after QFT frame folding (schematic with n_pos=4, T=3; actual deployment n_pos=8, T=32). "
    "Each step contains only one coin rotation and one diagonal layer D, and the Fourier transform appears only once.",
    6.2)
H("4.3 +/-1 Direction Fusion (OPT-3)", 2)
P("Each walk step contains two conditional shifts, forward and reverse. After separating the unconditional phase "
  "(coin-independent) from the controlled phase, the two directions can be fused into: n single-qubit phase gates (0 "
  "CNOTs) plus n-1 controlled-phase gates. Fig. 3 gives an intuitive comparison of three conditional-shift "
  "implementations: multi-controlled ripple cascade, stepwise QFT incrementer, and folded diagonal layer. From Table 1, "
  "after fusion each step has only 7 controlled-phase gates left, and the controlled phase can be scheduled "
  "contiguously with the next coin rotation, further compressing the scheduling gap.")
FIG("fig_3.png",
    "Fig. 3 Comparison of three implementations of the conditional-shift operator: (a) multi-controlled ripple carry "
    "(v1 baseline); (b) stepwise QFT incrementer; (c) the folded diagonal layer D of this work (only single-qubit "
    "phase gates and controlled-phase gates).", 6.2)
H("4.4 Compile-Time Bit-Plane Relabeling (OPT-4)", 2)
P("The cyclic shift and plane rotation of the QPR layer are implemented by a SWAP network in the baseline. Analysis "
  "shows that the cyclic relabeling of bit planes is a fixed-point transformation of the cross-plane CNOT chain and "
  "the injection structure: after absorbing the rotation result into the renaming of subsequent plane indices, all "
  "SWAPs can be eliminated at compile time. After elimination the QPR forward circuit has only 8 Toffoli and "
  "injections (Step 3) and 8 cross-plane CNOTs (Step 4), totaling 56 CNOTs and depth 113 layers; the inverse circuit "
  "is its reverse order. The forward and inverse circuits are shown in Fig. 4.")
FIG("fig_4.png",
    "Fig. 4 QPR circuit after compile-time relabeling: (a) forward; (b) inverse. On the 20-qubit NEQR register only "
    "8 Toffoli and injections and 8 CNOTs remain, and the SWAP network is completely eliminated.", 6.2)
H("4.5 Resource Summary", 2)
P("Table 2 and Fig. 5 summarize the two-qubit gate budget and maximum coherence depth of a single 64x64 image under "
  "three implementations. The improved scheme (v2) compresses 363.5x relative to the v1 multi-controlled "
  "implementation and 118.8x relative to the stepwise QFT implementation; the depth caliber drops from 259,841 layers "
  "to 1,302 layers. Using 'expected number of errors < 1' as the criterion, the required two-qubit error-rate upper "
  "bound is relaxed from 8.4e-6 to 1.5e-3, entering the same order of magnitude as the median of publicly calibrated "
  "IBM device data (about 1e-3).")
TBL("Table 2 Two-qubit gate budget and error budget of a single 64x64 image (optimization level 0, structural caliber)",
    ["Implementation", "Walk executions", "Total two-qubit gates", "Max block depth", "Req. two-qubit error-rate bound"],
    [["v1 multi-controlled ripple, T=128, both ends", "2x16", "3,809,336", "259,841", "8.4e-6"],
     ["v1 stepwise QFT, T=128, both ends", "2x16", "1,245,240", "39,424", "2.6e-5"],
     ["v2 QFT frame folding, T=32, encryption only", "1x16", "10,480", "1,302", "1.5e-3"]],
    widths=[2.2, 0.9, 1.2, 1.0, 1.4])
FIG("fig_5.png",
    "Fig. 5 Two-qubit gate budget (a) and maximum coherent-block depth (b) of the three implementations, log scale.",
    5.8)

# ======================================================================
H("5 Architecture-Level Hardening", 1)
H("5.1 Permutation-Seed Encapsulation (PSE)", 2)
P("The use of the walk measurement distribution is changed from 'directly serving as the permutation' to 'serving "
  "as an entropy source': the encryption end derives the walk angles from the master key and nonce, executes the "
  "folded walk and measures each position distribution, hashes the distribution into a 256-bit permutation seed S; S "
  "is transmitted in ciphertext form through the ciphertext header. The decryption end only needs to recover S from "
  "the key stream and then deterministically rebuild the permutation pi from S by the Fisher-Yates expansion "
  "algorithm, without executing any quantum circuit at all. This mechanism brings three effects: first, decryption "
  "correctness is completely decoupled from hardware noise - even if the walk distribution is perturbed by noise, "
  "only the value of seed S changes, and S itself is reliably delivered through the ciphertext header; second, the "
  "quantum-resource overhead is halved and moved outside the decryption path, with the decryption-end two-qubit gate "
  "budget being only the 56 of the QPR inverse circuit; third, the walk distribution is still injected into the "
  "permutation in a key-dependent manner, so the statistical-security properties (Sections 6.6, 6.7) are unaffected.")
H("5.2 Randomized Nonce and Integrity Tag", 2)
P("Key derivation is changed to SHA3-256(master key || nonce), where the nonce is generated by a 20-qubit quantum "
  "hybrid random-number generator (QHRNG): a Hadamard layer provides uniform superposition, a parameterized rotation "
  "layer injects device characteristics, a DCX chain produces entanglement correlations, and 2^13 measurements output "
  "a 16-byte nonce (circuit in Fig. 6). The nonce enters the ciphertext header in plaintext, so the same plaintext "
  "under different nonces yields different ciphertext and the scheme gains semantic security. The ciphertext header "
  "also carries a 16-byte MAC tag covering the nonce, seed and all payload; the MAC is verified before decryption, "
  "and any tampering is rejected before the image is recovered, avoiding a chosen-ciphertext attack surface on the "
  "decryption oracle. The ciphertext header totals 64 bytes (nonce 16 + seed 32 + MAC 16), 1.56% of the payload.")
FIG("fig_6.png",
    "Fig. 6 Quantum hybrid random-number generator circuit (20 qubits): Hadamard layer, parameterized rotation layer, "
    "DCX entanglement chain and measurement.", 6.0)
H("5.3 BCD-A Modular-Addition-S-box Bidirectional Diffusion", 2)
P("A differential analysis of the baseline XOR bidirectional chain shows a structural flaw: the forward chain "
  "c[i] = r[i] XOR c[i-1] is a linear recurrence, and the differential produced by a single-byte input change cancels "
  "alternately by position parity during propagation, with a measured single-byte change propagating to only 37.79% "
  "of output bytes; under a fixed nonce, the NPCR of a single-bit plaintext perturbation is only 5.83%. To this end "
  "the bidirectional chain is replaced by a cascade of modular addition and the AES S-box (BCD-A): the forward chain "
  "is changed to c[i] = S(r[i] (plus) ks1[i] (plus) c[i-1]), and the reverse chain is expanded by the inverse S-box "
  "and modular subtraction, ensuring reversibility while making the differential amplify monotonically along the "
  "chain. The measured single-byte change propagation rate rises to 99.66%, and the NPCR/UACI of a single-bit "
  "plaintext perturbation under fixed nonce reaches 99.90%/33.59% (Tables 3, 6). This result also shows that, beyond "
  "the quantum permutation layer, the nonlinear design of the classical diffusion layer remains the key link "
  "determining differential performance.")
TBL("Table 3 Diffusion-layer differential comparison (output propagation rate of single-byte input change)",
    ["Diffusion layer", "Spread rate", "NPCR under 1-bit plaintext perturbation (fixed nonce)"],
    [["XOR bidirectional chain (baseline)", "37.79%", "5.83%"],
     ["Mod-add + S-box bidirectional chain (BCD-A)", "99.66%", "99.90%"]],
    widths=[2.2, 1.2, 2.4])

# ======================================================================
H("6 Experimental Results and Analysis", 1)
H("6.1 Experimental Setup", 2)
P("All experiments are based on Qiskit 1.x and the Aer state-vector/density-matrix backend, with circuit accounting "
  "unified to the hardware basis {rz, sx, x, cx} and noise evaluation using the density-matrix method with a "
  "depolarizing-channel model. The test images are taken from the Gonzalez digital-image-processing standard test "
  "library; six copyright-unencumbered images - Lake, Baboon, Peppers, Cameraman, Jetplane and House - are selected, "
  "uniformly scaled to 64x64, 8-bit grayscale; some classical test images have disputed copyright ownership and are "
  "not used in this work, and all test images come from freely usable textbook standard libraries. The master key is "
  "a fixed 26-byte string, the nonce is generated by QHRNG in the main experiment and fixed separately in the "
  "sensitivity experiment to isolate variables. All illustrations are rendered at 600 dpi.")
H("6.2 Losslessness and Encryption Effect", 2)
P("Table 4 gives the main experimental results on the six images. All 6 images achieve byte-wise lossless decryption, "
  "6/6 round-trip correctness; all MAC tags pass verification. The ciphertext information entropy is between 7.9498 "
  "and 7.9637, with deviation from the 8-bit theoretical upper bound no more than 0.051, and independent of plaintext "
  "complexity - the House plaintext with the lowest entropy (5.5982) still yields ciphertext entropy 7.9498 after "
  "encryption, indicating that the confusion-diffusion cascade has completely erased the plaintext statistical "
  "features. Fig. 7 visually shows the plaintext, ciphertext and ciphertext histogram: the six ciphertexts are "
  "uniformly noise-like visually, with no visible peak/valley structure in the histogram.")
TBL("Table 4 Main experimental results on the standard image set (T=32, folded walk)",
    ["Image", "Plaintext entropy", "Ciphertext entropy", "NPCR/%", "UACI/%", "chi^2 (ciphertext)", "Lossless round-trip", "MAC"],
    [["Lake", "7.4195", "7.9541", "99.56", "30.52", "253.1", "Pass", "Pass"],
     ["Baboon", "6.9729", "7.9637", "99.73", "26.83", "202.5", "Pass", "Pass"],
     ["Peppers", "7.4700", "7.9512", "99.61", "29.00", "277.8", "Pass", "Pass"],
     ["Cameraman", "6.9709", "7.9566", "99.61", "31.02", "245.8", "Pass", "Pass"],
     ["Jetplane", "6.6790", "7.9586", "99.58", "30.88", "233.9", "Pass", "Pass"],
     ["House", "5.5982", "7.9498", "99.73", "29.84", "282.2", "Pass", "Pass"]],
    widths=[1.1, 0.9, 0.9, 0.8, 0.8, 1.0, 0.9, 0.7])
P("In the table chi^2 is the Pearson chi-square statistic of the ciphertext grayscale distribution against the "
  "uniform distribution, with 255 degrees of freedom and a 0.05 significance-level critical value of 293.2; all six "
  "images are below the critical value, i.e., the ciphertext grayscale distribution is not significantly different "
  "from the uniform distribution. It should be noted that the NPCR/UACI in the table are difference measures between "
  "plaintext and ciphertext, whose values fluctuate with the plaintext distribution itself, reflecting the overall "
  "rewriting ability of confusion and diffusion; the sensitivity test in the differential-attack sense is given in "
  "Section 6.4 by ciphertext-ciphertext comparison, with ideal values NPCR is about 99.61%, UACI is about 33.46% [26].",
  indent=True)
FIG("fig_7.png",
    "Fig. 7 Plaintext (top), ciphertext (middle) and ciphertext grayscale histogram (bottom) of the six test images.",
    6.2)
H("6.3 Correlation Analysis", 2)
P("Adjacent-pixel correlation is a classical test item for image encryption. Table 5 lists the Pearson correlation "
  "coefficients of 4096 adjacent-pixel pairs in the horizontal, vertical and diagonal directions for the six images "
  "in plaintext and ciphertext. The plaintext correlation is between 0.7582 and 0.9366, showing the typical "
  "strong-correlation feature of natural images; after encryption all are pressed into [-0.0353, 0.0353] with no "
  "directional preference. Fig. 8(b)(c) gives the scatter distribution of horizontally adjacent pixels of the Lake "
  "image before and after encryption: the plaintext scatters tightly clustered on the diagonal, while the ciphertext "
  "scatters uniformly over the whole plane, consistent with the visual and numerical conclusions.")
TBL("Table 5 Correlation coefficients of adjacent pixels (plaintext -> ciphertext, H/V/D)",
    ["Image", "Plaintext (H/V/D)", "Ciphertext (H/V/D)"],
    [["Lake", "0.9021 / 0.9086 / 0.8399", "-0.0058 / -0.0232 / 0.0081"],
     ["Baboon", "0.8713 / 0.8944 / 0.7988", "0.0184 / -0.0040 / -0.0015"],
     ["Peppers", "0.8614 / 0.8962 / 0.7748", "0.0059 / 0.0081 / -0.0119"],
     ["Cameraman", "0.9152 / 0.9366 / 0.8649", "0.0353 / -0.0027 / 0.0101"],
     ["Jetplane", "0.8840 / 0.8402 / 0.7582", "0.0035 / -0.0213 / 0.0051"],
     ["House", "0.9164 / 0.8931 / 0.8389", "0.0170 / -0.0009 / 0.0149"]],
    widths=[1.2, 2.3, 2.3])
FIG("fig_8.png",
    "Fig. 8 Statistical-characteristic analysis: (a) Lake plaintext/ciphertext histogram comparison; (b)(c) "
    "horizontal adjacent-pixel scatter (plaintext/ciphertext); (d) horizontal correlation comparison; (e) "
    "plaintext/ciphertext information entropy; (f) ciphertext chi-square statistic vs critical value.", 6.2)
H("6.4 Key, Nonce and Plaintext Sensitivity", 2)
P("The sensitivity test isolates each variable under a fixed nonce, and the results are shown in Table 6. Under three "
  "perturbations - master key flipped by 1 bit, nonce flipped by 1 bit, and plaintext flipped by 1 bit - the "
  "ciphertext NPCR is 99.66%, 99.76% and 99.90% respectively, and the UACI all fall near the ideal value 33.46%; "
  "among them the plaintext sensitivity is measured under the most stringent condition of fixed nonce and is "
  "contributed entirely by the diffusion layer, directly verifying the effectiveness of the BCD-A fix in Section 5.3. "
  "When decrypting with a wrong key, the NPCR of the output against the plaintext is 99.54%, and the MAC verification "
  "must fail, so the attacker cannot obtain any key-verification bypass from the decryption output.")
TBL("Table 6 Sensitivity test (Lake, T=32, fixed nonce)",
    ["Perturbation", "NPCR/%", "UACI/%", "MAC check"],
    [["Master key flipped by 1 bit (re-encryption)", "99.66", "33.81", "-"],
     ["nonce flipped by 1 bit (re-encryption)", "99.76", "33.36", "-"],
     ["Plaintext flipped by 1 bit (fixed nonce)", "99.90", "33.59", "-"],
     ["Decryption with wrong key (vs plaintext)", "99.54", "30.99", "Fail"]],
    widths=[2.4, 1.0, 1.0, 1.0])
H("6.5 Trimming the Walk Step Count T", 2)
P("The walk step count T is a linear factor of the circuit overhead, and its lower bound is determined by whether the "
  "security metrics saturate. Table 7 gives the key metrics of Lake and Baboon at T in {8,16,32,64,128}. Entropy and "
  "correlation are already near saturation at T=8, and NPCR/UACI enter a fluctuation band after T=16; increasing T "
  "from 32 to 128 brings no systematic improvement. Accordingly T*=32 is taken as the default parameter, compressing "
  "the walk-circuit overhead to 1/4 of the baseline (relative to T=128) while maintaining statistical metrics. This "
  "trimming is orthogonal to the frame folding in Section 4.2: the former cuts the linear factor and the latter cuts "
  "the per-step constant, and their gains multiply.")
TBL("Table 7 Walk step-count trimming experiment (Lake / Baboon)",
    ["T", "Ciphertext entropy (Lake/Baboon)", "NPCR/%", "UACI/%", "Corr. H", "chi^2"],
    [["8", "7.9537 / 7.9546", "99.58 / 99.56", "30.60 / 26.72", "-0.0450 / 0.0089", "254.8 / 256.4"],
     ["16", "7.9575 / 7.9604", "99.78 / 99.71", "30.82 / 26.78", "-0.0260 / 0.0032", "236.5 / 224.5"],
     ["32", "7.9593 / 7.9516", "99.66 / 99.56", "30.12 / 26.65", "0.0020 / -0.0087", "228.8 / 275.4"],
     ["64", "7.9571 / 7.9533", "99.66 / 99.68", "30.42 / 26.80", "0.0092 / -0.0110", "239.9 / 260.8"],
     ["128", "7.9597 / 7.9554", "99.68 / 99.56", "30.77 / 27.16", "0.0056 / -0.0152", "226.9 / 243.2"]],
    widths=[0.5, 1.8, 1.5, 1.5, 1.7, 1.4], fontsize=8)
H("6.6 NIST SP 800-22 Randomness Test", 2)
P("The ciphertexts under 62 different nonces are concatenated into a 2,031,616-bit aggregated bit stream, and the "
  "NIST SP 800-22 test suite [20] is executed at 262,144 bits per sequence, 7 sequences in total, significance level "
  "alpha=0.01; the results are shown in Table 8. Of the 3850 executable sub-tests, 3803 pass, a pass rate of 98.8%, "
  "consistent with the expected about 99% at alpha=0.01; the marginal failing items concentrate on individual "
  "sequences of Frequency, Runs and Cumulative Sums, and are excluded as the inherent fluctuation of the test items "
  "under a small number of sequences through a control experiment with the SHA3-CTR reference random stream under the "
  "same pipeline, rather than a systematic flaw of the ciphertext stream. Overlap Template and Linear Complexity are "
  "not applicable due to sequence-length constraints and are marked N/A.")
TBL("Table 8 NIST SP 800-22 test details (2.03 Mbit aggregated ciphertext, 7 sequences x 262,144 bit, alpha=0.01)",
    ["Test item", "Pass/Total", "Pass rate", "Mean p-value"],
    [["Frequency (Monobit)", "6/7", "0.857", "0.4885"],
     ["Frequency within Block", "7/7", "1.000", "0.3833"],
     ["Runs", "6/7", "0.857", "0.4838"],
     ["Longest Run of Ones", "7/7", "1.000", "0.3750"],
     ["Matrix Rank", "7/7", "1.000", "0.4149"],
     ["Spectral (DFT)", "7/7", "1.000", "0.5682"],
     ["Non-overlapping Template", "3542/3584", "0.988", "0.4925"],
     ["Maurer Universal", "7/7", "1.000", "0.9980"],
     ["Serial", "14/14", "1.000", "0.4057"],
     ["Approximate Entropy", "7/7", "1.000", "0.5754"],
     ["Cumulative Sums", "12/14", "0.857", "0.3887"],
     ["Random Excursions", "55/56", "0.982", "0.5349"],
     ["Random Excursions Variant", "126/126", "1.000", "0.5981"],
     ["Total", "3803/3850", "0.988", "-"]],
    widths=[2.4, 1.2, 1.0, 1.0], fontsize=8)
H("6.7 Attacks and Robustness", 2)
P("Table 9 gives the decryption behavior under tamper and noise attacks. Any single-bit ciphertext flip causes MAC "
  "verification to fail, with a tamper-detection rate of 5/5; even if the attacker bypasses the MAC and decrypts "
  "directly, a single-bit flip affects only local pixels (PSNR 38.31 dB), thanks to PSE limiting permutation to "
  "deterministic expansion rather than noise-sensitive re-running. The PSNR of 16x16 occlusion and 25% cropping is "
  "19.51 dB and 10.96 dB respectively, indicating that large-scale ciphertext loss makes content unrecoverable - an "
  "acceptable behavior under the MAC-first-reject mechanism: ciphertext-level integrity protection takes priority "
  "over local recoverability, which is a different design orientation from discriminable tampering (e.g., "
  "content-level watermarking). The salt-and-pepper noise experiment (1%, 5%) is also entirely detected by the MAC.")
TBL("Table 9 Decryption behavior under tamper and noise attacks (Lake)",
    ["Attack", "PSNR/dB", "NPCR vs plaintext/%", "MAC detected"],
    [["Single-bit flip", "38.31", "0.29", "Yes"],
     ["16x16 occlusion (cipher block zeroed)", "19.51", "24.83", "Yes"],
     ["25% cropping", "10.96", "92.63", "Yes"],
     ["Salt-and-pepper noise 1%", "23.30", "10.91", "Yes"],
     ["Salt-and-pepper noise 5%", "16.34", "43.24", "Yes"]],
    widths=[2.4, 1.1, 1.4, 1.0])
P("For known-plaintext and chosen-plaintext attacks, Table 10 gives the ciphertext properties of three special "
  "plaintexts - all-zero, all-255 and checkerboard - under the same key and nonce: the ciphertext entropy of all "
  "three exceeds 7.95, all chi-square statistics are below the critical value, and the NPCR against the Lake "
  "ciphertext is about 99.5%. This shows that even if the attacker can deliberately construct extreme plaintext and "
  "obtain the corresponding ciphertext, the ciphertext still has no exploitable statistical structure; combined with "
  "the key stream's dependence on the nonce (Section 6.4), the ciphertext of the same plaintext in different "
  "communication sessions is completely different, so a chosen-plaintext attack cannot be reused across sessions.")
TBL("Table 10 Ciphertext properties under special plaintext (fixed key and nonce)",
    ["Special plaintext", "Ciphertext entropy", "chi^2", "Corr. H", "NPCR vs Lake ciphertext/%"],
    [["All-zero image", "7.9598", "229.6", "0.0031", "99.54"],
     ["All-255 image", "7.9522", "269.0", "-0.0145", "99.54"],
     ["Checkerboard image", "7.9573", "238.6", "-0.0019", "99.58"]],
    widths=[1.4, 1.0, 0.9, 1.0, 1.7])
H("6.8 Overhead and Throughput", 2)
P("Table 11 summarizes the protocol overhead and throughput. The 64-byte ciphertext header is 1.56% of the payload, "
  "within an acceptable range; in a pure-Python state-vector simulation environment, encryption averages 1.194 s/image "
  "(including the quantum walk and QHRNG) and decryption averages 0.365 s/image. It should be noted that the "
  "simulation throughput does not represent hardware throughput; the architectural significance is that the decryption "
  "path is completely classical, so the decryption throughput is not limited by quantum-circuit queuing in hardware "
  "deployment, and quantum resources appear only in the two short-depth blocks of the walk and QHRNG at the "
  "encryption end.")
TBL("Table 11 Protocol overhead and throughput (simulation environment)",
    ["Metric", "Value"],
    [["Ciphertext header length (nonce+seed+MAC)", "64 B (1.56% of payload)"],
     ["Encryption time (mean)", "1.194 s/image"],
     ["Decryption time (mean, no quantum circuit)", "0.365 s/image"],
     ["Encryption throughput", "3,430.5 B/s"],
     ["Decryption throughput", "11,213.6 B/s"]],
    widths=[2.8, 2.0])

# ======================================================================
H("7 Comparison with Recent Literature", 1)
H("7.1 Comparison Objects and Dimensions", 2)
P("To locate the position of this scheme in the field, representative works from 2019-2026 that are close to or "
  "complementary to the route of this work are selected for comparison, including: quantum-walk key-stream scheme "
  "[5], scheme combining quantum walk with dynamic DNA [6], medical-image scheme with cascaded Chebyshev map [23], "
  "quantum hybrid random-number-generator scheme [7], quantum hyper-scrambling and multi-layer DNA scheme [8], "
  "real-device chaotic OTP scheme [9], and three works focusing on the DTQW circuit implementation itself "
  "[10,11,12]. The comparison dimensions cover six items: image representation, confusion/key source, diffusion "
  "mechanism, execution platform, two-qubit gate-count disclosure and noise evaluation, with results in Table 12.")
TBL("Table 12 Qualitative comparison with recent representative schemes",
    ["Scheme", "Representation/object", "Confusion or key source", "Diffusion", "Platform", "Gate count disclosed", "Noise eval"],
    [["Abd-El-Atty 2019 [5]", "NEQR", "Quantum-walk key stream on cycle", "XOR", "Classical simulation", "No", "No"],
     ["Abd El-Latif 2025 [6]", "Classical image", "Quantum walk + dynamic DNA", "DNA XOR", "Classical simulation", "No", "No"],
     ["Alblehai 2025 [23]", "Medical image", "Quantum walk + Chebyshev map", "Chaotic XOR", "Classical simulation", "No", "No"],
     ["Sridharan 2025 [8]", "NEQR", "QTRNG + 6D hyperchaos", "Multi-layer DNA", "Classical simulation", "No", "No"],
     ["Gururaja 2026 [7]", "NEQR", "QHRNG + bit-level perm. + selective QFT", "Quantum XOR", "Aer + IBM Torino real device", "No", "Partial"],
     ["Abdelfatah 2025 [9]", "NEQR", "MLIH chaotic OTP", "CNOT chain", "IBM Sherbrooke real device", "No (O(n) claimed)", "Partial"],
     ["Wing-Bocanegra 2023 [11]", "DTQW circuit", "- (walk itself)", "-", "IBM Falcon real device", "Yes (decomp. compare)", "Yes (l1 distance)"],
     ["Nandi 2025 [12]", "DTQW circuit", "- (walk itself)", "-", "Classical simulation", "Yes (gates halved)", "No"],
     ["This work", "NEQR", "QFT frame-folded walk + PSE", "Mod-add + S-box (BCD-A) + DNA", "Aer + parametric noise model", "Yes (layer-wise audit)", "Yes (depth-aware)"]],
    widths=[1.5, 0.9, 1.7, 1.2, 1.4, 1.1, 0.9], fontsize=7.5)
P("Table 12 reveals three facts. First, the vast majority of quantum image-encryption schemes use classical "
  "simulation as the only verification means, consistent with the observation of recent surveys [24,28]; only [7,9] "
  "are executed on real devices, and their execution objects are key generation or single-layer circuits rather than "
  "a complete encryption/decryption pipeline. Second, except for [11,12] focusing on the walk circuit itself, "
  "image-encryption schemes generally do not disclose two-qubit gate counts and depth, making their NISQ "
  "feasibility impossible to check independently. Third, this scheme is more complementary than competitive with "
  "[7,9]: the gate-level accounting and depth-aware noise-evaluation method of this work also apply to the circuit "
  "audit of these schemes.", indent=True)
H("7.2 Quantitative Comparison", 2)
P("Fig. 9(a) compares the image scale actually demonstrated on quantum resources by each scheme: the works focusing "
  "on the walk circuit [10,11,12] demonstrate a scale of 16 pixels (about 4 positions x 4 steps), while the "
  "real-device encryption schemes [7,9] and this scheme reach 4096 pixels. In differential resistance (Fig. 9(b)), "
  "the plaintext sensitivity NPCR of this scheme is 99.90% (fixed nonce) and falls in the neighborhood of the "
  "theoretical optimum 99.6094% together with the self-reported 99.97% of [9], and the UACI 33.59% also falls in the "
  "ideal interval 33.28%-33.72%. It should be noted that the metric caliber of the literatures (image scale, "
  "perturbation position, statistical sample count) is not entirely consistent, and this comparison is used to "
  "characterize magnitude and completeness rather than ranking.")
TBL("Table 13 Comparison of key quantitative metrics (values taken from each literature's self-report)",
    ["Metric", "Abdelfatah 2025 [9]", "Gururaja 2026 [7]", "This work"],
    [["Platform", "IBM Sherbrooke (127 qubit)", "IBM Torino + Aer", "Aer + parametric noise model"],
     ["Ciphertext entropy", "7.9977", "NIST pass-rate gain (entropy not reported)", "7.9498-7.9637 (64x64)"],
     ["NPCR", "99.97%", "Not reported", "99.90% (1-bit plaintext, fixed nonce)"],
     ["UACI", "33.89%", "Not reported", "33.59%"],
     ["Two-qubit gates/pixel", "Not disclosed", "Not disclosed", "2.56 (10,480/4096)"],
     ["Error-budget analysis", "Not given", "Qualitative noise-environment compare", "Quantified (error-rate bound 1.5e-3)"]],
    widths=[1.3, 1.7, 1.7, 2.0], fontsize=8)
FIG("fig_9.png",
    "Fig. 9 Comparison with recent literature: (a) image scale demonstrated on quantum resources by each scheme (log "
    "scale); (b) verification-completeness matrix (checkmark explicitly reported, tilde partial/modeled, cross not "
    "reported; compiled from the abstract and body self-reports of each literature).", 6.2)
H("7.3 Relation to DTQW Circuit-Implementation Works", 2)
P("The circuit-level mechanisms of this work are directly related to the DTQW circuit-implementation literature. "
  "Acasiete et al. first implemented walks on cycles, lattices and complete graphs on IBM devices (16 nodes, depth "
  "about 100) [10]; Wing-Bocanegra and Venegas-Andraca systematically compared the performance of the "
  "block-diagonal-decomposition route on Falcon devices, pointing out that the number of multi-controlled gates "
  "determines experimental accuracy [11]; Nandi et al. halved the gate count and depth of the increment-decrement "
  "implementation with an ancilla-free technique [12]; and the Fourier diagonalization of the conditional-shift "
  "operator itself traces back to the classical result of Douglas and Wang [29]. Building on this, this work advances "
  "one step further: frame folding cancels the whole Fourier-transform sequence related to the number of steps, "
  "reducing the per-step cost to a constant magnitude, and for the first time places such optimization in a complete "
  "image-encryption pipeline, cooperating with permutation-seed encapsulation to eliminate the hidden structural "
  "overhead of dual-end walk re-execution.")

# ======================================================================
H("8 Noise Robustness and NISQ Feasibility", 1)
H("8.1 Error Budget", 2)
P("Using 'expected number of errors per block < 1' as the criterion, the tolerable upper bound on the two-qubit "
  "error rate is the reciprocal of the gate count (Table 2). The v1 baseline requires 8.4e-6, about three orders of "
  "magnitude below the current device calibration value, and is infeasible; the v2 folded implementation requires "
  "1.5e-3, entering the same order of magnitude as the median ECR-gate error rate of IBM superconducting devices "
  "(about 1e-3, see the device calibration cited in [9]). This 184x relaxation comes entirely from the circuit "
  "structure itself, without relying on error correction or error mitigation.")
H("8.2 Depth-Aware Noise Evaluation", 2)
P("Fig. 10 gives the degradation curve of the single-block walk distribution under a depolarizing channel. At "
  "p=1e-4, the folded implementation has total-variation distance from the noiseless distribution of only 0.090 and "
  "Pearson correlation with the noiseless distribution of 0.9985, while the same metric of the multi-controlled "
  "implementation has degraded to 0.926/0.439; after p >= 1e-3 all three implementations degrade significantly, "
  "indicating that depth compression expands the usable noise window by about one order of magnitude, but does not "
  "eliminate the fundamental dependence on noise. This is exactly the value of the PSE mechanism in Section 5.1: the "
  "distribution serves only as a one-time entropy source, and the consequence of its noise perturbation is absorbed "
  "by the explicit seed in the ciphertext header, so encryption/decryption consistency no longer depends on "
  "distribution precision. In other words, circuit depth reduction expands the feasible window, and architecture "
  "encapsulation makes degradation outside the window no longer fatal; the two form a complementary complete design.")
FIG("fig_10.png",
    "Fig. 10 Degradation of the walk distribution under depolarizing noise: (a) total-variation distance; (b) "
    "correlation coefficient with the noiseless distribution; (c) relation between walk steps and degradation under "
    "the folded implementation.", 6.2)
H("8.3 Limitations and Outlook", 2)
P("This scheme still has several limitations. First, the depth of the NEQR preparation circuit is not included in the "
  "accounting and optimization, while the preparation depth grows superlinearly with resolution and is the primary "
  "bottleneck at large image scales [25]; block-wise preparation and amplitude-encoding loading are viable follow-up "
  "directions. Second, the noise evaluation is based on the depolarizing channel and the density-matrix method, and "
  "does not cover the joint action of crosstalk, readout error and coherent error; end-to-end execution on real "
  "devices is a necessary next step. Third, there is still a gap between the demonstrated scale of 64x64 and the "
  "real-device executable scale, and the trustworthy execution of QHRNG and the walk block depends on continuous "
  "acquisition of device calibration data. Fourth, the AQFT approximate Fourier transform brings limited additional "
  "benefit in the folding framework (about 2% in measurement), and more aggressive circuit synthesis is worth "
  "exploring.")

# ======================================================================
H("9 Conclusion", 1)
P("At the circuit level: the four mechanisms - QFT-controlled incrementer, QFT frame folding, +/-1 direction fusion "
  "and compile-time bit-plane relabeling - reduce the walk block to 648 CNOTs and depth 1302 layers, compress the "
  "per-64x64-image two-qubit gate budget by 363.5x, lower the maximum coherence depth by about 200x, and relax the "
  "required two-qubit error-rate upper bound by 184x to 1.5e-3. At the architecture level: permutation-seed "
  "encapsulation removes the walk entirely from the decryption path, decoupling decryption correctness from hardware "
  "noise; the randomized nonce and MAC tag provide semantic security and integrity protection; and for the "
  "differential-cancellation flaw of the XOR diffusion chain, the modular-addition-S-box bidirectional chain raises "
  "the single-byte propagation rate from 37.79% to 99.66%. At the experimental level: all six "
  "copyright-unencumbered 64x64 standard images are decrypted losslessly, with ciphertext entropy 7.9498-7.9637, "
  "absolute correlation below 0.036, NPCR/UACI of a single-bit plaintext perturbation under fixed nonce reaching "
  "99.90%/33.59%, all tamper events detected, and 2.03 Mbit of aggregated ciphertext passing the NIST SP 800-22 "
  "test suite at 98.8%.")

# ======================================================================
H("References", 1)
refs = [
    "M. Kaur, V. Kumar. A comprehensive review on image encryption techniques. Archives of Computational Methods in Engineering, 2020, 27: 15-43.",
    "Y. Zhang, K. Lu, Y. Gao, M. Wang. NEQR: a novel enhanced quantum representation of digital images. Quantum Information Processing, 2013, 12(8): 2833-2860.",
    "P. Q. Le, F. Dong, K. Hirota. A flexible representation of quantum images for polynomial preparation, image compression, and processing operations. Quantum Information Processing, 2011, 10(1): 63-84.",
    "N. Jiang, W.-Y. Wu, L. Wang. The quantum realization of Arnold and Fibonacci image scrambling. Quantum Information Processing, 2014, 13(5): 1223-1236.",
    "B. Abd-El-Atty, A. M. Iliyasu, et al. Quantum image encryption based on quantum walks on a circle. Quantum Information Processing, 2019, 18(9): 272.",
    "A. A. Abd El-Latif, M. Almousa, B. Abd-El-Atty. A robust image encryption scheme based on quantum walks and dynamic DNA for secure cloud applications. IEEE Access, 2025, 13: 119110-119130. DOI: 10.1109/ACCESS.2025.3582705.",
    "T. S. Gururaja, P. Pravinkumar. Quantum secure image encryption using hybrid QTRNG and QPRNG. Scientific Reports, 2026, 16: 5151. DOI: 10.1038/s41598-026-35111-6.",
    "S. Sridharan, T. S. Gururaja, R. Amirtharajan, P. Praveenkumar. Quantum scrambling and DNA based multilayer image encryption with QTRNG and 6D hyperchaotic keys. Scientific Reports, 2025, 15: 33933. DOI: 10.1038/s41598-025-10133-8.",
    "R. I. Abdelfatah, R. M. Elsobky, S. A. Khamis. Ultra-secure quantum protection for e-healthcare images: hybrid chaotic one-time pad with cipher chaining encryption framework. Journal of King Saud University - Computer and Information Sciences, 2025, 37(6): 158.",
    "F. Acasiete, F. P. Agostini, J. K. Khatibi Moqadam, R. Portugal. Implementation of quantum walks on IBM quantum computers. Quantum Information Processing, 2020, 19(12): 426.",
    "A. Wing-Bocanegra, S. E. Venegas-Andraca. Circuit implementation of discrete-time quantum walks via the shunt decomposition method. Quantum Information Processing, 2023, 22(3): 146.",
    "B. Nandi, S. Singha, A. Datta, A. Saha, A. Chakrabarti. Robust implementation of discrete-time quantum walks in any finite-dimensional quantum system. Modern Physics Letters A, 2025, 40: 2550035.",
    "R. Saini, B. K. Behera, S. Al-Kuwari, A. Farouk. NEQRX: Efficient Quantum Image Encryption with Reduced Circuit Complexity. arXiv:2204.07996, 2022.",
    "X. Liu, D. Xiao, Y. Xiang. Quantum image encryption using intra and inter bit permutation based on logistic map. IEEE Access, 2019, 7: 6937-6946.",
    "X. Chai, X. Fu, Z. Gan, Y. Lu, Y. Chen. A chaotic image encryption algorithm based on dynamic DNA encoding and chaos. Signal Processing, 2019, 155: 44-62.",
    "\u0141. Pawela. Temporal-order-controlled Parrondo reversal in a periodically switched quantum walk acting on NEQR image states. arXiv:2508.16382, 2026.",
    "N. Li, Z. Tang. A quantum reversible color-to-grayscale conversion scheme via image encryption based on true random numbers and two-dimensional quantum walks. Signal Processing, 2025, 233: 109949.",
    "J. Fridrich. Symmetric ciphers based on two-dimensional chaotic maps. International Journal of Bifurcation and Chaos, 1998, 8(6): 1259-1284.",
    "J. Preskill. Quantum computing in the NISQ era and beyond. Quantum, 2018, 2: 79.",
    "NIST. SP 800-22 Rev.1a: A Statistical Test Suite for Random and Pseudorandom Number Generators for Cryptographic Applications. National Institute of Standards and Technology, 2010.",
    "NIST. SP 800-90B: Recommendation for the Entropy Sources Used for Random Bit Generation. National Institute of Standards and Technology, 2018.",
    "M. Kumari, S. Gupta. Performance comparison between chaos and quantum-chaos based image encryption techniques. Multimedia Tools and Applications, 2021, 80: 33213-33255.",
    "F. Alblehai, A. A. Abd El-Latif, P. Pławiak, B. Abd-El-Atty. Cascading quantum walks with Chebyshev map for designing a robust medical image encryption algorithm. Scientific Reports, 2025, 15: 6685. DOI: 10.1038/s41598-025-90725-6.",
    "A. Kumar, O. P. Singh, S. Purkayastha. Quantum computing in image encryption: a comprehensive survey of methods, challenges, and future directions. Journal of Information Security and Applications, 2026, 99: 104452.",
    "S. Shrestha, D. Rai, N. B. Adhikari. Enhancing medical image security leveraging quantum key distribution. IET Quantum Communication, 2026, 7(1): e70040.",
    "Y. Wu, J. P. Noonan, S. Agaian. NPCR and UACI randomness tests for image encryption. Journal of Selected Areas in Telecommunications, 2011.",
    "G. Cui, X. Zhou, H. Wang, et al. Optical color image encryption algorithm based on two-dimensional quantum walking. Electronics, 2024, 13(11): 2026.",
    "H. Zhang, X. Feng, J. Sun, et al. Chaotic image security techniques and developments: a review. Mathematics, 2025, 13(12): 1976.",
    "B. L. Douglas, J. B. Wang. Efficient quantum circuit implementation of quantum walks. Physical Review A, 2009, 79: 052335.",
    "G. Chen, X. Song, S. E. Venegas-Andraca, et al. QIRHSI: novel quantum image representation based on HSI color space model. Quantum Information Processing, 2022, 21(5).",
]
for i, r in enumerate(refs, 1):
    p = doc.add_paragraph()
    run = p.add_run("[%d]  %s" % (i, r))
    _font(run, cn="Times New Roman", en="Times New Roman", size=9)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.left_indent = Pt(24)
    p.paragraph_format.first_line_indent = Pt(-24)

out = os.path.join(PAPERS, "NISQ_QuantumWalk_Image_Encryption_English_Final.docx")
doc.save(out)
print("saved:", out, "%.2f MB" % (os.path.getsize(out) / 1048576.0))
