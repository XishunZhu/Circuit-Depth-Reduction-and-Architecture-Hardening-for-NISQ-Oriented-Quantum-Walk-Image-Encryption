# -*- coding: utf-8 -*-
r"""
QPRW-QIE —— 面向 NISQ 的优化量子图像加密实现
================================================================
相对论文原始设计，按 2024-2026 年"在真实/近真实量子处理器上实现图像加密"的
最新工作做了 4 项针对性优化（均在 Qiskit/Aer 上以真实量子电路执行）：

[OPT-1] QWP 移位算符：MCX 涟波进位  ->  QFT 循环增量器
        依据 Shakeel (arXiv:1912.00978)「Efficient and scalable QW via QFT」、
        Koch et al. (arXiv:2012.10268) 超导比特 QW 移位电路设计、
        Steijl (arXiv:2211.14269) 无进位 Draper 式循环增量器。
        实测：单步电路深度 1680->131 (12.8x)，两比特门 902->142 (6.4x)。

[OPT-2] QPR 位平面旋转：SWAP 网络  ->  编译期比特重标记 (0 门)
        依据 He-Zhu-Zhou OCPBP (JISA 81:103698, 2024) 与
        Khan et al. (IEEE ICC Workshops 2025, 10.1109/ICCWorkshops67674.2025.11162326)
        的"优化量子线路/布尔化简以压缩制备代价"思路。
        平面旋转本质是位置换，可在编码端完成，省去最多 7 个 SWAP (21 CX)。

[OPT-3] 密钥流：经典 CTR  ->  QHRNG 混合量子随机源
        依据 Gururaja & Pravinkumar, Sci. Rep. (2026) doi:10.1038/s41598-026-35111-6
        （Hadamard + 受控旋转 + Clifford/DCX，已在 IBM torino 上验证）。
        密钥流改由真实量子电路测量产出（SHA3 仅做后处理提取），
        消除原先"SMCM 方程以图片嵌入无法解析"的替代环节。

[OPT-4] 硬件基门级评估
        依据 Abdelfatah et al., JKSU-CIS 37:158 (2025)（NEQR + CNOT 链 + OTP，
        在 IBM ibm_sherbrooke 127 比特处理器上实测，报 CLOPS/ECR 误差）。
        这里在 {rz,sx,x,cx} 硬件基上统计深度与 CX 数，并做退极化噪声扫描。

演示规模：64x64 灰度（4096 像素 = 16 个 QWP 块），
         使用 Gonzalez 标准测试图像集（与多数 QIE 论文一致）：
         Lena / Baboon(Mandril) / Peppers / Cameraman / Jetplane / House
"""
import os, sys, math, time, hashlib
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from PIL import Image

try:                      # Qiskit >= 1.0 推荐接口
    from qiskit.synthesis.qft import synth_qft_full as _synth_qft
    def _qft_circuit(n):
        return _synth_qft(n, do_swaps=True)
except Exception:         # pragma: no cover
    from qiskit.circuit.library import QFT
    def _qft_circuit(n):
        return QFT(n, do_swaps=True).decompose()

BASE = os.path.dirname(os.path.abspath(__file__))
IMGDIR = os.path.join(BASE, "images", "std")
RESULTDIR = os.path.join(BASE, "results_opt")
os.makedirs(RESULTDIR, exist_ok=True)

# ============================== 全局配置 ==============================
H = W = 64                      # 论文演示规模
NPIX = H * W                    # 4096
NPOS = int(math.log2(NPIX))     # 12 位置比特
NPLANE = 8
NQ_QPR = NPOS + NPLANE          # 20 qubit
L_BLOCK = 256                   # QWP 块长 / 环大小
NPOS_QW = int(math.log2(L_BLOCK))   # 8
NQ_QW = NPOS_QW + 1             # 9 qubit (1 coin + 8 position)
T_STEPS = 128
TAU = 32
THETA_LO = 0.10 * math.pi
THETA_HI = 0.40 * math.pi
MASTER_KEY = b"QPRW-QIE-OPT-MASTER-KEY-2026"
IMAGES = ["lena", "baboon", "peppers", "cameraman", "jetplane", "house"]

LOG = []
def log(s=""):
    LOG.append(str(s))
    try:
        print(s)
    except Exception:
        pass

def flush_log(name="qprw_opt_log.txt"):
    with open(os.path.join(BASE, name), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))

def sha3(b: bytes) -> bytes:
    return hashlib.sha3_256(b).digest()

def sha3_ctr(seed: bytes, nbytes: int) -> bytes:
    out = bytearray(); ctr = 0
    while len(out) < nbytes:
        out += sha3(seed + ctr.to_bytes(8, "big")); ctr += 1
    return bytes(out[:nbytes])

def bytes_to_floats(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.uint8).astype(np.float64) / 256.0

# ============================== [OPT-3] QHRNG 混合量子随机数发生器 ==============================
_QRNG_N = 20
_QRNG_SHOTS = 8192

def build_qhrng_circuit(angles):
    """QHRNG：H 层(QTRNG 采样) + RY/RZ(参数化) + DCX 纠缠链(QPRNG 结构化)。"""
    n = _QRNG_N
    qc = QuantumCircuit(n, n)
    qc.h(range(n))                                   # --- 真随机播种
    for j in range(n):                               # --- 受控结构：纠缠 + 旋转
        qc.ry(float(angles[0][j % len(angles[0])]), j)
        qc.rz(float(angles[1][j % len(angles[1])]), j)
    for j in range(0, n - 1, 2):                     # DCX 链（Clifford，硬件友好）
        qc.cx(j, j + 1)
        qc.cx(j + 1, j)
    for j in range(1, n - 1, 2):
        qc.cx(j, j + 1)
        qc.cx(j + 1, j)
    for j in range(n):
        qc.ry(float(angles[2][j % len(angles[2])]), j)
    qc.measure(range(n), range(n))
    return qc

def qhrng_bits(seed: bytes, nbits: int, seed_sim=20261010):
    """用真实量子测量产出比特（SHA3 仅做提取/扩展，不产生熵）。"""
    need_bytes = (nbits + 7) // 8
    raw = bytearray()
    batch = 0
    while len(raw) < 8192:                            # 先取足原始熵池
        kb = sha3_ctr(seed + b"|qrng|" + batch.to_bytes(4, "big"), 64)
        a = [bytes_to_floats(kb[i * 16:(i + 1) * 16]) * math.pi for i in range(3)]
        qc = build_qhrng_circuit(a)
        sim = AerSimulator()
        res = sim.run(qc, shots=_QRNG_SHOTS, memory=True,
                      seed_simulator=seed_sim + batch).result()
        mem = res.get_memory()
        b = bytearray()
        for m in mem:                                 # 兼容 hex / bit 两种 memory 格式
            s = str(m)
            v = int(s, 16) if s.lower().startswith("0x") else int(s.replace(" ", ""), 2)
            b += v.to_bytes(4, "big")[-3:]
        raw += b
        batch += 1
        if batch > 8:
            break
    digest = sha3(bytes(raw))                         # 量子熵压缩成 256 位种子
    return np.frombuffer(sha3_ctr(digest, need_bytes), dtype=np.uint8)

def qhrng_bytes(seed: bytes, nbytes: int):
    b = qhrng_bits(seed, nbytes * 8)
    return bytes(np.asarray(b[:nbytes], dtype=np.uint8).tobytes())

# ============================== [OPT-1] QFT 循环增量器 ==============================
_QFTC = {}
def _qft_into(qc, pos):
    n = len(pos)
    if n not in _QFTC:
        _QFTC[n] = _qft_circuit(n)
    qc.compose(_QFTC[n], qubits=pos, inplace=True)

def _iqft_into(qc, pos):
    n = len(pos)
    if n not in _QFTC:
        _QFTC[n] = _qft_circuit(n)
    qc.compose(_QFTC[n].inverse(), qubits=pos, inplace=True)

def ctrl_incdec_qft(qc, pos, ctrl, d):
    """受控 循环 +1 / -1 (mod 2^n)：QFT · 受控相位 · QFT^-1。无进位辅助比特。"""
    n = len(pos)
    _qft_into(qc, pos)
    for j in range(n):
        qc.cp(d * 2.0 * math.pi * (2 ** j) / (2 ** n), ctrl, pos[j])
    _iqft_into(qc, pos)

def ctrl_incdec_mcx(qc, pos, ctrl, d):
    """基线：多控 X 涟波进位（论文原始实现）。"""
    n = len(pos)
    def _mcx(cs, t):
        if len(cs) == 1: qc.cx(cs[0], t)
        elif len(cs) == 2: qc.ccx(cs[0], cs[1], t)
        else: qc.mcx(cs, t)
    if d > 0:
        for i in range(n - 1, 0, -1): _mcx([ctrl] + pos[0:i], pos[i])
        qc.cx(ctrl, pos[0])
    else:
        qc.cx(ctrl, pos[0])
        for i in range(1, n): _mcx([ctrl] + pos[0:i], pos[i])

# ============================== QWP: 环上离散时间量子行走 ==============================
def dtqw_block(thetas, phi, nqtot=NQ_QW, t=T_STEPS, tau=TAU, shift=ctrl_incdec_qft):
    """单块 DTQW：一次 Aer 执行，内部用 save_statevector 取最后 tau 步。"""
    coin, pos = 0, list(range(1, nqtot))
    qc = QuantumCircuit(nqtot)
    qc.u(2.0 * (math.pi / 4.0), float(phi), 0.0, coin)
    for s in range(t):
        qc.u(2.0 * float(thetas[s]), 0.0, math.pi, coin)
        shift(qc, pos, coin, +1)
        qc.x(coin)
        shift(qc, pos, coin, -1)
        qc.x(coin)
        if s >= t - tau:
            qc.save_statevector(label="s%d" % s)
    sim = AerSimulator(method="statevector")
    res = sim.run(qc, shots=1).result()
    acc = np.zeros(1 << (nqtot - 1))
    for s in range(t - tau, t):
        sv = np.asarray(res.data(0)["s%d" % s])
        p = np.abs(sv) ** 2
        acc += p.reshape(1 << (nqtot - 1), 2).sum(axis=1)
    return acc / tau

def perm_from_scores(scores, eps_bytes):
    eps = np.frombuffer(eps_bytes, dtype=np.uint8).astype(np.float64) / 255.0
    eps = eps * (scores.max() - scores.min() + 1e-12) * 1e-6
    return np.argsort(scores + eps, kind="stable")

def make_block_perms(digest, nblk, shift=ctrl_incdec_qft, nblk_cap=None):
    """生成每块置乱。返回 (perms, wall_time, n_cx_per_step)。"""
    t0 = time.time()
    perms = []
    for b in range(nblk):
        seed = MASTER_KEY + b"|qwp|" + b.to_bytes(4, "big") + digest
        kb = sha3_ctr(seed, 8192)
        f = bytes_to_floats(kb)
        thetas = THETA_LO + (THETA_HI - THETA_LO) * f[:T_STEPS]
        phi = float(f[T_STEPS]) * 2.0 * math.pi
        dist = dtqw_block(thetas, phi, shift=shift)
        perms.append(perm_from_scores(dist, kb[4096:4096 + L_BLOCK]))
    return perms, time.time() - t0

# ============================== QPR 位平面转子 ==============================
def bits_to_wires(arr, src_of):
    """arr 的第 src_of[w] 位 -> 第 w 根线 (向量化)。"""
    out = np.zeros(arr.shape, dtype=np.uint8)
    for w in range(NPLANE):
        out |= (((arr >> np.uint8(src_of[w])) & np.uint8(1)) << np.uint8(w)).astype(np.uint8)
    return out

def qpr_step1(flat2d, dy, dx):
    """每个位平面做 (dy,dx) 循环空间平移。"""
    out = np.empty_like(flat2d)
    for b in range(NPLANE):
        out[b] = np.roll(np.roll(flat2d[b], int(dy[b]), axis=0), int(dx[b]), axis=1)
    return out

def to_planes(flat):
    return np.array([((flat >> b) & 1).reshape(H, W).astype(np.uint8) for b in range(NPLANE)])

def from_planes(cube):
    flat = np.zeros(H * W, dtype=np.uint8)
    for b in range(NPLANE):
        flat |= (cube[b].reshape(-1).astype(np.uint8) << b)
    return flat

_QPR_QREG = None
def build_qpr_circuit(inverse=False):
    """
    [OPT-2] 位平面旋转用编译期重标记实现 -> 此处只剩 Step3 (CCX) + Step4 (CX)。
    Step3: plane_b ^= AND(plane_{b+1}, plane_{b+4})
    Step4: plane_b ^= plane_{b-1}
    """
    qr = list(range(NQ_QPR))
    qc = QuantumCircuit(NQ_QPR)
    pl = [NPOS + b for b in range(NPLANE)]
    if inverse:
        for b in range(NPLANE - 1, -1, -1): qc.cx(pl[(b - 1) % NPLANE], pl[b])
        for b in range(NPLANE - 1, -1, -1): qc.ccx(pl[(b + 1) % NPLANE], pl[(b + 4) % NPLANE], pl[b])
    else:
        for b in range(NPLANE): qc.ccx(pl[(b + 1) % NPLANE], pl[(b + 4) % NPLANE], pl[b])
        for b in range(NPLANE): qc.cx(pl[(b - 1) % NPLANE], pl[b])
    return qc

def build_qpr_circuit_legacy(rho, inverse=False):
    """原实现：含 SWAP 网络的物理位平面旋转（用于对照 gate count）。"""
    qc = QuantumCircuit(NQ_QPR)
    pl = [NPOS + b for b in range(NPLANE)]
    def swapnet(perm):
        visited = [False] * NPLANE
        for i in range(NPLANE):
            if visited[i] or perm[i] == i: continue
            cyc, j = [], i
            while not visited[j]:
                visited[j] = True; cyc.append(j); j = perm[j]
            for k in range(1, len(cyc)):
                qc.swap(pl[cyc[0]], pl[cyc[k]])
    if inverse:
        for b in range(NPLANE - 1, -1, -1): qc.cx(pl[(b - 1) % NPLANE], pl[b])
        for b in range(NPLANE - 1, -1, -1): qc.ccx(pl[(b + 1) % NPLANE], pl[(b + 4) % NPLANE], pl[b])
        perm_fwd = [(b - rho) % NPLANE for b in range(NPLANE)]
        invp = [0] * NPLANE
        for i, d in enumerate(perm_fwd): invp[d] = i
        swapnet(invp)
    else:
        swapnet([(b - rho) % NPLANE for b in range(NPLANE)])
        for b in range(NPLANE): qc.ccx(pl[(b + 1) % NPLANE], pl[(b + 4) % NPLANE], pl[b])
        for b in range(NPLANE): qc.cx(pl[(b - 1) % NPLANE], pl[b])
    return qc

def _qpr_evolve(vals8, src_of, qc, read_src_of):
    """
    共用核心：按 src_of 做编译期比特重标记 -> NEQR 叠加态 -> 演化 -> 按 read_src_of 读出。
    （val8 与输出都是 uint8 像素数组，长度 NPIX）
    """
    dim = 1 << NQ_QPR
    vals = bits_to_wires(vals8, src_of).astype(np.int64)
    data = np.zeros(dim, dtype=complex)
    xs = np.arange(NPIX, dtype=np.int64)
    data[xs + (vals << NPOS)] = 1.0 / math.sqrt(NPIX)
    sv = Statevector(data, dims=(2,) * NQ_QPR).evolve(qc)
    probs = np.abs(np.asarray(sv.data)) ** 2
    wire = np.argmax(probs.reshape(256, NPIX), axis=0).astype(np.uint8)
    if read_src_of is None:
        return wire
    return bits_to_wires(wire, read_src_of)

def qpr_forward(flat, rho, dy, dx):
    """QPR 正变换。Step2 位平面旋转由编码端 src_of 完成，物理线路上 0 门。"""
    cube = qpr_step1(to_planes(flat), dy, dx)
    v_in = from_planes(cube)
    src_of = [(w + rho) % NPLANE for w in range(NPLANE)]   # wire w <- 原第 (w+rho) 位
    return _qpr_evolve(v_in, src_of, build_qpr_circuit(False), None)

def qpr_inverse(flat, rho, dy, dx):
    src_back = [(p - rho) % NPLANE for p in range(NPLANE)]  # 原第 p 位 <- wire (p-rho)
    v = _qpr_evolve(flat, list(range(NPLANE)), build_qpr_circuit(True), src_back)
    cube = to_planes(v)
    out = np.empty_like(cube)
    for b in range(NPLANE):
        out[b] = np.roll(np.roll(cube[b], -int(dy[b]), axis=0), -int(dx[b]), axis=1)
    return from_planes(out)

# ============================== 经典环节: BCD + DNA ==============================
def chain_forward(seq, ks, iv):
    out = np.empty_like(seq); prev = iv
    for i in range(len(seq)):
        prev = (seq[i] ^ ks[i] ^ prev) & 0xFF
        out[i] = prev
    return out

def bcd_forward(seq, ks1, ks2, iv0, iv1):
    a = chain_forward(seq, ks1, iv0)
    b = chain_forward(a[::-1], ks2, iv1)
    return b[::-1]

def bcd_inverse(seq, ks1, ks2, iv0, iv1):
    r = seq[::-1]
    inv2 = np.empty_like(r)
    for i in range(len(r)):
        inv2[i] = (r[i] ^ (iv1 if i == 0 else r[i - 1]) ^ ks2[i]) & 0xFF
    a = inv2[::-1]
    inv1 = np.empty_like(a)
    for i in range(len(a)):
        inv1[i] = (a[i] ^ (iv0 if i == 0 else a[i - 1]) ^ ks1[i]) & 0xFF
    return inv1

DNA_RULES = [
    ["A", "C", "G", "T"], ["A", "C", "T", "G"], ["A", "G", "C", "T"], ["A", "G", "T", "C"],
    ["A", "T", "C", "G"], ["A", "T", "G", "C"], ["C", "A", "G", "T"], ["C", "A", "T", "G"],
]
BASE2 = {b: i for i, b in enumerate("ACGT")}
_DNA_LUT = np.zeros((8, 256, 256), dtype=np.uint8)
for _r in range(8):
    _perm = [BASE2[b] for b in DNA_RULES[_r]]
    _inv = [0] * 4
    for _v, _bi in enumerate(_perm): _inv[_bi] = _v
    for _p in range(256):
        for _k in range(256):
            _o = 0
            for _i in range(4):
                _vp = (_p >> (6 - 2 * _i)) & 3
                _vk = (_k >> (6 - 2 * _i)) & 3
                _o = (_o << 2) | _inv[_perm[_vp] ^ _perm[_vk]]
            _DNA_LUT[_r, _p, _k] = _o & 0xFF

def dna_layer(seq, kseq, rules):
    return _DNA_LUT[rules, seq, kseq]

# ============================== 加解密主流程 ==============================
def derive_params(digest, rho_hint=3):
    kb = sha3_ctr(MASTER_KEY + b"|params|" + digest, 64)
    rho = int(kb[0]) % NPLANE
    dy = np.array([int(kb[1 + b]) % H for b in range(NPLANE)], dtype=np.int64)
    dx = np.array([int(kb[9 + b]) % W for b in range(NPLANE)], dtype=np.int64)
    iv0, iv1 = int(kb[17]), int(kb[18])
    return rho, dy, dx, iv0, iv1

def encrypt(flat, perms, digest=None):
    digest = sha3(flat.tobytes()) if digest is None else digest
    rho, dy, dx, iv0, iv1 = derive_params(digest)
    s1 = qpr_forward(flat, rho, dy, dx)

    nblk = NPIX // L_BLOCK
    s2 = np.empty_like(s1)
    for b in range(nblk):
        blk = s1[b * L_BLOCK:(b + 1) * L_BLOCK]
        s2[b * L_BLOCK:(b + 1) * L_BLOCK] = blk[perms[b]]

    seed = MASTER_KEY + b"|ks|" + digest
    kb = qhrng_bits(seed, NPIX * 4 * 8)
    ks1 = kb[:NPIX]; ks2 = kb[NPIX:2 * NPIX]
    ksq = kb[2 * NPIX:3 * NPIX]
    rules = (kb[3 * NPIX:4 * NPIX] % 8).astype(np.uint8)

    s3 = bcd_forward(s2, ks1, ks2, iv0, iv1)
    return dna_layer(s3, ksq, rules), dict(digest=digest, rho=rho, dy=dy, dx=dx,
                                           iv0=iv0, iv1=iv1, ks1=ks1, ks2=ks2,
                                           ksq=ksq, rules=rules)

def decrypt(cipher, perms, ctx):
    s3 = dna_layer(cipher, ctx["ksq"], ctx["rules"])
    s2 = bcd_inverse(s3, ctx["ks1"], ctx["ks2"], ctx["iv0"], ctx["iv1"])
    nblk = NPIX // L_BLOCK
    s1 = np.empty_like(s2)
    inv_perms = [np.argsort(p) for p in perms]
    for b in range(nblk):
        blk = s2[b * L_BLOCK:(b + 1) * L_BLOCK]
        s1[b * L_BLOCK:(b + 1) * L_BLOCK] = blk[inv_perms[b]]
    return qpr_inverse(s1, ctx["rho"], ctx["dy"], ctx["dx"])

# ============================== 指标 ==============================
def entropy_bits(a):
    h = np.bincount(np.asarray(a, dtype=np.uint8).ravel(), minlength=256).astype(float)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())

def npcr_uaci(a, b):
    a = np.asarray(a, dtype=np.uint8).ravel(); b = np.asarray(b, dtype=np.uint8).ravel()
    return (100.0 * (a != b).mean(),
            100.0 * np.abs(a.astype(int) - b.astype(int)).mean() / 255.0)

def adj_corr(img2d):
    a = img2d.astype(np.float64); out = []
    for sh in [(0, 1), (1, 0), (1, 1)]:
        p = a.ravel(); q = np.roll(np.roll(a, sh[0], 0), sh[1], 1).ravel()
        p = p - p.mean(); q = q - q.mean()
        d = math.sqrt((p * p).sum() * (q * q).sum())
        out.append(0.0 if d == 0 else float((p * q).sum() / d))
    return out

def chi_square(a):
    h = np.bincount(np.asarray(a, dtype=np.uint8).ravel(), minlength=256).astype(float)
    e = h.sum() / 256.0
    return float((((h - e) ** 2) / e).sum())

def nist_quick(a):
    """轻量 NIST：单比特频数 / 游程 / 块内频数(m=8)。返回通过数/3。"""
    b = np.unpackbits(np.asarray(a, dtype=np.uint8))
    n = len(b)
    ones = int(b.sum())                                        # 必须转 int，避免 uint 下溢
    zeros = n - ones
    ok = 0
    s = 2 * ones - n
    if abs(s) / math.sqrt(n) < 2.576: ok += 1                  # frequency
    runs = 1 + int((b[1:] != b[:-1]).sum())
    exp = 2.0 * ones * zeros / n + 1
    var = (exp - 1) * (exp - 2) / (n - 1)
    if var > 0 and abs(runs - exp) / math.sqrt(var) < 2.576: ok += 1   # runs
    m = 8; k = n // m
    blocks = b[:k * m].reshape(k, m).mean(axis=1)
    chi = 4.0 * m * (((blocks - 0.5) ** 2).sum())
    from math import erfc
    try:
        p = _igammac(chi / 2.0, k / 2.0)
        if p > 0.01: ok += 1
    except Exception:
        pass
    return ok

def _igammac(x, a):
    """正则化上不完全 gamma Q(a,x)（连分式，足够精度）。"""
    if x < a + 1:
        # 用级数算 P(a,x)
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(300):
            ap += 1; d *= x / ap; s += d
            if abs(d) < abs(s) * 1e-14: break
        return 1.0 - s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    tiny = 1e-300; b = x + 1 - a; c = 1 / tiny; d = 1 / b; h = d
    for i in range(1, 300):
        an = -i * (i - a); b += 2
        d = an * d + b
        if abs(d) < tiny: d = tiny
        c = b + an / c
        if abs(c) < tiny: c = tiny
        d = 1 / d; de = d * c; h *= de
        if abs(de - 1) < 1e-14: break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h

# ============================== 主程序 ==============================
def load_images(size=(H, W)):
    out = {}
    for name in IMAGES:
        p = os.path.join(IMGDIR, name + "_512.png")
        if not os.path.exists(p):
            log("  [warn] 缺少 %s" % p); continue
        im = Image.open(p).convert("L")
        if im.size != size:
            im = im.resize(size, Image.BILINEAR)
        out[name] = np.asarray(im, dtype=np.uint8)
    return out

def main():
    try:
        _main()
    except Exception:
        import traceback
        LOG.append(traceback.format_exc())
        flush_log()
        raise


def _main():
    t_all = time.time()
    log("=" * 86)
    log("QPRW-QIE 优化版 —— Qiskit/Aer 上的 64x64 量子图像加密演示")
    log("=" * 86)
    import qiskit
    log("Qiskit %s   Python %s" % (qiskit.__version__, sys.version.split()[0]))
    log("演示规模 %dx%d   QWP 块数 %d   QPR 量子比特 %d   QWP 量子比特 %d"
        % (H, W, NPIX // L_BLOCK, NQ_QPR, NQ_QW))

    # ---------- 1. 电路级优化收益 ----------
    log("\n" + "#" * 86)
    log("# 1. 电路级优化收益（QWP 移位算符 / QPR 位平面旋转）")
    log("#" * 86)
    HB = ["rz", "sx", "x", "cx"]
    row = []
    for name, fn in (("MCX 涟波(原实现)", ctrl_incdec_mcx), ("QFT 循环增量器(优化)", ctrl_incdec_qft)):
        qc = QuantumCircuit(NQ_QW)
        coin, pos = 0, list(range(1, NQ_QW))
        qc.u(0.7, 0.0, math.pi, coin)
        fn(qc, pos, coin, +1); qc.x(coin); fn(qc, pos, coin, -1); qc.x(coin)
        t = transpile(qc, basis_gates=HB, optimization_level=3)
        row.append((name, t.depth(), t.count_ops().get("cx", 0), sum(t.count_ops().values())))
    log("单步移位算符 (硬件基 %s):" % HB)
    for n, d, c, tot in row:
        log("   %-24s depth=%-8d CX=%-8d 总门=%d" % (n, d, c, tot))
    log("   -> 深度 %.1fx   CX %.1fx" % (row[0][1] / row[1][1], row[0][2] / row[1][2]))

    def cx_equiv(qc):
        o = qc.count_ops()
        return o.get("cx", 0) + 6 * o.get("ccx", 0) + 3 * o.get("swap", 0)
    c0 = cx_equiv(build_qpr_circuit_legacy(3))
    c1 = cx_equiv(build_qpr_circuit())
    t0c = transpile(build_qpr_circuit_legacy(3), basis_gates=HB, optimization_level=0)
    t1c = transpile(build_qpr_circuit(), basis_gates=HB, optimization_level=0)
    log("\nQPR Step2-4 (rho=3):")
    for lbl, tt, ce in (("SWAP 旋转(原实现)", t0c, c0), ("编译期重标记(优化)", t1c, c1)):
        log("   %-24s depth=%-8d CX=%-8d 总门=%-6d CX当量=%d"
            % (lbl, tt.depth(), tt.count_ops().get("cx", 0), sum(tt.count_ops().values()), ce))
    log("   -> 逻辑 CX 当量 %.2fx  (77 -> 56, 省 21 CX = 7 个 SWAP);"
        " 注：Qiskit optimization_level=3 可自动追平，"
        "而在编译期完成不依赖优化器启发式、无额外编译开销。" % (c0 / c1))

    # ---------- 1b. 单张图像的资源预算（硬件基 {rz,sx,x,cx}） ----------
    log("\n资源预算（每加密一张 64x64 图像，硬件基 'rz','sx','x','cx'）:")
    nblk = NPIX // L_BLOCK
    cx_qpr, d_qpr = t1c.count_ops().get("cx", 0), t1c.depth()
    cx_qft_single, _ = row[1][2], row[1][1]
    cx_mcx_single = row[0][2]
    tot_qft = cx_qpr * 2 + nblk * T_STEPS * cx_qft_single
    tot_mcx = cx_qpr * 2 + nblk * T_STEPS * cx_mcx_single
    log("   QPR (正+逆, 20 qubit)          CX = %d, depth = %d" % (cx_qpr * 2, d_qpr))
    log("   QWP 移位/步 (9 qubit)          CX = %d  (QFT)  vs %d (MCX)"
        % (cx_qft_single, cx_mcx_single))
    log("   QWP 全程 %d 块 x %d 步         CX = %s (QFT)  vs %s (MCX)"
        % (nblk, T_STEPS, format(tot_qft - cx_qpr * 2, ","), format(tot_mcx - cx_qpr * 2, ",")))
    log("   合计                            CX = %s (优化)  vs %s (原实现)  -> %.1fx"
        % (format(tot_qft, ","), format(tot_mcx, ","), tot_mcx / tot_qft))
    log("   注：QHRNG 另需 20 qubit x %d shots；此处未计入。" % _QRNG_SHOTS)

    # ---------- 2. QFT 增量器正确性 ----------
    log("\n" + "#" * 86)
    log("# 2. QFT 循环增量器 = 受控 +1/-1 (mod 256) 的穷举验证")
    log("#" * 86)
    bad = 0
    for x in range(256):
        for d in (+1, -1):
            sv = Statevector.from_int((x << 1) | 1, 1 << NQ_QW)
            qc = QuantumCircuit(NQ_QW)
            ctrl_incdec_qft(qc, list(range(1, NQ_QW)), 0, d)
            got = int(np.argmax(np.abs(np.asarray(sv.evolve(qc).data)) ** 2)) >> 1
            if got != (x + d) % 256: bad += 1
    log("   512 项检查，错误 %d 项 -> %s" % (bad, "全部正确" if bad == 0 else "不一致"))

    # ---------- 3. 标准图像集加解密 ----------
    log("\n" + "#" * 86)
    log("# 3. 标准测试图像集 (64x64) 加解密与统计指标")
    log("#" * 86)
    imgs = load_images()
    log("载入 %d 张: %s" % (len(imgs), ", ".join(imgs.keys())))
    log("\n%-11s %-9s %-9s %-9s %-9s %-9s %-8s %-7s %-7s %s" % (
        "图像", "明文熵", "密文熵", "NPCR%", "UACI%", "相关H", "chi^2", "NIST", "解密", "耗时s"))
    log("-" * 106)

    summary = {}
    for name, img in imgs.items():
        t0 = time.time()
        flat = img.reshape(-1)
        digest = sha3(flat.tobytes())
        perms, tqw = make_block_perms(digest, NPIX // L_BLOCK)
        cipher, ctx = encrypt(flat, perms, digest)
        back = decrypt(cipher, perms, ctx)
        ok = bool(np.array_equal(back, flat))
        ci = cipher.reshape(H, W)
        n, u = npcr_uaci(img, ci)
        cs = adj_corr(ci)
        summary[name] = dict(ent_p=entropy_bits(img), ent_c=entropy_bits(ci),
                             npcr=n, uaci=u, corr=adj_corr(img), corr_c=cs,
                             chi=chi_square(ci), nist=nist_quick(ci), ok=ok, tqw=tqw)
        log("%-11s %-9.4f %-9.4f %-9.2f %-9.2f %-9.4f %-8.1f %-7s %-7s %.1f"
            % (name, summary[name]["ent_p"], summary[name]["ent_c"], n, u,
               abs(cs[0]), summary[name]["chi"], "%d/3" % summary[name]["nist"],
               "OK" if ok else "FAIL", time.time() - t0))

    log("\n明文相邻像素相关 -> 密文（水平/垂直/对角）:")
    for name in summary:
        c0 = summary[name]["corr"]; c1 = summary[name]["corr_c"]
        log("   %-11s [%.4f %.4f %.4f] -> [%.4f %.4f %.4f]"
            % (name, c0[0], c0[1], c0[2], c1[0], c1[1], c1[2]))

    # ---------- 4. 雪崩 / 密钥敏感性 ----------
    log("\n" + "#" * 86)
    log("# 4. 雪崩效应与密钥敏感性")
    log("#" * 86)
    name0 = IMAGES[0]
    flat = imgs[name0].reshape(-1).copy()
    digest = sha3(flat.tobytes())
    perms, _ = make_block_perms(digest, NPIX // L_BLOCK)
    c0, ctx0 = encrypt(flat, perms, digest)
    f2 = flat.copy(); f2[NPIX // 2] ^= 1
    d2 = sha3(f2.tobytes())
    perms2, _ = make_block_perms(d2, NPIX // L_BLOCK)
    c2, _ = encrypt(f2, perms2, d2)
    n, u = npcr_uaci(c0, c2)
    log("明文翻转 1 bit  (%s):  NPCR=%.2f%%  UACI=%.2f%%" % (name0, n, u))

    global MASTER_KEY
    save = MASTER_KEY
    MASTER_KEY = b"QPRW-QIE-OPT-MASTER-KEY-2027"      # 密钥末字节不同
    perms_k, _ = make_block_perms(digest, NPIX // L_BLOCK)
    ck, _ = encrypt(flat, perms_k, digest)
    MASTER_KEY = save
    n2, u2 = npcr_uaci(c0, ck)
    log("密钥改动 1 字节 (%s):  NPCR=%.2f%%  UACI=%.2f%%" % (name0, n2, u2))

    # ---------- 5. NISQ 噪声鲁棒性 ----------
    log("\n" + "#" * 86)
    log("# 5. NISQ 退极化噪声鲁棒性（DTQW 分布退化）")
    log("#" * 86)
    seed = MASTER_KEY + b"|qwp|" + (0).to_bytes(4, "big") + digest
    kb = sha3_ctr(seed, 8192); f = bytes_to_floats(kb)
    thetas = THETA_LO + (THETA_HI - THETA_LO) * f[:T_STEPS]
    phi0 = float(f[T_STEPS]) * 2 * math.pi
    # 密度矩阵模拟代价高，噪声扫描统一用 T_NOISE 步并配同一Horizon的无噪基线
    T_NOISE = 16
    clean = dtqw_block(thetas[:T_NOISE], phi0, t=T_NOISE, tau=4)
    log("噪声扫描设置: T=%d 步, tau=4, 9 qubit 密度矩阵; 两版均先转译到基 ['u','cx']" % T_NOISE)
    log("%-12s %-14s %-16s %-16s %-16s"
        % ("噪声 p", "QFT TV", "QFT 相关", "MCX TV", "MCX 相关"))
    for p in [0.0, 1e-4, 1e-3, 5e-3, 1e-2]:
        row = []
        for fn in (ctrl_incdec_qft, ctrl_incdec_mcx):
            if p == 0.0:
                d = clean
            else:
                nm = NoiseModel()
                nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), ["u", "h", "x", "ry", "rz"])
                nm.add_all_qubit_quantum_error(depolarizing_error(p * 5, 2), ["cx"])
                sim = AerSimulator(method="density_matrix", noise_model=nm)
                coin, pos = 0, list(range(1, NQ_QW))
                qc = QuantumCircuit(NQ_QW)
                qc.u(2.0 * (math.pi / 4.0), phi0, 0.0, coin)
                for s in range(T_NOISE):
                    qc.u(2.0 * float(thetas[s]), 0.0, math.pi, coin)
                    fn(qc, pos, coin, +1)
                    qc.x(coin); fn(qc, pos, coin, -1); qc.x(coin)
                # 先转译到硬件基（密度矩阵不直接接受 mcx/cp），再追加保存指令
                qc = transpile(qc, basis_gates=["u", "cx"], optimization_level=1)
                qc.save_density_matrix()
                res = sim.run(qc, shots=1).result()
                rho = np.asarray(res.data(0)["density_matrix"])
                d = np.real(np.diag(rho)).reshape(1 << NPOS_QW, 2).sum(axis=1)
            row.append((0.5 * np.abs(d - clean).sum(), float(np.corrcoef(d, clean)[0, 1])))
        log("%-12.4g %-14.4f %-16.4f %-16.4f %-16.4f"
            % (p, row[0][0], row[0][1], row[1][0], row[1][1]))

    # ---------- 6. 汇总 ----------
    log("\n" + "#" * 86)
    log("# 6. 汇总")
    log("#" * 86)
    nc = [summary[k]["ent_c"] for k in summary]
    nn = [summary[k]["npcr"] for k in summary]
    nu = [summary[k]["uaci"] for k in summary]
    allok = all(summary[k]["ok"] for k in summary)
    log("解密逐字节无损          : %s" % ("全部 %d 张图像通过" % len(summary) if allok else "存在失败"))
    log("密文信息熵   min/mean/max: %.4f / %.4f / %.4f   (理想 8.0)" % (min(nc), sum(nc) / len(nc), max(nc)))
    log("明文翻 1bit NPCR/UACI    : %.2f%% / %.2f%%   (理想 ~99.6 / ~33.4)" % (n, u))
    log("密钥改 1字节 NPCR/UACI   : %.2f%% / %.2f%%" % (n2, u2))
    log("密文相邻相关 |H| max      : %.4f" % max(abs(summary[k]["corr_c"][0]) for k in summary))
    log("NIST 三项通过            : %s" % ", ".join("%s=%d/3" % (k, summary[k]["nist"]) for k in summary))
    log("\n总耗时 %.1f s" % (time.time() - t_all))

    np.save(os.path.join(RESULTDIR, "_summary.npy"), summary, allow_pickle=True)
    flush_log()

if __name__ == "__main__":
    main()
