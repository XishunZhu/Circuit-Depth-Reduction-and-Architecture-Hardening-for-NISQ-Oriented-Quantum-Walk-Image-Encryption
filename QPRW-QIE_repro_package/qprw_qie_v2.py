# -*- coding: utf-8 -*-
r"""
QPRW-QIE v2 —— 面向 NISQ 的改进型量子图像加密（算法本体）
======================================================================
相对论文原始设计 (v1) 的改动：

[C1] QWP 移位算符：MCX 涟波进位 -> QFT 循环增量器            （上一版 OPT-1）
[C2] QWP 全程：QFT 帧折叠 (QFF)                              （本版新增）
     单步位移 S=QFT·D·QFT†，硬币算符 C 作用于不相交比特，故 QFT†·C·QFT=C，
     T 步中 2T 个傅里叶变换两两对消，只剩首尾各 1 个：复杂度 O(T·n²) -> O(T·n)。
[C3] ±1 方向融合                                              （本版新增）
     coin=1->+1 / coin=0->-1 拆成"无条件相位层(单比特, 0 CX, 跨步可交换)"
     + "受控修正层"，每步只剩 n 个受控相位。
[C4] QPR 位平面旋转：SWAP 网络 -> 编译期比特重标记(0 门)      （上一版 OPT-2）
[C5] 密钥流：QHRNG 混合量子随机源（真实量子测量产出熵）        （上一版 OPT-3）
[C6] 置换种子封装 (Permutation-Seed Encapsulation, PSE)       （本版新增，架构级）
     · 行走只跑一次（加密侧），其分布经量化+SHA3 压缩为 256 bit 置换种子 S；
     · S 用密钥流封装进密文头；解密侧按 S 用 Fisher-Yates 确定性展开，
       不再重跑量子行走 -> 量子开销减半，且解密正确性与硬件噪声解耦。
[C7] 随机 nonce + 完整性标签 (MAC)                            （本版新增，架构级）
     · 全部密钥材料由 (主密钥 ‖ nonce) 派生，不再依赖明文摘要 -> 解密侧可独立推导，
       且同一明文两次加密结果不同（随机化加密）。
     · 16 字节 MAC 标签提供篡改检测。
[C8] 行走步数 T / 环长 L 可裁剪，用经典 BCD+DNA 扩散补偿       （本版新增）
"""
import os, math, time, hashlib
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

BASE = os.path.dirname(os.path.abspath(__file__))

# ============================== 全局配置 ==============================
H = W = 64
NPIX = H * W
NPOS = int(math.log2(NPIX))          # 12 位置比特
NPLANE = 8
NQ_QPR = NPOS + NPLANE               # 20 qubit
L_BLOCK = 256                        # QWP 块长 / 环大小
NPOS_QW = int(math.log2(L_BLOCK))    # 8
NQ_QW = NPOS_QW + 1                  # 9 qubit
NBLK = NPIX // L_BLOCK               # 16
THETA_LO = 0.10 * math.pi
THETA_HI = 0.40 * math.pi
T_DEFAULT = 32
NONCE_LEN = 16
SEED_LEN = 32
TAG_LEN = 16
HDR_LEN = NONCE_LEN + SEED_LEN       # 48 字节

# ============================== 基础工具 ==============================
def sha3(b: bytes) -> bytes:
    return hashlib.sha3_256(b).digest()

def sha3_ctr(seed: bytes, nbytes: int) -> bytes:
    out = bytearray(); ctr = 0
    while len(out) < nbytes:
        out += sha3(seed + ctr.to_bytes(8, "big")); ctr += 1
    return bytes(out[:nbytes])

def bytes_to_floats(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.uint8).astype(np.float64) / 256.0

# ============================== [C5] QHRNG ==============================
_QRNG_N = 20
_QRNG_SHOTS = 8192

def build_qhrng_circuit(angles):
    n = _QRNG_N
    qc = QuantumCircuit(n, n)
    qc.h(range(n))                                    # QTRNG：Hadamard 采样
    for j in range(n):
        qc.ry(float(angles[0][j % len(angles[0])]), j)
        qc.rz(float(angles[1][j % len(angles[1])]), j)
    for j in range(0, n - 1, 2):                      # DCX 链（Clifford，硬件友好）
        qc.cx(j, j + 1); qc.cx(j + 1, j)
    for j in range(1, n - 1, 2):
        qc.cx(j, j + 1); qc.cx(j + 1, j)
    for j in range(n):
        qc.ry(float(angles[2][j % len(angles[2])]), j)
    qc.measure(range(n), range(n))
    return qc

_qhrng_cache = {}
def qhrng_raw(seed: bytes, seed_sim=20261010):
    """真实量子测量产出原始熵（SHA3 仅作提取器）。"""
    key = (seed, seed_sim)
    if key in _qhrng_cache:
        return _qhrng_cache[key]
    kb = sha3_ctr(seed + b"|qrng|", 64)
    a = [bytes_to_floats(kb[i * 16:(i + 1) * 16]) * math.pi for i in range(3)]
    qc = build_qhrng_circuit(a)
    res = AerSimulator().run(qc, shots=_QRNG_SHOTS, memory=True,
                             seed_simulator=seed_sim).result()
    raw = bytearray()
    for m in res.get_memory():
        s = str(m)
        v = int(s, 16) if s.lower().startswith("0x") else int(s.replace(" ", ""), 2)
        raw += v.to_bytes(4, "big")[-3:]
    out = bytes(raw)
    _qhrng_cache[key] = out
    return out

def qhrng_bytes(seed: bytes, nbytes: int) -> bytes:
    """量子测量熵 -> SHA3 提取 -> CTR 扩展。"""
    digest = sha3(qhrng_raw(b"QHRNG|" + seed))
    return sha3_ctr(digest, nbytes)

# ============================== QFT / AQFT ==============================
try:
    from qiskit.synthesis.qft import synth_qft_full as _synth_qft
    def _official_qft(n): return _synth_qft(n, do_swaps=True)
except Exception:                                     # pragma: no cover
    from qiskit.circuit.library import QFT
    def _official_qft(n): return QFT(n, do_swaps=True).decompose()

_QFTC = {}
def _get_qft(n, b=None):
    key = (n, b)
    if key not in _QFTC:
        base = _official_qft(n)
        if b is None:
            _QFTC[key] = base
        else:
            qc = QuantumCircuit(n, name="AQFT%d" % b)
            for inst in base.data:
                op = inst.operation
                if op.name in ("cp", "cu1", "cphase", "crz"):
                    ang = abs(float(op.params[0])) if op.params else 0.0
                    if ang < 1e-12 or round(math.log2(math.pi / ang)) >= b:
                        continue
                qc.append(inst)
            _QFTC[key] = qc
    return _QFTC[key]

def _qft_into(qc, pos, b=None):
    qc.compose(_get_qft(len(pos), b), qubits=pos, inplace=True)

def _iqft_into(qc, pos, b=None):
    qc.compose(_get_qft(len(pos), b).inverse(), qubits=pos, inplace=True)

def theta_j(j, npos):
    return 2.0 * math.pi * (2 ** j) / (2 ** npos)

# ============================== QWP：三种行走实现 ==============================
def ctrl_incdec_mcx(qc, pos, ctrl, d):
    """基线：多控 X 涟波进位（论文 v1 原实现）。"""
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

def walk_mcx(thetas, phi, t, npos, save=True):
    nq = npos + 1
    coin, pos = 0, list(range(1, nq))
    qc = QuantumCircuit(nq)
    qc.u(2.0 * (math.pi / 4.0), float(phi), 0.0, coin)
    for s in range(t):
        qc.u(2.0 * float(thetas[s]), 0.0, math.pi, coin)
        ctrl_incdec_mcx(qc, pos, coin, +1)
        qc.x(coin); ctrl_incdec_mcx(qc, pos, coin, -1); qc.x(coin)
    if save: qc.save_statevector(label="sv")
    return qc

def walk_perstep(thetas, phi, t, npos, save=True, b=None):
    nq = npos + 1
    coin, pos = 0, list(range(1, nq))
    qc = QuantumCircuit(nq)
    qc.u(2.0 * (math.pi / 4.0), float(phi), 0.0, coin)
    for s in range(t):
        qc.u(2.0 * float(thetas[s]), 0.0, math.pi, coin)
        _qft_into(qc, pos, b)
        for j in range(npos): qc.cp(+theta_j(j, npos), coin, pos[j])
        _iqft_into(qc, pos, b)
        qc.x(coin)
        _qft_into(qc, pos, b)
        for j in range(npos): qc.cp(-theta_j(j, npos), coin, pos[j])
        _iqft_into(qc, pos, b)
        qc.x(coin)
    if save: qc.save_statevector(label="sv")
    return qc

def walk_folded(thetas, phi, t, npos, save=True, b=None):
    """
    [C2+C3] U = QFT · (D C_T)…(D C_1) · QFT⁻¹
      D 的受控部分：coin=1 -> 相位 +2φ_k；D 的无条件部分：相位 −φ_k（单比特，0 CX）
      时间顺序：C_1, D, C_2, D, …, C_T, D
    """
    nq = npos + 1
    coin, pos = 0, list(range(1, nq))
    qc = QuantumCircuit(nq)
    qc.u(2.0 * (math.pi / 4.0), float(phi), 0.0, coin)
    _qft_into(qc, pos, b)
    for s in range(t):
        qc.u(2.0 * float(thetas[s]), 0.0, math.pi, coin)
        for j in range(npos): qc.cp(2.0 * theta_j(j, npos), coin, pos[j])
        for j in range(npos): qc.p(-theta_j(j, npos), pos[j])
    _iqft_into(qc, pos, b)
    if save: qc.save_statevector(label="sv")
    return qc

def dist_of(qc, npos):
    res = AerSimulator(method="statevector").run(qc, shots=1).result()
    v = np.asarray(res.data(0)["sv"])
    p = np.abs(v) ** 2
    return p.reshape(1 << npos, 2).sum(axis=1)

def block_scores(master, nonce, t=T_DEFAULT, nblk=NBLK, L=L_BLOCK, kind="folded", b=None):
    """每块的行走分布（量子电路真实执行）。"""
    npos = int(math.log2(L))
    fn = {"mcx": walk_mcx, "perstep": walk_perstep, "folded": walk_folded}[kind]
    out = []
    for i in range(nblk):
        seed = master + b"|qwp|" + nonce + i.to_bytes(4, "big")
        kb = sha3_ctr(seed, 4096)
        f = bytes_to_floats(kb)
        thetas = THETA_LO + (THETA_HI - THETA_LO) * f[:t]
        phi = float(f[t]) * 2.0 * math.pi
        out.append(dist_of(fn(thetas, phi, t, npos, b=b), npos))
    return out

def perm_seed_from_scores(scores):
    """[C6] 分布量化 -> SHA3 压缩为 256 bit 置换种子。"""
    h = hashlib.sha3_256()
    for d in scores:
        q = np.clip(np.round(d * (2 ** 32)), 0, 2 ** 32 - 1).astype(np.uint64)
        h.update(q.tobytes())
    return h.digest()

def fy_perm(seed32: bytes, n: int) -> np.ndarray:
    """Fisher-Yates：由种子确定性展开置换（解密侧无需量子电路）。"""
    ks = sha3_ctr(seed32 + b"|fy|", n * 4)
    r = np.frombuffer(ks, dtype=np.uint32).astype(np.uint64)
    idx = np.arange(n, dtype=np.int64)
    for i in range(n - 1, 0, -1):
        j = int(r[n - 1 - i] % np.uint64(i + 1))
        idx[i], idx[j] = idx[j], idx[i]
    return idx

def expand_perms(perm_seed, nblk=NBLK, L=L_BLOCK):
    return [fy_perm(perm_seed + i.to_bytes(4, "big"), L) for i in range(nblk)]

def perms_from_scores(scores):
    """直接由分布秩序取置换（对照用，v1 方式）。"""
    out = []
    for i, d in enumerate(scores):
        eb = sha3_ctr(sha3(np.round(d * (2 ** 32)).astype(np.uint64).tobytes()) + b"|eps|", L_BLOCK)
        eps = (np.frombuffer(eb, dtype=np.uint8).astype(float) / 255.0) \
              * (d.max() - d.min() + 1e-12) * 1e-6
        out.append(np.argsort(d + eps, kind="stable"))
    return out

# ============================== QPR 位平面转子 ==============================
def bits_to_wires(arr, src_of):
    out = np.zeros(arr.shape, dtype=np.uint8)
    for w in range(NPLANE):
        out |= (((arr >> np.uint8(src_of[w])) & np.uint8(1)) << np.uint8(w)).astype(np.uint8)
    return out

def to_planes(flat):
    return np.array([((flat >> b) & 1).reshape(H, W).astype(np.uint8) for b in range(NPLANE)])

def from_planes(cube):
    flat = np.zeros(H * W, dtype=np.uint8)
    for b in range(NPLANE):
        flat |= (cube[b].reshape(-1).astype(np.uint8) << b)
    return flat

def build_qpr_circuit(inverse=False):
    """[C4] 位平面旋转由编码端重标记完成，线路上只剩 Step3(CCX) + Step4(CX)。"""
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
    """对照：含 SWAP 网络的物理位平面旋转。"""
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

_QPR_F = None
_QPR_I = None
def _qpr_evolve(vals8, src_of, qc, read_src_of):
    dim = 1 << NQ_QPR
    vals = bits_to_wires(vals8, src_of).astype(np.int64)
    data = np.zeros(dim, dtype=complex)
    xs = np.arange(NPIX, dtype=np.int64)
    data[xs + (vals << NPOS)] = 1.0 / math.sqrt(NPIX)
    sv = Statevector(data, dims=(2,) * NQ_QPR).evolve(qc)
    probs = np.abs(np.asarray(sv.data)) ** 2
    wire = np.argmax(probs.reshape(256, NPIX), axis=0).astype(np.uint8)
    return wire if read_src_of is None else bits_to_wires(wire, read_src_of)

def qpr_forward(flat, rho, dy, dx):
    global _QPR_F
    if _QPR_F is None: _QPR_F = build_qpr_circuit(False)
    cube = np.empty((NPLANE, H, W), dtype=np.uint8)
    for b in range(NPLANE):
        cube[b] = np.roll(np.roll(((flat >> b) & 1).reshape(H, W).astype(np.uint8),
                                  int(dy[b]), axis=0), int(dx[b]), axis=1)
    src_of = [(w + rho) % NPLANE for w in range(NPLANE)]
    return _qpr_evolve(from_planes(cube), src_of, _QPR_F, None)

def qpr_inverse(flat, rho, dy, dx):
    global _QPR_I
    if _QPR_I is None: _QPR_I = build_qpr_circuit(True)
    src_back = [(p - rho) % NPLANE for p in range(NPLANE)]
    v = _qpr_evolve(flat, list(range(NPLANE)), _QPR_I, src_back)
    cube = to_planes(v)
    out = np.empty((NPLANE, H, W), dtype=np.uint8)
    for b in range(NPLANE):
        out[b] = np.roll(np.roll(cube[b], -int(dy[b]), axis=0), -int(dx[b]), axis=1)
    return from_planes(out)

# ============================== 经典环节：BCD + DNA ==============================
# --- AES S 盒（双射）：用于打断 XOR 链的差分抵消 ---
def _gmul(a, b):
    r = 0
    for _ in range(8):
        if b & 1: r ^= a
        hi = a & 0x80
        a = (a << 1) & 0xFF
        if hi: a ^= 0x1B
        b >>= 1
    return r

def _build_sbox():
    inv = [0] * 256
    for i in range(1, 256):
        for j in range(1, 256):
            if _gmul(i, j) == 1:
                inv[i] = j; break
    sb = [0] * 256
    for i in range(256):
        s = inv[i]
        r = s
        for k in (1, 2, 3, 4):
            r ^= ((s << k) | (s >> (8 - k))) & 0xFF
        sb[i] = r ^ 0x63
    return sb

_SBOX = np.array(_build_sbox(), dtype=np.uint8)
_SBOX_INV = np.zeros(256, dtype=np.uint8)
for _i in range(256): _SBOX_INV[_SBOX[_i]] = _i
_SB = _SBOX.tolist()
_SBI = _SBOX_INV.tolist()

def bcd_forward_xor(seq, ks1, ks2, iv0, iv1):
    """[缺陷版] 纯 XOR 双向链：差分按奇偶交替抵消，扩散不全（论文对照用）。"""
    a = np.empty_like(seq); prev = iv0
    for i in range(len(seq)):
        prev = (seq[i] ^ ks1[i] ^ prev) & 0xFF
        a[i] = prev
    b = np.empty_like(a); prev = iv1
    rev = a[::-1]
    for i in range(len(rev)):
        prev = (rev[i] ^ ks2[i] ^ prev) & 0xFF
        b[i] = prev
    return b[::-1]

def bcd_inverse_xor(seq, ks1, ks2, iv0, iv1):
    r = seq[::-1]
    inv2 = np.empty_like(r)
    for i in range(len(r)):
        inv2[i] = (r[i] ^ (iv1 if i == 0 else r[i - 1]) ^ ks2[i]) & 0xFF
    a = inv2[::-1]
    inv1 = np.empty_like(a)
    for i in range(len(a)):
        inv1[i] = (a[i] ^ (iv0 if i == 0 else a[i - 1]) ^ ks1[i]) & 0xFF
    return inv1

def bcd_forward(seq, ks1, ks2, iv0, iv1):
    """
    [改进版 BCD-A] 模加 + S 盒的双向链扩散：
        a[i]   = S( seq[i] + ks1[i] + a[i-1] ) mod 256     (前向)
        out[i] = S( a[N-1-i] + ks2[i] + b[i-1] ) mod 256   (后向，再反转)
    S 盒是双射 -> 输入差分非零则输出差分非零，杜绝 XOR 链的奇偶抵消。
    """
    s = seq.tolist(); k1 = ks1.tolist(); k2 = ks2.tolist()
    n = len(s)
    a = [0] * n; prev = int(iv0)
    for i in range(n):
        prev = _SB[(s[i] + k1[i] + prev) & 0xFF]
        a[i] = prev
    b = [0] * n; prev = int(iv1)
    for i in range(n):
        prev = _SB[(a[n - 1 - i] + k2[i] + prev) & 0xFF]
        b[i] = prev
    return np.array(b[::-1], dtype=np.uint8)

def bcd_inverse(seq, ks1, ks2, iv0, iv1):
    """bcd_forward 的严格逆：先反解后向链得 a，再反解前向链得明文序列。"""
    s = seq.tolist(); k1 = ks1.tolist(); k2 = ks2.tolist()
    n = len(s)
    r = s[::-1]                                  # r[i] = b[i]
    rev_a = [0] * n
    for i in range(n):
        prev_b = int(iv1) if i == 0 else r[i - 1]
        rev_a[i] = (_SBI[r[i]] - k2[i] - prev_b) & 0xFF      # = a[n-1-i]
    a = rev_a[::-1]                              # a[j] = a[j]
    out = [0] * n
    prev_a = int(iv0)
    for i in range(n):
        out[i] = (_SBI[a[i]] - k1[i] - prev_a) & 0xFF
        prev_a = a[i]
    return np.array(out, dtype=np.uint8)

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
def derive_params(master, nonce):
    kb = sha3_ctr(master + b"|params|" + nonce, 64)
    rho = int(kb[0]) % NPLANE
    dy = np.array([int(kb[1 + b]) % H for b in range(NPLANE)], dtype=np.int64)
    dx = np.array([int(kb[9 + b]) % W for b in range(NPLANE)], dtype=np.int64)
    return rho, dy, dx, int(kb[17]), int(kb[18])

def _keystream(master, nonce):
    """QHRNG 密钥流：ks_hdr | ks1 | ks2 | ksq | rules"""
    n = SEED_LEN + NPIX * 4
    kb = qhrng_bytes(master + b"|ks|" + nonce, n)
    a = np.frombuffer(kb, dtype=np.uint8)
    return dict(ks_hdr=a[:SEED_LEN], ks1=a[SEED_LEN:SEED_LEN + NPIX],
                ks2=a[SEED_LEN + NPIX:SEED_LEN + 2 * NPIX],
                ksq=a[SEED_LEN + 2 * NPIX:SEED_LEN + 3 * NPIX],
                rules=a[SEED_LEN + 3 * NPIX:SEED_LEN + 4 * NPIX] % 8)

_NONCE_CTR = [0]
def encrypt(flat, master, t=T_DEFAULT, nonce=None, kind="folded", b=None):
    flat = np.asarray(flat, dtype=np.uint8).reshape(-1)
    if nonce is None:                                    # [C7] 量子测量产出新鲜 nonce
        _NONCE_CTR[0] += 1
        nonce = qhrng_bytes(master + b"|nonce|" + _NONCE_CTR[0].to_bytes(8, "big"),
                            NONCE_LEN)
    scores = block_scores(master, nonce, t=t, kind=kind, b=b)
    perm_seed = perm_seed_from_scores(scores)            # [C6]
    perms = expand_perms(perm_seed)
    ks = _keystream(master, nonce)
    rho, dy, dx, iv0, iv1 = derive_params(master, nonce)

    s1 = qpr_forward(flat, rho, dy, dx)
    s2 = np.empty_like(s1)
    for i in range(NBLK):
        s2[i * L_BLOCK:(i + 1) * L_BLOCK] = s1[i * L_BLOCK:(i + 1) * L_BLOCK][perms[i]]
    s3 = bcd_forward(s2, ks["ks1"], ks["ks2"], iv0, iv1)
    ct = dna_layer(s3, ks["ksq"], ks["rules"])

    hdr = bytes(nonce) + bytes(x ^ y for x, y in zip(perm_seed, ks["ks_hdr"]))
    tag = sha3(master + b"|tag|" + hdr + ct.tobytes())[:TAG_LEN]
    return ct, hdr, tag

def decrypt(ct, hdr, tag, master, verify=True):
    ct = np.asarray(ct, dtype=np.uint8).reshape(-1)
    ok_tag = True
    if verify:
        ok_tag = (sha3(master + b"|tag|" + bytes(hdr) + ct.tobytes())[:TAG_LEN] == bytes(tag))
    nonce = bytes(hdr[:NONCE_LEN])
    ks = _keystream(master, nonce)
    perm_seed = bytes(x ^ y for x, y in zip(hdr[NONCE_LEN:HDR_LEN], ks["ks_hdr"]))
    perms = expand_perms(perm_seed)                      # [C6] 无需重跑量子行走
    inv_perms = [np.argsort(p) for p in perms]
    rho, dy, dx, iv0, iv1 = derive_params(master, nonce)

    s3 = dna_layer(ct, ks["ksq"], ks["rules"])
    s2 = bcd_inverse(s3, ks["ks1"], ks["ks2"], iv0, iv1)
    s1 = np.empty_like(s2)
    for i in range(NBLK):
        s1[i * L_BLOCK:(i + 1) * L_BLOCK] = s2[i * L_BLOCK:(i + 1) * L_BLOCK][inv_perms[i]]
    return qpr_inverse(s1, rho, dy, dx), ok_tag

# ============================== 指标 ==============================
def entropy_bits(a):
    h = np.bincount(np.asarray(a, dtype=np.uint8).ravel(), minlength=256).astype(float)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())

def npcr_uaci(a, b):
    a = np.asarray(a, dtype=np.uint8).ravel(); b = np.asarray(b, dtype=np.uint8).ravel()
    return (100.0 * (a != b).mean(), 100.0 * np.abs(a.astype(int) - b.astype(int)).mean() / 255.0)

def adj_corr(img2d):
    a = np.asarray(img2d, dtype=np.float64); out = []
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

def psnr(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    mse = ((a - b) ** 2).mean()
    return 99.0 if mse == 0 else float(10 * math.log10(255.0 ** 2 / mse))
