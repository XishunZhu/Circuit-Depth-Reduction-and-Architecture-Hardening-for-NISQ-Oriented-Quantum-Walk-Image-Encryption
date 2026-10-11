# -*- coding: utf-8 -*-
"""
QPRW-QIE: 量子比特面转子 + 量子游走驱动分块置换 + 双向链式扩散 的图像加密算法
================================================================================
流水线 (灰度, 尺寸 2^n × 2^n):

  P0 密钥派生    K(256 bit) ⊕ H=SHA3-256(明文) -> 混沌初值;  对外只发布 T = H ⊕ HK
  A  QPR         量子比特面转子: 逐面循环移位 -> 面间轮转 -> 跨面可逆非线性注入
  B  QWP         量子游走驱动的分块置换 (QW 概率分布 -> argsort -> 块内置换)
  C  BCD         双向链式扩散 (前向链 + 反向链), 密钥流由 SMCM 逐块重播种
  D  DNA         动态 DNA 编码 / DNA-XOR (逐像素规则切换)
  E  AUTH        分块认证标签 (篡改检测与定位)

模块与文献对照见 report.md。本文件在经典计算机上以双精度模拟上述量子线路的等价行为。
"""
import hashlib
import math

import numpy as np

# =====================================================================
# 1. SMCM —— Sine-Modulated Coupled Map (本工作定义的二维超混沌映射)
# =====================================================================
#   x_{n+1} = mod( sin( pi * ( a * 4 x_n (1-x_n) + b * y_n ) ) + 2, 1 )
#   y_{n+1} = mod( sin( pi * ( c * 4 y_n (1-y_n) + d * x_{n+1} ) ) + 2, 1 )
#   默认 (a,b,c,d) = (1.00, 1.60, 1.00, 1.70); 实测 Lyapunov ≈ (1.86, 0.31) —— 双正, 超混沌

SMCM_DEF = dict(a=1.00, b=1.60, c=1.00, d=1.70)


def _step(x, y, p):
    xn = math.fmod(math.sin(math.pi * (p[0] * 4.0 * x * (1.0 - x) + p[1] * y)) + 2.0, 1.0)
    if xn < 0.0:
        xn += 1.0
    yn = math.fmod(math.sin(math.pi * (p[2] * 4.0 * y * (1.0 - y) + p[3] * xn)) + 2.0, 1.0)
    if yn < 0.0:
        yn += 1.0
    return xn, yn


def smcm(n, x0, y0, p=None, discard=1000):
    """长度 n 的双通道混沌序列。"""
    p = tuple(p) if p is not None else (SMCM_DEF['a'], SMCM_DEF['b'], SMCM_DEF['c'], SMCM_DEF['d'])
    x, y = _frac(x0), _frac(y0)
    for _ in range(discard):
        x, y = _step(x, y, p)
    X = np.empty(n)
    Y = np.empty(n)
    for i in range(n):
        x, y = _step(x, y, p)
        X[i] = x
        Y[i] = y
    return X, Y


def _jac(x, y, p):
    z1 = p[0] * 4.0 * x * (1.0 - x) + p[1] * y
    s1 = math.pi * math.cos(math.pi * z1)
    fx_x = s1 * p[0] * 4.0 * (1.0 - 2.0 * x)
    fx_y = s1 * p[1]
    xn = math.fmod(math.sin(math.pi * z1) + 2.0, 1.0)
    z2 = p[2] * 4.0 * y * (1.0 - y) + p[3] * xn
    s2 = math.pi * math.cos(math.pi * z2)
    return np.array([[fx_x, fx_y],
                     [s2 * p[3] * fx_x,
                      s2 * (p[2] * 4.0 * (1.0 - 2.0 * y) + p[3] * fx_y)]])


def smcm_lyapunov(x0=0.37, y0=0.61, n=20000, discard=2000, **kw):
    p = dict(SMCM_DEF)
    p.update(kw)
    p = (p['a'], p['b'], p['c'], p['d'])
    x, y = _frac(x0), _frac(y0)
    for _ in range(discard):
        x, y = _step(x, y, p)
    Q = np.eye(2)
    lam = np.zeros(2)
    for _ in range(n):
        Q, R = np.linalg.qr(np.dot(_jac(x, y, p), Q))
        dg = np.abs(np.diag(R))
        dg[dg == 0] = 1e-14
        lam += np.log(dg)
        x, y = _step(x, y, p)
    return lam / n


def smcm_bytes(X, mul=1e12):
    """混沌浮点序列 -> [0,255] 字节流 (取低位尾数, 实测 chi2≈241/255, 自相关≈0.001)。"""
    return np.mod(np.floor(X * mul).astype(np.int64), 256).astype(np.uint8)


def hash_stream(seed, n, tag=b'WH'):
    """SHA3-256 计数器模式字节流: 每字节分布近完美均匀 (chi2≈df), 用作白化源。"""
    out = bytearray()
    ctr = 0
    while len(out) < n:
        out += hashlib.sha3_256(tag + seed + ctr.to_bytes(4, 'little')).digest()
        ctr += 1
    return np.frombuffer(bytes(out[:n]), dtype=np.uint8)


def whiten(kb, seed):
    """密钥流白化: SMCM 量化字节 ⊕ SHA3-CTR 字节。

    浮点轨道的 mantissa 量化存在轻微非均匀性 (密钥流熵 ≈7.9991 bit/byte),
    与均匀的 SHA3-CTR 字节异或后分布被推向 iid 均匀 (熵 ≈7.9999 bit/byte),
    同时保留 SMCM 状态对密钥流的决定性贡献。"""
    return np.bitwise_xor(np.asarray(kb, dtype=np.uint8), hash_stream(seed, len(kb)))


# =====================================================================
# 2. DTQW —— 环上离散时间量子游走
# =====================================================================

def dtqw_cycle(N, t, theta, init='symmetric', avg_last=0):
    """N 位置环上的 DTQW, t 步; 每步 = 位移 S 后接硬币 C(θ)。

    theta: 标量 / 长度 t (逐步) / 形状 (t, N) (逐步且逐位置)
    avg_last: >0 时对最后 avg_last 步的概率分布取时间平均 (抑制弹道振荡)
    返回归一化概率分布 P (长度 N)。
    """
    psi = np.zeros((N, 2), dtype=np.complex128)
    if init == 'symmetric':
        psi[0, 0] = 1.0 / math.sqrt(2.0)
        psi[0, 1] = 1.0j / math.sqrt(2.0)
    else:
        psi[0, 0] = 1.0
    th = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    acc = np.zeros(N)
    L = int(avg_last)
    for s in range(t):
        a0 = np.roll(psi[:, 0], 1)        # |x,0> <- |x-1,0>
        a1 = np.roll(psi[:, 1], -1)       # |x,1> <- |x+1,1>
        tt = th[s] if th.ndim > 1 else (th[s] if th.size > 1 else float(th[0]))
        co = np.cos(tt)
        si = np.sin(tt)
        psi[:, 0] = co * a0 - si * a1
        psi[:, 1] = si * a0 + co * a1
        if L and s >= t - L:
            acc += np.abs(psi[:, 0]) ** 2 + np.abs(psi[:, 1]) ** 2
    if L:
        P = acc / L
    else:
        P = np.abs(psi[:, 0]) ** 2 + np.abs(psi[:, 1]) ** 2
    return P / (P.sum() + 1e-300)


# ---- QW 调度: 分块置换 + 逐块重播种密钥流 ----
NQ_DEFAULT = 256          # QW 环大小 = 分块长度
TQ_DEFAULT = 128          # QW 步数 (弹道扩散 2t = NQ, 覆盖整个环)
AVG_LAST = 32             # 时间平均步数


def qw_schedule(N, key_bits, nq=NQ_DEFAULT, t=TQ_DEFAULT, avg_last=AVG_LAST, discard=300,
                ks_whiten=True):
    """生成 nblk 个 QW 置换与整幅图的密钥流。

    硬币角 θ(s,x) = π * (0.10 + 0.30 * frac( A(s) + B(x) + φ_b ))   # 安全区 [0.10π, 0.40π]
    其中 A/B 由密钥驱动的 SMCM 生成, φ_b 为逐块相位并与上一块密钥流反馈耦合。
    每块的置换 π_b = argsort(P_b + 微扰), 密钥流由 SHA3(P_b) 重播种 SMCM 得到,
    再经 SHA3-CTR 白化 (ks_whiten=True) 消除量化偏差。
    """
    A, _ = smcm(t, _frac(key_bits[0]), _frac(key_bits[1]))
    B, _ = smcm(nq, _frac(key_bits[4]), _frac(key_bits[5]))
    J, _ = smcm(nq, _frac(key_bits[6]), _frac(key_bits[7]))   # 破同分微扰
    nblk = int(math.ceil(N / float(nq)))
    perms = []
    ks = []
    feedback = 0.0
    phi0 = _frac(key_bits[3])
    for b in range(nblk):
        phi = (phi0 + b * 0.6180339887498949 + feedback) % 1.0
        theta = math.pi * (0.10 + 0.30 * np.mod(A[:, None] + B[None, :] + phi, 1.0))
        P = dtqw_cycle(nq, t, theta, avg_last=avg_last)
        perms.append(np.argsort(P + 1e-16 * J, kind='stable').astype(np.int64))
        seed = hashlib.sha3_256(P.tobytes() + bytes([b & 0xFF])).digest()
        X, _ = smcm(nq, seed[0] / 255.0, seed[1] / 255.0, discard=discard)
        kb = smcm_bytes(X)
        if ks_whiten:
            kb = whiten(kb, hashlib.sha3_256(b'KS' + seed).digest())
        ks.append(kb)
        feedback = kb[-1] / 256.0
    return perms, np.concatenate(ks)[:N]


def apply_perms(flat, perms, nq=NQ_DEFAULT):
    """加密方向: out[j] = in[π[j]]。"""
    out = np.empty_like(flat)
    for b, p in enumerate(perms):
        s = b * nq
        seg = flat[s:s + nq]
        if seg.size == nq:
            out[s:s + nq] = seg[p]
        else:
            out[s:s + seg.size] = seg[p[p < seg.size]]
    return out


def invert_perms(flat, perms, nq=NQ_DEFAULT):
    out = np.empty_like(flat)
    for b, p in enumerate(perms):
        s = b * nq
        seg = flat[s:s + nq]
        if seg.size == nq:
            out[s:s + nq][p] = seg
        else:
            m = p < seg.size
            out[s:s + seg.size][p[m]] = seg[m]
    return out


# =====================================================================
# 3. QPR —— 量子比特面转子
# =====================================================================

def to_bitplanes(img):
    V = np.empty(img.shape + (8,), dtype=np.uint8)
    for b in range(8):
        V[:, :, b] = (img >> b) & 1
    return V


def from_bitplanes(V):
    img = np.zeros(V.shape[:2], dtype=np.uint8)
    for b in range(8):
        img |= (V[:, :, b].astype(np.uint8) << b)
    return img


def rotor_params(key_bits, H, W):
    X, Y = smcm(24, _frac(key_bits[2]), _frac(key_bits[3]))
    rows = (np.floor(X[:8] * H)).astype(np.int64) % H
    cols = (np.floor(Y[:8] * W)).astype(np.int64) % W
    rho = int(np.floor(X[8] * 8)) % 8
    return rows, cols, rho


def qpr_forward(V, rows, cols, rho):
    V1 = np.empty_like(V)
    for b in range(8):
        V1[:, :, b] = np.roll(np.roll(V[:, :, b], int(rows[b]), axis=0), int(cols[b]), axis=1)
    V2 = np.empty_like(V1)
    for b in range(8):
        V2[:, :, b] = V1[:, :, (b - int(rho)) % 8]
    for b in range(8):                      # 非线性跨面注入 (8 层 Toffoli, 可逆)
        V2[:, :, b] ^= (V2[:, :, (b + 1) % 8] & V2[:, :, (b + 4) % 8])
    for b in range(8):                      # 循环线性面链 (8 层 CNOT): 任一面翻转 -> 8 面全翻
        V2[:, :, b] ^= V2[:, :, (b - 1) % 8]
    return V2


def qpr_inverse(V, rows, cols, rho):
    V2 = V.copy()
    for b in range(7, -1, -1):              # 逆序解除循环线性面链
        V2[:, :, b] ^= V2[:, :, (b - 1) % 8]
    for b in range(7, -1, -1):              # 逆序解除非线性注入
        V2[:, :, b] ^= (V2[:, :, (b + 1) % 8] & V2[:, :, (b + 4) % 8])
    V1 = np.empty_like(V2)
    for b in range(8):
        V1[:, :, (b - int(rho)) % 8] = V2[:, :, b]
    out = np.empty_like(V1)
    for b in range(8):
        out[:, :, b] = np.roll(np.roll(V1[:, :, b], -int(rows[b]), axis=0), -int(cols[b]), axis=1)
    return out


# =====================================================================
# 4. 链式扩散
# =====================================================================

def chain_forward(S, K, iv):
    S = S.astype(np.int64)
    K = K.astype(np.int64)
    N = S.size
    C = np.empty(N, dtype=np.int64)
    prev = int(iv)
    for i in range(N):
        prev = S[i] ^ K[i] ^ prev
        C[i] = prev
    return C.astype(np.uint8)


def chain_inverse(C, K, iv):
    C = C.astype(np.int64)
    K = K.astype(np.int64)
    S = np.empty(C.size, dtype=np.int64)
    prev = int(iv)
    for i in range(C.size):
        S[i] = C[i] ^ K[i] ^ prev
        prev = C[i]
    return S.astype(np.uint8)


def bcd_forward(S, K, iv0, iv1):
    """前向链 C1[i]=S[i]^K[i]^C1[i-1]; 反向链 C[i]=C1[i]^K[N-1-i]^C[i+1]。"""
    C1 = chain_forward(S, K, iv0)                       # Pass 1
    D = chain_forward(C1[::-1].copy(), K, iv1)          # Pass 2 (在反转序列上做前向链)
    return D[::-1].copy()


def bcd_inverse(C, K, iv0, iv1):
    C1 = chain_inverse(C[::-1].copy(), K, iv1)[::-1].copy()
    return chain_inverse(C1, K, iv0)


# =====================================================================
# 5. 动态 DNA 编码 / DNA-XOR
# =====================================================================
BASES = ['A', 'C', 'G', 'T']
RULES = [('A', 'C', 'G', 'T'), ('A', 'C', 'T', 'G'), ('A', 'G', 'C', 'T'), ('A', 'G', 'T', 'C'),
         ('A', 'T', 'C', 'G'), ('A', 'T', 'G', 'C'), ('C', 'A', 'G', 'T'), ('C', 'A', 'T', 'G')]
_XOR = {('A', 'A'): 'A', ('A', 'C'): 'C', ('A', 'G'): 'G', ('A', 'T'): 'T',
        ('C', 'A'): 'C', ('C', 'C'): 'A', ('C', 'G'): 'T', ('C', 'T'): 'G',
        ('G', 'A'): 'G', ('G', 'C'): 'T', ('G', 'G'): 'A', ('G', 'T'): 'C',
        ('T', 'A'): 'T', ('T', 'C'): 'G', ('T', 'G'): 'C', ('T', 'T'): 'A'}
_XORIDX = {(BASES.index(p), BASES.index(q)): BASES.index(v) for (p, q), v in _XOR.items()}


def _build_dna_tables():
    tab = np.zeros((8, 256, 256), dtype=np.uint8)
    for r, rule in enumerate(RULES):
        enc = np.array([BASES.index(ch) for ch in rule], dtype=np.int64)   # 2bit -> 碱基下标
        inv = np.zeros(4, dtype=np.int64)
        for pair in range(4):
            inv[enc[pair]] = pair
        PV = np.array([[(v >> (2 * (3 - j))) & 3 for j in range(4)] for v in range(256)])
        EV = enc[PV]
        for k in range(256):
            EK = enc[np.array([(k >> (2 * (3 - j))) & 3 for j in range(4)])]
            C = np.array([_XORIDX[(int(EV[v, j]), int(EK[j]))]
                          for v in range(256) for j in range(4)]).reshape(256, 4)
            out = np.zeros(256, dtype=np.int64)
            for j in range(4):
                out |= inv[C[:, j]] << (2 * (3 - j))
            tab[r, :, k] = out.astype(np.uint8)
    return tab


DNA_TAB = _build_dna_tables()


def dna_xor_cipher(seq, kseq, rules):
    """逐像素动态规则 DNA-XOR (自反: 同密钥同规则再做一次即还原)。"""
    out = np.empty(seq.size, dtype=np.uint8)
    for r in range(8):
        m = (rules == r)
        if m.any():
            out[m] = DNA_TAB[r][seq[m], kseq[m]]
    return out


# =====================================================================
# 6. 密钥派生 / 主加解密
# =====================================================================

def _frac(v):
    v = math.fmod(abs(float(v)), 1.0)
    if v < 1e-6:
        v += 0.3183099
    if v > 1 - 1e-6:
        v -= 0.3183099
    return v


def derive_key(master_hex, plain_bytes):
    """主密钥 K(256 bit) 与明文摘要 H 耦合; 对外只发布 T = H ⊕ HK (不泄露明文摘要)。"""
    K = bytes.fromhex(master_hex)
    H = hashlib.sha3_256(plain_bytes).digest()
    HK = hashlib.sha3_256(b'HK' + K).digest()
    T = bytes(a ^ b for a, b in zip(H, HK))
    seed = hashlib.sha3_256(b'SEED' + K + H).digest()
    return [b / 255.0 for b in seed], T


def recover_key_bits(master_hex, T):
    K = bytes.fromhex(master_hex)
    HK = hashlib.sha3_256(b'HK' + K).digest()
    H = bytes(a ^ b for a, b in zip(T, HK))
    seed = hashlib.sha3_256(b'SEED' + K + H).digest()
    return [b / 255.0 for b in seed]


def _ivs(master_hex, salt=b''):
    M = bytes.fromhex(master_hex)
    return (hashlib.sha3_256(b'IV0' + salt + M).digest()[0],
            hashlib.sha3_256(b'IV1' + salt + M).digest()[0])


DEFAULT_CFG = dict(qpr=True, qw_perm=True, two_way=True, dna=True, arnold=0,
                   block=64, nq=NQ_DEFAULT, t=TQ_DEFAULT, whiten=True)


def encrypt(img, master_hex, cfg=None):
    """cfg 键: qpr / qw_perm / two_way / dna / arnold / block / nq / t"""
    c = dict(DEFAULT_CFG)
    if cfg:
        c.update(cfg)
    cfg = c

    H, W = img.shape
    N = H * W
    key_bits, T = derive_key(master_hex, img.tobytes())
    iv0, iv1 = _ivs(master_hex)

    S = img.ravel().copy()
    if cfg['arnold'] > 0 and H == W:
        S = arnold_scramble(img, 3, 5, cfg['arnold']).ravel().copy()

    if cfg['qpr']:
        rows, cols, rho = rotor_params(key_bits, H, W)
        S = from_bitplanes(qpr_forward(to_bitplanes(S.reshape(H, W)), rows, cols, rho)).ravel()
    else:
        rows, cols, rho = np.zeros(8, dtype=np.int64), np.zeros(8, dtype=np.int64), 0

    perms, K = qw_schedule(N, key_bits, nq=cfg['nq'], t=cfg['t'], ks_whiten=cfg.get('whiten', True))
    if cfg['qw_perm']:
        S = apply_perms(S, perms, cfg['nq'])

    if cfg['two_way']:
        C = bcd_forward(S, K, iv0, iv1)
    else:
        C = chain_forward(S, K, iv0)

    if cfg['dna']:
        Xk, Yk = smcm(N, _frac(key_bits[8]), _frac(key_bits[1]))
        kseq = smcm_bytes(Yk)
        if cfg.get('whiten', True):
            kseq = whiten(kseq, hashlib.sha3_256(b'DNA' + T).digest())
        rules = (np.floor(Xk * 8)).astype(np.int64) % 8
        C = dna_xor_cipher(C, kseq, rules)
    else:
        rules = None

    tag = auth_tag(C.reshape(H, W), key_bits, cfg['block'])
    aux = dict(T=T, iv0=iv0, iv1=iv1, rows=rows.tolist(), cols=cols.tolist(), rho=int(rho),
               tag=tag, shape=(H, W), **{k: v for k, v in cfg.items()})
    return C.reshape(H, W), aux


def decrypt(cipher, master_hex, aux):
    H, W = aux['shape']
    N = H * W
    cfg = aux
    key_bits = recover_key_bits(master_hex, aux['T'])
    C = np.ascontiguousarray(cipher).ravel().copy()

    if cfg['dna']:
        Xk, Yk = smcm(N, _frac(key_bits[8]), _frac(key_bits[1]))
        kseq = smcm_bytes(Yk)
        if cfg.get('whiten', True):
            kseq = whiten(kseq, hashlib.sha3_256(b'DNA' + cfg['T']).digest())
        rules = (np.floor(Xk * 8)).astype(np.int64) % 8
        C = dna_xor_cipher(C, kseq, rules)

    perms, K = qw_schedule(N, key_bits, nq=cfg['nq'], t=cfg['t'], ks_whiten=cfg.get('whiten', True))
    if cfg['two_way']:
        S = bcd_inverse(C, K, aux['iv0'], aux['iv1'])
    else:
        S = chain_inverse(C, K, aux['iv0'])

    if cfg['qw_perm']:
        S = invert_perms(S, perms, cfg['nq'])

    if cfg['qpr']:
        V = qpr_inverse(to_bitplanes(S.reshape(H, W)), np.array(aux['rows']),
                        np.array(aux['cols']), aux['rho'])
        S = from_bitplanes(V).ravel()
    img = S.reshape(H, W).copy()

    if cfg['arnold'] > 0 and H == W:
        img = arnold_unscramble(img, 3, 5, cfg['arnold'])
    return img


def arnold_scramble(img, a, b, rounds):
    n = img.shape[0]
    idx = np.arange(n)
    X, Y = np.meshgrid(idx, idx, indexing='ij')
    out = img.copy()
    for _ in range(rounds):
        Xn = (X + a * Y) % n
        Yn = (b * X + (a * b + 1) * Y) % n
        out = out[Xn, Yn]
        X, Y = Xn, Yn
    return out


def arnold_unscramble(img, a, b, rounds):
    n = img.shape[0]
    idx = np.arange(n)
    X, Y = np.meshgrid(idx, idx, indexing='ij')
    src = np.zeros((n, n, 2), dtype=np.int64)
    cx, cy = X.copy(), Y.copy()
    for _ in range(rounds):
        cx, cy = (cx + a * cy) % n, (b * cx + (a * b + 1) * cy) % n
    src[cx, cy, 0] = X
    src[cx, cy, 1] = Y
    return img[src[:, :, 0], src[:, :, 1]]


def auth_tag(cipher2d, key_bits, block=64):
    """分块认证标签: 每 block×block 块 -> 1 字节 (QW 硬币由块摘要与密钥共同决定)。"""
    H, W = cipher2d.shape
    tags = bytearray()
    for i in range(0, H, block):
        for j in range(0, W, block):
            h = hashlib.sha3_256(cipher2d[i:i + block, j:j + block].tobytes()).digest()
            s = ((h[0] + int(key_bits[6] * 255)) % 256) / 255.0
            P = dtqw_cycle(64, 8, math.pi * (0.10 + 0.30 * s))
            tags.append(int(math.floor(P[0] * 1e12)) % 256)
    return bytes(tags)


def verify_tag(cipher2d, key_bits, tag, block=64):
    t2 = auth_tag(cipher2d, key_bits, block)
    bad = [i for i in range(min(len(tag), len(t2))) if tag[i] != t2[i]]
    return (len(bad) == 0), bad


# =====================================================================
# 7. 基线
# =====================================================================

def baseline_arnold_chain(img, master_hex, rounds=3, nq=NQ_DEFAULT, t=TQ_DEFAULT):
    """B1: 经典两段式 —— Arnold 全局置乱 + 混沌单向链式扩散。"""
    H, W = img.shape
    N = H * W
    M = bytes.fromhex(master_hex)
    kb = [M[i % 32] / 255.0 for i in range(8)]
    P = arnold_scramble(img, 3, 5, rounds)
    X, _ = smcm(N, _frac(kb[0]), _frac(kb[1]))
    K = smcm_bytes(X)
    iv = hashlib.sha3_256(b'IV' + M).digest()[0]
    return chain_forward(P.ravel(), K, iv).reshape(H, W)


def baseline_qw_chain(img, master_hex, nq=NQ_DEFAULT, t=TQ_DEFAULT):
    """B2: 近期代表性结构 —— QW 密钥流 + 单向链式扩散 (本文方案去掉 QPR/双向链/DNA)。"""
    H, W = img.shape
    N = H * W
    M = bytes.fromhex(master_hex)
    kb = [M[i % 32] / 255.0 for i in range(8)]
    _, K = qw_schedule(N, kb, nq=nq, t=t)
    iv = hashlib.sha3_256(b'IV' + M).digest()[0]
    return chain_forward(img.ravel(), K, iv).reshape(H, W)


def baseline_qw_xor(img, master_hex, nq=NQ_DEFAULT, t=TQ_DEFAULT):
    """B0: QW 密钥流 + 逐点 XOR (无链式) —— 用于说明链式扩散的必要性。"""
    H, W = img.shape
    N = H * W
    M = bytes.fromhex(master_hex)
    kb = [M[i % 32] / 255.0 for i in range(8)]
    _, K = qw_schedule(N, kb, nq=nq, t=t)
    return (img.ravel() ^ K).reshape(H, W)
