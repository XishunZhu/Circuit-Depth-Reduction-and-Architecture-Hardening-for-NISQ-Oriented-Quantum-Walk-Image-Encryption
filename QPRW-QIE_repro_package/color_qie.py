# -*- coding: utf-8 -*-
"""
彩色图像扩展 (QPRW-QIE → 彩色): GQIR/QIRHSI 风格的跨通道可逆耦合
================================================================
把 QPRW-QIE 推广到 RGB 三通道:
  1) 每通道独立做完整 QPR (量子比特面转子, 各通道密钥偏移);
  2) 跨通道转子 CC-Rotor: 在 (x,y,bit-plane) 上做 R→G→B 的循环 CNOT 链
     (类 QPR 的线性面链, 但跨通道) —— 把"通道独立性"压成耦合整体, 可逆;
  3) 三通道字节 (逐像素 R,G,B) 拼成 3·H·W 的流, 复用灰度流程的 QWP 置换 +
     双向链 + DNA;
  4) 分通道 AUTH。

对应 report.md §7 后续工作 4 (GQIR / QIRHSI 思路 + 类 QPR 跨通道耦合)。
"""
import json
import hashlib
import math
import os

import numpy as np

from qprw_qie import (derive_key, recover_key_bits, _ivs, rotor_params,
                      to_bitplanes, from_bitplanes, qpr_forward, qpr_inverse,
                      qw_schedule, apply_perms, invert_perms, bcd_forward,
                      bcd_inverse, chain_forward, chain_inverse, dna_xor_cipher,
                      auth_tag, DEFAULT_CFG, smcm, smcm_bytes, _frac, whiten)

RNG = np.random.default_rng(12345)


# ---- 合成彩色测试图 (256×256×3) ----
def synthetic_color(n=256):
    ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    gx = (xs / (n - 1) * 255).astype(np.uint8)
    gy = (ys / (n - 1) * 255).astype(np.uint8)
    R = gx
    G = gy
    B = ((gx.astype(np.int16) + gy.astype(np.int16)) // 2).astype(np.uint8)
    cy, cx = n // 3, n // 3
    m = (ys - cy) ** 2 + (xs - cx) ** 2 < (n // 10) ** 2
    R[m] = 255 - R[m]; G[m] = 30; B[m] = 200
    cy2, cx2 = 2 * n // 3, 2 * n // 3
    m2 = (ys - cy2) ** 2 + (xs - cx2) ** 2 < (n // 8) ** 2
    R[m2] = 40; G[m2] = 255 - G[m2]; B[m2] = 60
    return np.stack([R, G, B], axis=-1)


# ---- 跨通道转子 CC-Rotor ----
def cc_rotor_forward(Vcc):
    """Vcc: (H,W,8,3). 跨通道循环 CNOT 链: G^=R; B^=G (逐比特面)。可逆。"""
    V = Vcc.copy()
    for b in range(8):
        V[:, :, b, 1] ^= V[:, :, b, 0]
        V[:, :, b, 2] ^= V[:, :, b, 1]
    return V


def cc_rotor_inverse(Vcc):
    V = Vcc.copy()
    for b in range(7, -1, -1):
        V[:, :, b, 2] ^= V[:, :, b, 1]
        V[:, :, b, 1] ^= V[:, :, b, 0]
    return V


def color_encrypt(img, master_hex, cfg=None):
    c = dict(DEFAULT_CFG)
    if cfg:
        c.update(cfg)
    cfg = c
    H, W, _ = img.shape
    key_bits, T = derive_key(master_hex, img.tobytes())
    iv0, iv1 = _ivs(master_hex)

    # 1) 每通道 QPR (各通道密钥偏移)
    channels = []
    rparams = []
    for ch in range(3):
        rows, cols, rho = rotor_params(np.array(key_bits), H, W)
        rows = (rows + ch * 3) % H
        cols = (cols + (ch + 1) * 5) % W
        rho = (rho + ch * 3) % 8
        V = to_bitplanes(img[:, :, ch])
        Vr = qpr_forward(V, rows, cols, rho)
        channels.append(Vr)
        rparams.append((rows.tolist(), cols.tolist(), int(rho)))

    VR = np.stack(channels, axis=-1)            # (H,W,8,3)
    # 2) 跨通道转子
    Vcc = cc_rotor_forward(VR)

    # 3) 把 8 比特面 (含 QPR + 跨通道耦合) 按通道塌缩为字节图像,
    #    再以逐像素 R,G,B 字节流进入 QWP + 双向链 + DNA
    ch_bytes = [from_bitplanes(Vcc[:, :, :, ch]) for ch in range(3)]
    cipher_plain = np.stack(ch_bytes, axis=-1)                       # (H,W,3)
    stream = cipher_plain.ravel().astype(np.uint8)                   # 长度 3·H·W
    N = stream.size

    perms, K = qw_schedule(N, key_bits, nq=cfg['nq'], t=cfg['t'], discard=80,
                           ks_whiten=cfg.get('whiten', True))
    if cfg['qw_perm']:
        stream = apply_perms(stream, perms, cfg['nq'])
    if cfg['two_way']:
        C = bcd_forward(stream, K, iv0, iv1)
    else:
        C = chain_forward(stream, K, iv0)
    if cfg['dna']:
        Xk, Yk = smcm(N, _frac(key_bits[8]), _frac(key_bits[1]))
        kseq = smcm_bytes(Yk)
        if cfg.get('whiten', True):
            kseq = whiten(kseq, hashlib.sha3_256(b'DNA' + T).digest())
        rules = (np.floor(Xk * 8)).astype(np.int64) % 8
        C = dna_xor_cipher(C, kseq, rules)
    cipher = C.reshape(H, W, 3)

    tags = [auth_tag(cipher[:, :, ch], key_bits, cfg['block']) for ch in range(3)]
    aux = dict(T=T, iv0=iv0, iv1=iv1, rparams=rparams, tags=tags,
               shape=(H, W), **{k: v for k, v in cfg.items()})
    return cipher, aux


def color_decrypt(cipher, master_hex, aux):
    H, W = aux['shape']
    cfg = aux
    key_bits = recover_key_bits(master_hex, aux['T'])
    C = np.ascontiguousarray(cipher).ravel().astype(np.uint8)         # 长度 3·H·W

    if cfg['dna']:
        Xk, Yk = smcm(C.size, _frac(key_bits[8]), _frac(key_bits[1]))
        kseq = smcm_bytes(Yk)
        if cfg.get('whiten', True):
            kseq = whiten(kseq, hashlib.sha3_256(b'DNA' + cfg['T']).digest())
        rules = (np.floor(Xk * 8)).astype(np.int64) % 8
        C = dna_xor_cipher(C, kseq, rules)

    perms, K = qw_schedule(C.size, key_bits, nq=cfg['nq'], t=cfg['t'], discard=80,
                           ks_whiten=cfg.get('whiten', True))
    if cfg['two_way']:
        S = bcd_inverse(C, K, aux['iv0'], aux['iv1'])
    else:
        S = chain_inverse(C, K, aux['iv0'])
    if cfg['qw_perm']:
        S = invert_perms(S, perms, cfg['nq'])

    cipher_plain = S.reshape(H, W, 3)
    ch_planes = [to_bitplanes(cipher_plain[:, :, ch]) for ch in range(3)]   # (H,W,8)
    Vcc = np.stack(ch_planes, axis=-1)                                    # (H,W,8,3)

    VR = cc_rotor_inverse(Vcc)
    out_ch = []
    for ch in range(3):
        rows, cols, rho = (np.array(aux['rparams'][ch][0]),
                           np.array(aux['rparams'][ch][1]), aux['rparams'][ch][2])
        V = qpr_inverse(VR[:, :, :, ch], rows, cols, rho)
        out_ch.append(from_bitplanes(V))
    return np.stack(out_ch, axis=-1)


# ---- 指标 (复用 run_experiments) ----
def _metrics_chs(plain, cipher):
    import run_experiments as re_
    res = {}
    for ch, name in enumerate(['R', 'G', 'B']):
        p = plain[:, :, ch]
        c = cipher[:, :, ch]
        res[name] = dict(entropy=re_.entropy(c), chi2=re_.chi_square(c),
                         corr=re_.correlation(c), psnr_vs_plain=re_.psnr(p, c))
    return res


def _npcr_uaci(c1, c2):
    import run_experiments as re_
    return re_.npcr_uaci(c1, c2)


def main():
    master_hex = 'a1b2c3d4e5f60718293a4b5c6d7e8f90' + '1029384756aabbccddeeff0011223344'
    img = synthetic_color(256)
    C, aux = color_encrypt(img, master_hex)
    D = color_decrypt(C, master_hex, aux)
    exact = np.array_equal(img, D)
    mse = float(np.mean((img.astype(float) - D.astype(float)) ** 2))

    img2 = img.copy()
    img2[128, 128, 0] ^= 1
    C2, _ = color_encrypt(img2, master_hex)
    npcr, uaci = _npcr_uaci(C, C2)

    k2 = ('f' if master_hex[0] != 'f' else 'e') + master_hex[1:]
    Ck, _ = color_encrypt(img, k2)
    knpcr, kuaci = _npcr_uaci(C, Ck)

    ch_metrics = _metrics_chs(img, C)

    out = {
        'image_shape': list(img.shape),
        'decrypt_exact': bool(exact),
        'decrypt_mse': mse,
        'npcr': npcr, 'uaci': uaci,
        'key_npcr': knpcr, 'key_uaci': kuaci,
        'per_channel': ch_metrics,
    }
    os.makedirs('results', exist_ok=True)
    with open('results/color.json', 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('color decrypt exact =', exact, ' MSE =', mse)
    print('NPCR = %.3f  UACI = %.3f  keyNPCR = %.3f' % (npcr, uaci, knpcr))
    print('per-channel entropy:', {k: round(v['entropy'], 4) for k, v in ch_metrics.items()})
    print('per-channel corrH:', {k: round(v['corr']['H'], 4) for k, v in ch_metrics.items()})
    return out


if __name__ == '__main__':
    main()
