# -*- coding: utf-8 -*-
"""三项补充实验:
  T1 选择明文攻击(CPA): 用常量图像恢复密钥流, 检验明文耦合密钥派生的必要性
  T2 QPR 的跨比特面耦合: 明文单个 LSB 翻转后, 有多少个比特面被改动
  T3 QWP 置换对密钥的敏感性: 1 bit 密钥变化后置换位置保持不变的比例
"""
import io
import json
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qprw_qie as Q  # noqa
from run_experiments import shepp_logan, psnr, npcr_uaci  # noqa
# run_experiments 已把 stdout 换成 UTF-8 wrapper, 这里复用它 (不要二次包装, 否则底层被关闭)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'results')
KEY = hashlib.sha3_256(b'QPRW-QIE-demo-master-key').hexdigest()
FULL = dict(qpr=True, qw_perm=True, two_way=True, dna=True)


def t1_cpa():
    """T1: 攻击者可提交任意明文并获得密文 (CPA  oracle)。
    用全常量图像 I0=128 加密 -> 对密钥流与明文无关的方案, 密钥流可被完整恢复,
    再用它去解另一幅图的密文。"""
    img = shepp_logan(256)
    const = np.full((256, 256), 128, dtype=np.uint8)
    res = {}

    # --- B0: QW 密钥流 + 逐点 XOR ---
    C0 = Q.baseline_qw_xor(const, KEY)
    K_est = (C0.ravel() ^ 128).astype(np.uint8)          # 直接得到密钥流
    C_real = Q.baseline_qw_xor(img, KEY)
    rec = (C_real.ravel() ^ K_est).reshape(img.shape)
    res['B0_QW-ks+XOR'] = dict(
        attack='K = C(I0) XOR 128', broken=bool(np.array_equal(rec, img)),
        recovered_psnr=psnr(img, rec))

    # --- B2: QW 密钥流 + 单向链 ---
    M = bytes.fromhex(KEY)
    kb = [M[i % 32] / 255.0 for i in range(8)]
    iv = hashlib.sha3_256(b'IV' + M).digest()[0]
    C0b = Q.baseline_qw_chain(const, KEY)
    # 已知 S=128 常量, 由 C1[i]=S[i]^K[i]^C1[i-1] 反解 K
    C1 = C0b.ravel().astype(np.int64)
    N = C1.size
    K_est2 = np.empty(N, dtype=np.int64)
    prev = iv
    for i in range(N):
        K_est2[i] = C1[i] ^ 128 ^ prev
        prev = C1[i]
    C_real_b = Q.baseline_qw_chain(img, KEY).ravel().astype(np.int64)
    S_rec = np.empty(N, dtype=np.int64)
    prev = iv
    for i in range(N):
        S_rec[i] = C_real_b[i] ^ K_est2[i] ^ prev
        prev = C_real_b[i]
    rec2 = S_rec.astype(np.uint8).reshape(img.shape)
    res['B2_QW-ks+chain'] = dict(
        attack='逐像素反解链式关系得到 K', broken=bool(np.array_equal(rec2, img)),
        recovered_psnr=psnr(img, rec2))

    # --- 本文 QPRW-QIE ---
    Cc, auxc = Q.encrypt(const, KEY, FULL)
    Cr, auxr = Q.encrypt(img, KEY, FULL)
    # 攻击者只能拿到 T_c; 由于密钥流依赖 H(明文), 由常量图推出的"密钥流"对另一幅图无效
    pseudo = (Cc.ravel() ^ 128).astype(np.uint8)
    guess = (Cr.ravel() ^ pseudo).reshape(img.shape)
    res['QPRW-QIE'] = dict(
        attack='同上, 但密钥流 = f(K, SHA3(明文))', broken=bool(np.array_equal(guess, img)),
        recovered_psnr=psnr(img, guess))
    return res


def t2_bitplane_coupling(ntrial=200):
    """T2: 随机翻转单个像素的单个比特 -> 统计 QPR 之后有多少个比特面被改动。

    不做 QPR 时, 翻转 b 号比特只会改动第 b 个比特面 (恒为 1 个面);
    做 QPR 后, 跨面注入与循环面链把改动扩散到多个面。
    """
    img = shepp_logan(256)
    key_bits, _ = Q.derive_key(KEY, img.tobytes())
    rows, cols, rho = Q.rotor_params(key_bits, 256, 256)
    V = Q.to_bitplanes(img)
    A = Q.qpr_forward(V, rows, cols, rho)
    rng = np.random.RandomState(7)
    with_qpr, without = [], []
    for _ in range(ntrial):
        i, j, b = rng.randint(256), rng.randint(256), rng.randint(8)
        p = img.copy()
        p[i, j] ^= (1 << b)
        Vp = Q.to_bitplanes(p)
        Ap = Q.qpr_forward(Vp, rows, cols, rho)
        with_qpr.append(sum(1 for k in range(8) if (A[:, :, k] != Ap[:, :, k]).sum() > 0))
        without.append(sum(1 for k in range(8) if (V[:, :, k] != Vp[:, :, k]).sum() > 0))
    return dict(n_trials=ntrial,
                avg_planes_affected_with_QPR=float(np.mean(with_qpr)),
                avg_planes_affected_without_QPR=float(np.mean(without)),
                frac_flips_hitting_ge2_planes=float(np.mean(np.array(with_qpr) >= 2)),
                max_planes_observed=int(max(with_qpr)),
                note='共 8 个比特面; 不做 QPR 时恒为 1')


def t3_perm_sensitivity():
    """T3: 1 bit 密钥变化 -> QW 置换 π 中位置保持不变的比例 (随机置换期望 1/q)。"""
    key_bits, _ = Q.derive_key(KEY, shepp_logan(256).tobytes())
    k2 = ('f' if KEY[0] != 'f' else 'e') + KEY[1:]
    M2 = bytes.fromhex(k2)
    kb2 = [M2[i % 32] / 255.0 for i in range(8)]
    nq = Q.NQ_DEFAULT
    p1, _ = Q.qw_schedule(nq, key_bits, nq=nq, t=Q.TQ_DEFAULT)
    p2, _ = Q.qw_schedule(nq, kb2, nq=nq, t=Q.TQ_DEFAULT)
    same = float((p1[0] == p2[0]).sum()) / nq
    # 也统计固定点 (π[j]==j) 比例
    fixed1 = float((p1[0] == np.arange(nq)).sum()) / nq
    return dict(block_size=nq, frac_positions_unchanged_after_1bit_key_change=same,
                frac_fixed_points=fixed1,
                note='随机置换的期望: 位置不变比例 ≈ 1/q = %.4f' % (1.0 / nq))


if __name__ == '__main__':
    out = {}
    print('=== T1 选择明文攻击 (CPA) ===')
    r1 = t1_cpa()
    for k, v in r1.items():
        print('  %-16s 攻破=%-5s 恢复图像 PSNR=%s dB'
              % (k, v['broken'], ('%.2f' % v['recovered_psnr'])
                 if np.isfinite(v['recovered_psnr']) else 'inf(=完全恢复)'))
    out['T1_CPA'] = r1

    print('\n=== T2 QPR 跨比特面耦合 ===')
    r2 = t2_bitplane_coupling()
    print('  %d 次随机单比特翻转:' % r2['n_trials'])
    print('    无 QPR: 平均受影响比特面 %.2f / 8'
          % r2['avg_planes_affected_without_QPR'])
    print('    有 QPR: 平均受影响比特面 %.2f / 8  (最多 %d 面, %.0f%% 的翻转波及 ≥2 个面)'
          % (r2['avg_planes_affected_with_QPR'], r2['max_planes_observed'],
             100.0 * r2['frac_flips_hitting_ge2_planes']))
    out['T2_bitplane_coupling'] = r2

    print('\n=== T3 QWP 置换密钥敏感性 ===')
    r3 = t3_perm_sensitivity()
    print('  1 bit 密钥变化后位置不变比例 = %.4f (随机期望 %.4f)'
          % (r3['frac_positions_unchanged_after_1bit_key_change'], 1.0 / r3['block_size']))
    print('  置换不动点比例 = %.4f' % r3['frac_fixed_points'])
    out['T3_perm_sensitivity'] = r3

    with open(os.path.join(OUT, 'extra_tests.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\nsaved -> %s' % os.path.join(OUT, 'extra_tests.json'))
