# -*- coding: utf-8 -*-
"""
NIST 控制实验：验证检验工具正确性 + 界定 Linear Complexity 失效的生成器特异性
================================================================================
背景：nist_suite.py 显示 SMCM 密钥流与 numpy PCG64 均匀基准都未通过 Linear Complexity，
但其余 14 项 800-22 全过、800-90B 最小熵≈7.96。为给出诚实解释，本脚本做三件事：

  (1) os.urandom 控制：CSPRNG（操作系统密码学级随机源，FIPS 140）跑全套 800-22。
      若它通过 Linear Complexity，则证明检验工具本身正确、LC 失效是"生成器特异性"而非
      实现缺陷。
  (2) numpy PCG64DXSM 控制：另一类快速 PRNG，观察其是否同样在 LC 上表现异常，
      以佐证"快速/非密码学生成器的 L 分布偏集中"是普遍现象。
  (3) QPRW-QIE 真实密钥流：用 qw_schedule 在真实 256-bit 主密钥下逐块 SHA3-256 重播种
      生成加密实际所用扩散密钥流（2×10^6 字节），再跑 800-22。该密钥流比裸 SMCM 原始
      序列多了 SHA3 重播种这一密码学级混合层，给出更贴近实机的随机性结论。
"""
import json
import os

import numpy as np

import nist_suite as ns
from qprw_qie import qw_schedule, NQ_DEFAULT, TQ_DEFAULT


def _fails_below(d, thr=0.90):
    return [k for k, v in d['tests'].items() if v and v['prop'] < thr]


def main():
    out = {}

    # (1) os.urandom —— CSPRNG 黄金基准
    raw = os.urandom(2_000_000)
    bits = ns.bits_from_bytes(np.frombuffer(raw, dtype=np.uint8))
    res, nseq, per = ns.run_800_22(bits)
    out['os_urandom_crypto'] = {'nbits': int(bits.size), 'n_seq': nseq,
                                'bits_per_seq': int(per), 'tests': res}
    out['os_urandom_800_90B'] = ns.estimate_min_entropy(
        np.frombuffer(raw, dtype=np.uint8))

    # (2) numpy PCG64DXSM —— 另一快速 PRNG
    rng = np.random.Generator(np.random.PCG64DXSM(20261006))
    raw2 = rng.integers(0, 256, size=2_000_000, dtype=np.uint8)
    bits2 = ns.bits_from_bytes(raw2)
    res2, nseq2, per2 = ns.run_800_22(bits2)
    out['numpy_PCG64DXSM'] = {'nbits': int(bits2.size), 'n_seq': nseq2,
                              'bits_per_seq': int(per2), 'tests': res2}

    # (3) QPRW-QIE 真实密钥流（逐块 SHA3 重播种）
    master_hex = 'a1b2c3d4e5f60718293a4b5c6d7e8f90' + '1029384756aabbccddeeff0011223344'
    key_bits = [b / 255.0 for b in bytes.fromhex(master_hex)]
    N = 512_000                      # 512K 位 = 1000 个 LC 块所需, 同时控制 qw_schedule 耗时
    _, ks = qw_schedule(N, key_bits, nq=NQ_DEFAULT, t=TQ_DEFAULT)
    ks = ks[:N].astype(np.uint8)
    ks_bits = ns.bits_from_bytes(ks)
    res3, nseq3, per3 = ns.run_800_22(ks_bits)
    out['QPRW_QIE_real_keystream'] = {'nbytes': int(ks.size), 'nbits': int(ks_bits.size),
                                      'n_seq': nseq3, 'bits_per_seq': int(per3), 'tests': res3}
    out['QPRW_QIE_real_keystream_800_90B'] = ns.estimate_min_entropy(ks)

    out['_summary'] = {
        'os_urandom_fails_below_0.90': _fails_below(out['os_urandom_crypto']),
        'PCG64DXSM_fails_below_0.90': _fails_below(out['numpy_PCG64DXSM']),
        'QPRW_real_keystream_fails_below_0.90': _fails_below(out['QPRW_QIE_real_keystream']),
    }

    os.makedirs('results', exist_ok=True)
    with open('results/nist_control.json', 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print('=== os.urandom (CSPRNG) 800-22 低于 0.90 的测试 (应无):',
          out['_summary']['os_urandom_fails_below_0.90'])
    print('=== numpy PCG64DXSM 低于 0.90 的测试:',
          out['_summary']['PCG64DXSM_fails_below_0.90'])
    print('=== QPRW-QIE 真实密钥流 低于 0.90 的测试:',
          out['_summary']['QPRW_real_keystream_fails_below_0.90'])
    b = out['os_urandom_800_90B']
    print('=== os.urandom 最小熵(byte, IID保守)=%.4f' % b['min_entropy_byte_iid'])
    b3 = out['QPRW_QIE_real_keystream_800_90B']
    print('=== QPRW-QIE 真实密钥流 最小熵(byte, IID保守)=%.4f' % b3['min_entropy_byte_iid'])
    return out


if __name__ == '__main__':
    main()
