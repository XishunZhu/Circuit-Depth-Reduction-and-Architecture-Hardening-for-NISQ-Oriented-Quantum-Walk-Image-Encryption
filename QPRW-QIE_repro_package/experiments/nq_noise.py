# -*- coding: utf-8 -*-
"""
真机线路编译代价 + NISQ 去极化噪声评估 (QWP 相干性退化扫描)
================================================================
对应 report.md §7 后续工作 3: 全部结果为经典模拟, 需补 NISQ 噪声评估。

本模块做两件事:
  A. 线路编译代价: 给出 QWP 每段 (块) 的量子门数与电路深度量级估计
     (Coin 旋转门 + 条件位移 + QFT 对角化), 与 §3.8 对齐。
  B. NISQ 噪声评估: 用含"局部去极化噪声"的 DTQW 模拟器, 扫描噪声率 p,
     观察 QW 概率分布的支撑率 / 峰值-均值 随噪声退化的曲线。
     物理动机: 去极化破坏相干 -> 弹道扩散退化为扩散扩散 -> 环上只覆盖局部窗口,
     支撑率塌缩 (如同 §6 中 θ=0.5π 的退化, 但由噪声而非参数引起)。

由于本环境无 Qiskit, 采用"等价退相干模拟"而非真机编译; 这足以给出
QWP 对相干性的敏感度与可用噪声阈值 —— 这是 NISQ 评估的核心结论。
"""
import json
import math
import os

import numpy as np

from qprw_qie import NQ_DEFAULT, TQ_DEFAULT, AVG_LAST, smcm, smcm_bytes, _frac


# ---------------------------------------------------------------
# 用与 qw_schedule 完全一致的密钥派生子构造 θ (保证与真实算法可比)
# ---------------------------------------------------------------
def make_theta(key8, nq=NQ_DEFAULT, t=TQ_DEFAULT):
    A, _ = smcm(t, _frac(key8[0] / 255.0), _frac(key8[1] / 255.0))
    B, _ = smcm(nq, _frac(key8[4] / 255.0), _frac(key8[5] / 255.0))
    J, _ = smcm(nq, _frac(key8[6] / 255.0), _frac(key8[7] / 255.0))
    phi = (_frac(key8[3] / 255.0)) % 1.0
    theta = math.pi * (0.10 + 0.30 * np.mod(A[:, None] + B[None, :] + phi, 1.0))
    return theta, J


def perm_from_theta(theta, J, N, p, seed, n_realiz):
    P = dtqw_noisy(N, TQ_DEFAULT, theta, p, seed=seed, avg_last=AVG_LAST, n_realiz=n_realiz)
    return np.argsort(P + 1e-16 * J, kind='stable'), P


# ---------------------------------------------------------------
# A. 线路编译代价 (每段 / 每块的门数与深度)
# ---------------------------------------------------------------
def compile_cost(nq=NQ_DEFAULT, t=TQ_DEFAULT):
    """QWP 单块 (环大小 nq, t 步) 的编译代价估计。"""
    # Coin: 位置相关的旋转角 θ(x,s) 由 QFT + 受控旋转实现。
    # QFT 对角化后, 单步 Coin 可写成对角相位门, 深度 O(log nq)。
    qft_depth = int(math.ceil(math.log2(nq)))           # QFT 深度 ~ log nq
    coin_depth_per_step = qft_depth                     # 每步 Coin (对角化)
    shift_depth_per_step = 1                             # 条件位移 1 层
    step_depth = coin_depth_per_step + shift_depth_per_step
    total_depth = step_depth * t                         # 串行 t 步
    # 门数 (粗略): 每步 QFT 约 nq*log nq 个门 + 位移 nq 门
    gates_per_step = nq * qft_depth + nq
    total_gates = gates_per_step * t
    return {
        'nq': nq, 't': t,
        'qft_depth': qft_depth,
        'step_depth': step_depth,
        'total_depth_per_block': total_depth,
        'gates_per_step': gates_per_step,
        'total_gates_per_block': total_gates,
    }


# ---------------------------------------------------------------
# B. 含去极化噪声的 DTQW
# ---------------------------------------------------------------
def dtqw_noisy(N, t, theta, p, seed=0, avg_last=0, n_realiz=1):
    rng = np.random.default_rng(seed)
    th = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    acc_all = np.zeros(N)
    L = int(avg_last)
    for rz in range(n_realiz):
        psi = np.zeros((N, 2), dtype=np.complex128)
        psi[0, 0] = 1.0 / math.sqrt(2.0)
        psi[0, 1] = 1.0j / math.sqrt(2.0)
        acc = np.zeros(N)
        for s in range(t):
            a0 = np.roll(psi[:, 0], 1)
            a1 = np.roll(psi[:, 1], -1)
            tt = th[s] if th.size > 1 else float(th[0])
            co = np.cos(tt)
            si = np.sin(tt)
            psi[:, 0] = co * a0 - si * a1
            psi[:, 1] = si * a0 + co * a1
            if p > 0.0:
                # 局部去极化: 以概率 p 抹除每个位置硬币的相干 (保留概率质量)
                px = (np.abs(psi[:, 0]) ** 2 + np.abs(psi[:, 1]) ** 2)
                ph0 = rng.random(N) * 2.0 * math.pi
                ph1 = rng.random(N) * 2.0 * math.pi
                na = np.sqrt(px / 2.0) * np.exp(1j * ph0)
                nb = np.sqrt(px / 2.0) * np.exp(1j * ph1)
                mask = rng.random(N) < p
                psi[:, 0] = np.where(mask, na, psi[:, 0])
                psi[:, 1] = np.where(mask, nb, psi[:, 1])
            if L and s >= t - L:
                acc += np.abs(psi[:, 0]) ** 2 + np.abs(psi[:, 1]) ** 2
        P = (acc / L) if L else (np.abs(psi[:, 0]) ** 2 + np.abs(psi[:, 1]) ** 2)
        acc_all += P / (P.sum() + 1e-300)
    P = acc_all / n_realiz
    return P / (P.sum() + 1e-300)


def perm_sensitivity(key_a, key_b, N=NQ_DEFAULT, p=0.0, seed=0, n_realiz=2):
    """两密钥 (差 1 bit) 产生置换的位置不变比例。理想≈1/q=1/256=0.0039。"""
    th_a, J_a = make_theta(key_a, N)
    th_b, J_b = make_theta(key_b, N)
    Ja, _ = perm_from_theta(th_a, J_a, N, p, seed=seed, n_realiz=n_realiz)
    Jb, _ = perm_from_theta(th_b, J_b, N, p, seed=seed + 99, n_realiz=n_realiz)
    return float(np.mean(Ja == Jb)), Ja


def _perm_id_corr(perm):
    """置换 π 与恒等置换的相关性: 理想随机≈0, 结构化(经典游走)→高。"""
    j = np.arange(perm.size)
    u = (perm - perm.mean())
    v = (j - j.mean())
    denom = math.sqrt((u * u).sum() * (v * v).sum())
    return float((u * v).sum() / denom) if denom > 0 else 0.0


def classical_walk_metrics(N=NQ_DEFAULT, t=TQ_DEFAULT):
    """完全退相干极限: 经典随机游走 (±1, 各 0.5) 的概率分布与置换结构。"""
    p = np.zeros(N)
    p[0] = 1.0
    for _ in range(t):
        p = 0.5 * np.roll(p, 1) + 0.5 * np.roll(p, -1)   # 环上经典游走
    P = p / (p.sum() + 1e-300)
    perm = np.argsort(P + 1e-16 * np.arange(N))
    return dict(support=float(np.mean(P > 1e-6)), peak_over_mean=float(P.max() / (P.mean() + 1e-300)),
               perm_id_corr=_perm_id_corr(perm), label='classical (fully decohered)')


def metrics_at_p(p, N=NQ_DEFAULT, t=TQ_DEFAULT, n_realiz=6):
    key_a = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.uint8)
    key_b = key_a.copy()
    key_b[0] ^= 1                                   # 1 bit 密钥差
    th_a, J_a = make_theta(key_a, N)
    P = dtqw_noisy(N, t, th_a, p, seed=2026, avg_last=AVG_LAST, n_realiz=n_realiz)
    support = float(np.mean(P > 1e-6))
    peak_mean = float(P.max() / (P.mean() + 1e-300))
    sens, Ja = perm_sensitivity(key_a, key_b, N, p, seed=2026, n_realiz=2)
    return dict(p=p, support=support, peak_over_mean=peak_mean,
                perm_unchanged_frac=sens, perm_id_corr=_perm_id_corr(Ja))


def main():
    out = {}
    out['compile_cost'] = compile_cost()
    out['noise_sweep'] = []
    print('=== QWP 编译代价 (每块) ===')
    cc = out['compile_cost']
    print('  nq=%d t=%d  总深度=%d  总门数≈%d (QFT 对角化)'
          % (cc['nq'], cc['t'], cc['total_depth_per_block'], cc['total_gates_per_block']))

    print('=== 去极化噪声扫描 (支撑率塌缩 = 相干性退化) ===')
    for p in [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]:
        m = metrics_at_p(p)
        out['noise_sweep'].append(m)
        print('  p=%.3f  支撑率=%.3f  峰值/均值=%.2f  置换不变=%.4f  置换-恒等相关=%.3f'
              % (m['p'], m['support'], m['peak_over_mean'], m['perm_unchanged_frac'], m['perm_id_corr']))

    # 完全退相干基准 (经典随机游走)
    cw = classical_walk_metrics()
    out['classical_decohered_baseline'] = cw
    print('  经典游走(完全退相干): 支撑率=%.3f 峰值/均值=%.2f 置换-恒等相关=%.3f'
          % (cw['support'], cw['peak_over_mean'], cw['perm_id_corr']))

    # 退化阈值: 支撑率跌破 0.5 的噪声率
    thr = None
    for m in out['noise_sweep']:
        if m['support'] < 0.5:
            thr = m['p']
            break
    out['support_collapse_threshold_p'] = thr

    os.makedirs('results', exist_ok=True)
    with open('results/nq_noise.json', 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('  支撑率塌缩阈值 p ≈', thr)
    return out


if __name__ == '__main__':
    main()
