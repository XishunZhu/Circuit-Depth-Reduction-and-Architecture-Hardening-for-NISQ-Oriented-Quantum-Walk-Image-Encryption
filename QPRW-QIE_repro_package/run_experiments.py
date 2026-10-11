# -*- coding: utf-8 -*-
"""实验: 在合成体模与真实 CT 图像上评估 QPRW-QIE、4 组消融与 3 条基线。"""
import io
import json
import os
import sys
import time
import hashlib

import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qprw_qie as Q  # noqa

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'results')
os.makedirs(OUT, exist_ok=True)
RNG = np.random.RandomState(20261006)
KEY = hashlib.sha3_256(b'QPRW-QIE-demo-master-key').hexdigest()   # 256-bit 主密钥
FULL = dict(qpr=True, qw_perm=True, two_way=True, dna=True)


# ---------------- 测试图像 ----------------
def shepp_logan(n=256):
    E = [(1.0, .69, .92, 0, 0, 0), (-.8, .6624, .8740, 0, -.0184, 0),
         (-.2, .1100, .3100, .22, 0, -18), (-.2, .1600, .4100, -.22, 0, 18),
         (.1, .2100, .2500, 0, .35, 0), (.1, .0460, .0460, 0, .1, 0),
         (.1, .0460, .0460, 0, -.1, 0), (.1, .0460, .0230, -.08, -.605, 0),
         (.1, .0230, .0230, 0, -.606, 0), (.1, .0230, .0460, .06, -.605, 0)]
    y, x = np.mgrid[-1:1:complex(0, n), -1:1:complex(0, n)]
    img = np.zeros((n, n))
    for A, a, b, x0, y0, phi in E:
        t = np.deg2rad(phi)
        xr = (x - x0) * np.cos(t) + (y - y0) * np.sin(t)
        yr = -(x - x0) * np.sin(t) + (y - y0) * np.cos(t)
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1] += A
    return (img / img.max() * 255.0).astype(np.uint8)


def ct_patch(n=256, path=os.path.join(ROOT, 'figures', 'fig4_src_visualization.png')):
    a = np.asarray(Image.open(path).convert('L'), dtype=np.float64)
    step, k = 8, n // 8
    d = a[::step, ::step]
    hh, ww = d.shape
    s1 = np.cumsum(np.cumsum(np.r_[np.zeros((1, ww)), d], 0), 1)
    s2 = np.cumsum(np.cumsum(np.r_[np.zeros((1, ww)), d * d], 0), 1)

    def box(i, j, h, w):
        A = s1[i + h, j + w] - s1[i, j + w] - s1[i + h, j] + s1[i, j]
        B = s2[i + h, j + w] - s2[i, j + w] - s2[i + h, j] + s2[i, j]
        return B / (h * w) - (A / (h * w)) ** 2

    best, bi, bj = -1.0, 0, 0
    for i in range(0, hh - k, 4):
        for j in range(0, ww - k, 4):
            v = box(i, j, k, k)
            if v > best:
                best, bi, bj = v, i, j
    crop = a[bi * step:bi * step + n, bj * step:bj * step + n]
    if crop.shape != (n, n):
        crop = np.asarray(Image.fromarray(a.astype(np.uint8)).crop(
            (bj * step, bi * step, bj * step + n, bi * step + n)), dtype=np.float64)
    return ((crop - crop.min()) / max(crop.max() - crop.min(), 1e-6) * 255.0).astype(np.uint8)


# ---------------- 指标 ----------------
def entropy(img):
    h = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def chi_square(img):
    h = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    e = h.sum() / 256.0
    return float(((h - e) ** 2 / e).sum())


def correlation(img, nsamples=6000):
    h, w = img.shape
    a = img.ravel().astype(np.float64)
    ys = RNG.randint(0, h - 1, nsamples)
    xs = RNG.randint(0, w - 1, nsamples)
    idx = ys * w + xs

    def corr(i1, i2):
        u, v = a[i1], a[i2]
        su, sv = u.std(), v.std()
        if su < 1e-9 or sv < 1e-9:
            return 0.0
        return float(((u - u.mean()) * (v - v.mean())).mean() / (su * sv))

    return dict(H=corr(idx, idx + 1), V=corr(idx, idx + w), D=corr(idx, idx + w + 1))


def npcr_uaci(c1, c2):
    d = (c1 != c2)
    return (float(100.0 * d.sum() / d.size),
            float(100.0 * np.abs(c1.astype(np.int16) - c2.astype(np.int16)).mean() / 255.0))


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def local_entropy(img, bs=32):
    h, w = img.shape
    return float(np.mean([entropy(img[i:i + bs, j:j + bs])
                          for i in range(0, h, bs) for j in range(0, w, bs)]))


def eval_cfg(plain, enc_f, t_enc=0.0, dec_plain=None, dec_t=0.0, key_f=None):
    C = enc_f(plain)
    r = dict(entropy=entropy(C), local_entropy=local_entropy(C), chi2=chi_square(C),
             corr=correlation(C), psnr_vs_plain=psnr(plain, C),
             time_s=round(t_enc, 3), dec_time_s=round(dec_t, 3))
    p2 = plain.copy()
    p2[plain.shape[0] // 2, plain.shape[1] // 2] ^= 1
    r['npcr'], r['uaci'] = npcr_uaci(C, enc_f(p2))
    if key_f is not None:
        k2 = ('f' if KEY[0] != 'f' else 'e') + KEY[1:]
        r['key_npcr'], r['key_uaci'] = npcr_uaci(key_f(KEY), key_f(k2))
    if dec_plain is not None:
        r['decrypt_mse'] = float(np.mean((plain.astype(float) - dec_plain.astype(float)) ** 2))
        r['decrypt_exact'] = bool(np.array_equal(plain, dec_plain))
    return r, C


# ---------------- 主流程 ----------------
def main():
    report = {}
    lam = Q.smcm_lyapunov()
    print('SMCM Lyapunov = %s' % np.round(lam, 4))
    X, Y = Q.smcm(200000, 0.37, 0.61)
    KB = Q.smcm_bytes(X)
    h = np.bincount(KB, minlength=256).astype(float)
    report['chaos'] = dict(
        lyapunov=[float(v) for v in lam],
        keystream_chi2_255df=float(((h - h.sum() / 256.) ** 2 / (h.sum() / 256.)).sum()),
        keystream_autocorr=float(np.corrcoef(KB[:-1].astype(float), KB[1:].astype(float))[0, 1]),
        keystream_entropy=entropy(KB))

    images = {'phantom_shepp_logan': shepp_logan(256)}
    try:
        images['ct_lung_patch'] = ct_patch(256)
    except Exception as e:
        print('ct_patch failed:', e)

    CONFIGS = [
        ('QPRW-QIE(full)', dict(FULL), None),
        ('A1 w/o QPR', dict(FULL, qpr=False), None),
        ('A2 single-chain', dict(FULL, two_way=False), None),
        ('A3 w/o DNA', dict(FULL, dna=False), None),
        ('A4 w/o QW-perm', dict(FULL, qw_perm=False), None),
        ('B0 QW-ks+XOR', None, 'b0'),
        ('B1 Arnold+chain', None, 'b1'),
        ('B2 QW-ks+chain', None, 'b2'),
    ]

    for iname, img in images.items():
        print('\n==== %s  (plain entropy=%.4f, corrH=%+.4f) ===='
              % (iname, entropy(img), correlation(img)['H']))
        Image.fromarray(img).save(os.path.join(OUT, 'plain_%s.png' % iname))
        d = {'plain_stats': dict(entropy=entropy(img), corr=correlation(img),
                                 chi2=chi_square(img))}
        for name, cfg, base in CONFIGS:
            if base is None:
                t0 = time.time()
                C, aux = Q.encrypt(img, KEY, cfg)
                t1 = time.time()
                D = Q.decrypt(C, KEY, aux)
                t2 = time.time()
                r, _ = eval_cfg(img, lambda p: Q.encrypt(p, KEY, cfg)[0], t1 - t0,
                                dec_plain=D, dec_t=t2 - t1,
                                key_f=lambda k: Q.encrypt(img, k, cfg)[0])
                r['cfg'] = cfg
            else:
                f = {'b0': Q.baseline_qw_xor, 'b1': Q.baseline_arnold_chain,
                     'b2': Q.baseline_qw_chain}[base]
                t0 = time.time()
                C = f(img, KEY)
                t1 = time.time()
                r, _ = eval_cfg(img, lambda p: f(p, KEY), t1 - t0,
                                key_f=lambda k: f(img, k))
            d[name] = r
            print('  %-16s H=%.4f chi2=%9.1f corrH=%+.4f NPCR=%6.3f UACI=%5.2f '
                  'keyNPCR=%6.3f  t=%5.2fs%s'
                  % (name, r['entropy'], r['chi2'], r['corr']['H'], r['npcr'], r['uaci'],
                     r.get('key_npcr', float('nan')), r['time_s'],
                     '  exact=%s' % r.get('decrypt_exact') if 'decrypt_exact' in r else ''))
            if name == 'QPRW-QIE(full)':
                Image.fromarray(C).save(os.path.join(OUT, 'cipher_%s.png' % iname))
                # 篡改定位
                Cm = C.copy()
                Cm[16:48, 16:48] = 0
                okv, bad = Q.verify_tag(Cm, Q.recover_key_bits(KEY, aux['T']),
                                        aux['tag'], aux['block'])
                r['tamper_detected'] = bool(not okv)
                r['tamper_blocks_flagged'] = len(bad)
                r['n_auth_blocks'] = len(aux['tag'])
                # 遮挡鲁棒性
                Co = C.copy()
                Co[100:132, 100:132] = 0
                Do = Q.decrypt(Co, KEY, aux)
                r['occlusion_psnr'] = psnr(img, Do)
                print('      篡改检测=%s (%d/%d 块) | 32x32 遮挡后解密 PSNR=%.2f dB'
                      % (r['tamper_detected'], r['tamper_blocks_flagged'],
                         r['n_auth_blocks'], r['occlusion_psnr']))
        report[iname] = d

    # ---------------- QW 硬币角安全性扫描 (B2 结构) ----------------
    print('\n==== QW 硬币角扫描: 定位使 QW 掩模退化的参数区 ====')
    img = images[list(images.keys())[0]]
    sweep = []
    for frac in [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        P = Q.dtqw_cycle(Q.NQ_DEFAULT, Q.TQ_DEFAULT, frac * np.pi, avg_last=Q.AVG_LAST)
        supp = float((P > 1e-15).sum()) / P.size          # 有效支撑比例
        conc = float(P.max() * P.size)                     # 峰值/均值
        K = Q.smcm_bytes(np.repeat(np.linspace(0, 1, 8), 1))
        # 用固定角构造 B2 结构密钥流并评估
        def enc_fixed(p, frac=frac):
            N = p.size
            Ps, Ks = [], []
            for b in range(int(np.ceil(N / float(Q.NQ_DEFAULT)))):
                Pb = Q.dtqw_cycle(Q.NQ_DEFAULT, Q.TQ_DEFAULT, frac * np.pi,
                                  avg_last=Q.AVG_LAST)
                sd = hashlib.sha3_256(Pb.tobytes() + bytes([b & 255])).digest()
                Xb, _ = Q.smcm(Q.NQ_DEFAULT, sd[0] / 255., sd[1] / 255., discard=300)
                Ks.append(Q.smcm_bytes(Xb))
            Kk = np.concatenate(Ks)[:N]
            return Q.chain_forward(p.ravel(), Kk, 7).reshape(p.shape)
        c = enc_fixed(img)
        p2 = img.copy()
        p2[128, 128] ^= 1
        npc, uac = npcr_uaci(c, enc_fixed(p2))
        sweep.append(dict(theta_over_pi=frac, mask_support=supp, mask_peak_over_mean=conc,
                          entropy=entropy(c), npcr=npc, uaci=uac,
                          corr_H=correlation(c)['H']))
        print('  θ=%.2fπ  支撑=%.2f 峰值/均值=%7.1f  H=%.4f  NPCR=%6.3f  UACI=%5.2f'
              % (frac, supp, conc, sweep[-1]['entropy'], npc, uac))
    report['qw_coin_sweep'] = sweep

    with open(os.path.join(OUT, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print('\nsaved -> %s' % os.path.join(OUT, 'metrics.json'))


if __name__ == '__main__':
    main()
