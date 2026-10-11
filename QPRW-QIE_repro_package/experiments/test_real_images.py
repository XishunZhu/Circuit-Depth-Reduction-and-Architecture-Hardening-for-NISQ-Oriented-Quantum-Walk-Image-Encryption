# -*- coding: utf-8 -*-
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _HERE  # 包根; 所有绝对路径改为按脚本位置推导

"""对 D:\ZXS\量子加密 中的四副真实图像运行 QPRW-QIE 加密算法测试。

流程:
  1. 读取图像 -> 灰度 -> 双三次插值缩放为 256x256 uint8;
  2. 主算法 encrypt/decrypt, 验证无损恢复 (MSE=0);
  3. 计算密文信息熵、相邻像素相关性、密钥敏感度 NPCR/UACI、
     明文雪崩 NPCR/UACI、直方图卡方均匀性;
  4. 输出 results_real/metrics.json, 并保存密文/解密图用于论文插图。
"""
import os
import sys
import json
import time
import math
import numpy as np
from PIL import Image

sys.path.insert(0, ROOT)
from qprw_qie import encrypt, decrypt  # noqa

SRC = r'D:\ZXS\量子加密'
OUT = os.path.join(ROOT, 'results_real')
os.makedirs(OUT, exist_ok=True)

MASTER = ('a1b2c3d4e5f60718293a4b5c6d7e8f90'
          + '1029384756aabbccddeeff0011223344')
# 仅末位翻转 1 bit 的对抗密钥
MASTER2 = ('a1b2c3d4e5f60718293a4b5c6d7e8f90'
           + '1029384756aabbccddeeff0011223345')

IMAGES = [
    'airplane_original.png',
    'baboon_original.png',
    'peppers_original.png',
    'secret_epoch041_label[0 1]_batch5214.png',
]


def load_gray256(path):
    im = Image.open(path).convert('L')
    im = im.resize((256, 256), Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def shannon_entropy(img):
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    return float(-np.sum(p * np.log2(p)))


def chi_square_uniformity(img):
    """密文直方图均匀性卡方统计量 (自由度 255); 越小越接近均匀。"""
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    expected = img.size / 256.0
    return float(np.sum((hist - expected) ** 2 / expected))


def adjacency_correlation(img):
    """密文相邻像素线性相关系数: 水平 / 垂直 / 对角。"""
    a = img.astype(np.float64)

    def corr(x, y):
        mx, my = x.mean(), y.mean()
        cov = np.mean((x - mx) * (y - my))
        sx, sy = x.std(), y.std()
        if sx < 1e-12 or sy < 1e-12:
            return 0.0
        return float(cov / (sx * sy))

    h = corr(a[:, :-1].ravel(), a[:, 1:].ravel())
    v = corr(a[:-1, :].ravel(), a[1:, :].ravel())
    d = corr(a[:-1, :-1].ravel(), a[1:, 1:].ravel())
    return h, v, d


def npcr_uaci(a, b):
    a = a.ravel().astype(np.int64)
    b = b.ravel().astype(np.int64)
    npcr = float(np.sum(a != b) / a.size * 100.0)
    uaci = float(np.mean(np.abs(a - b)) / 255.0 * 100.0)
    return npcr, uaci


def plain_entropy(img):
    return shannon_entropy(img)


def main():
    results = {}
    summary_fig = {}
    for name in IMAGES:
        path = os.path.join(SRC, name)
        img = load_gray256(path)
        H, W = img.shape

        t0 = time.time()
        C, aux = encrypt(img, MASTER)
        te = time.time() - t0

        D = decrypt(C, MASTER, aux)
        mse = float(np.mean((img.astype(np.int64) - D.astype(np.int64)) ** 2))
        psnr = 1e12 if mse == 0 else float(10.0 * math.log10(255.0 ** 2 / mse))

        # 密钥敏感度: 1-bit 密钥翻转后的密文差异
        C2, _ = encrypt(img, MASTER2)
        key_npcr, key_uaci = npcr_uaci(C.ravel(), C2.ravel())

        # 明文雪崩: 单像素翻转后的密文差异 (同密钥)
        img2 = img.copy()
        img2.flat[0] ^= 1
        C3, _ = encrypt(img2, MASTER)
        pl_npcr, pl_uaci = npcr_uaci(C.ravel(), C3.ravel())

        # 明文指标
        p_ent = plain_entropy(img)
        p_h, p_v, p_d = adjacency_correlation(img)
        # 密文指标
        c_ent = shannon_entropy(C)
        c_h, c_v, c_d = adjacency_correlation(C)
        chi = chi_square_uniformity(C)

        # 保存插图: 原图 / 密文 / 解密图
        base = name.rsplit('.', 1)[0]
        Image.fromarray(img).save(os.path.join(OUT, base + '_plain.png'))
        Image.fromarray(C).save(os.path.join(OUT, base + '_cipher.png'))
        Image.fromarray(D).save(os.path.join(OUT, base + '_decrypt.png'))

        results[base] = {
            'file': name,
            'shape': [H, W],
            'encrypt_time_s': round(te, 3),
            'lossless': bool(mse == 0),
            'mse': mse,
            'psnr_db': psnr if psnr < 1e9 else 999.0,
            'plain_entropy': p_ent,
            'plain_corr_h': p_h, 'plain_corr_v': p_v, 'plain_corr_d': p_d,
            'cipher_entropy': c_ent,
            'cipher_corr_h': c_h, 'cipher_corr_v': c_v, 'cipher_corr_d': c_d,
            'cipher_chi2': chi,
            'key_sensitivity_npcr': key_npcr, 'key_sensitivity_uaci': key_uaci,
            'plain_avalanche_npcr': pl_npcr, 'plain_avalanche_uaci': pl_uaci,
        }
        summary_fig[base] = {'plain': p_ent, 'cipher': c_ent,
                             'corr_plain': (p_h + p_v + p_d) / 3.0,
                             'corr_cipher': (c_h + c_v + c_d) / 3.0}
        print('%-40s ent_plain=%.4f ent_cipher=%.4f corr_h/v/d=%.4f/%.4f/%.4f '
              'keyNPCR=%.3f MSE=%.0f t=%.2fs'
              % (base, p_ent, c_ent, c_h, c_v, c_d, key_npcr, mse, te))

    with open(os.path.join(OUT, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 跨图像平均
    n = len(results)
    avg = {k: round(sum(r[k] for r in results.values()) / n, 4)
           for k in ['encrypt_time_s', 'mse', 'psnr_db', 'plain_entropy',
                     'cipher_entropy', 'cipher_corr_h', 'cipher_corr_v',
                     'cipher_corr_d', 'cipher_chi2', 'key_sensitivity_npcr',
                     'key_sensitivity_uaci', 'plain_avalanche_npcr',
                     'plain_avalanche_uaci', 'plain_corr_h', 'plain_corr_v',
                     'plain_corr_d']}
    avg['lossless_all'] = all(r['lossless'] for r in results.values())
    results['_average'] = avg
    with open(os.path.join(OUT, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('\nAVERAGE:', json.dumps(avg, ensure_ascii=False))


if __name__ == '__main__':
    main()
