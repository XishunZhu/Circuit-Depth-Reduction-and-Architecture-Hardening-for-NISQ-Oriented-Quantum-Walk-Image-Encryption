# -*- coding: utf-8 -*-
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _HERE  # 包根; 所有绝对路径改为按脚本位置推导

"""对 D:\\ZXS\\量子加密 中的四幅真实彩色图像运行彩色扩展 QPRW-QIE 测试。

流程:
  1. 读取真彩 RGB 图像, 分别以原生 512x512 与缩放 256x256 两种规格测试;
  2. color_encrypt/color_decrypt (逐通道 QPR + CC-Rotor 跨通道耦合 + 白化密钥流),
     验证无损恢复 (MSE=0);
  3. 逐通道信息熵 / 卡方 / 相关性, 明文雪崩与密钥敏感度 NPCR/UACI;
  4. 输出 results_real/color_metrics.json, 保存明文/密文/解密图供论文插图。
"""
import io
import json
import math
import os
import sys
import time

import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, ROOT)
from color_qie import color_encrypt, color_decrypt  # noqa

SRC = r'D:\ZXS\量子加密'
OUT = os.path.join(ROOT, 'results_real')
os.makedirs(OUT, exist_ok=True)

MASTER = ('a1b2c3d4e5f60718293a4b5c6d7e8f90'
          + '1029384756aabbccddeeff0011223345')
MASTER2 = ('a1b2c3d4e5f60718293a4b5c6d7e8f90'
           + '1029384756aabbccddeeff0011223344')   # 1-bit 对抗密钥

IMAGES = [
    'airplane_original.png',
    'baboon_original.png',
    'peppers_original.png',
    'secret_epoch041_label[0 1]_batch5214.png',
]


def load_rgb(path, size=None):
    im = Image.open(path).convert('RGB')
    if size:
        im = im.resize((size, size), Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def entropy(x):
    h = np.bincount(x.ravel(), minlength=256).astype(np.float64)
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def chi2(x):
    h = np.bincount(x.ravel(), minlength=256).astype(np.float64)
    e = h.sum() / 256.0
    return float(((h - e) ** 2 / e).sum())


def corr3(x):
    a = x.astype(np.float64)
    h, w = a.shape

    def c(u, v):
        su, sv = u.std(), v.std()
        if su < 1e-12 or sv < 1e-12:
            return 0.0
        return float(np.mean((u - u.mean()) * (v - v.mean())) / (su * sv))

    return dict(H=c(a[:, :-1], a[:, 1:]), V=c(a[:-1, :], a[1:, :]),
                D=c(a[:-1, :-1], a[1:, 1:]))


def npcr_uaci(a, b):
    a = a.ravel()
    b = b.ravel()
    return (float(100.0 * np.sum(a != b) / a.size),
            float(100.0 * np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))) / 255.0))


def test_one(name, size, save_prefix=None):
    img = load_rgb(os.path.join(SRC, name), size)
    H, W, _ = img.shape
    t0 = time.time()
    C, aux = color_encrypt(img, MASTER)
    te = time.time() - t0
    D = color_decrypt(C, MASTER, aux)
    mse = float(np.mean((img.astype(np.int64) - D.astype(np.int64)) ** 2))

    # 明文雪崩 (单像素 R 通道 LSB 翻转)
    img2 = img.copy()
    img2[H // 2, W // 2, 0] ^= 1
    C2, _ = color_encrypt(img2, MASTER)
    pl_npcr, pl_uaci = npcr_uaci(C, C2)

    # 密钥敏感度
    Ck, _ = color_encrypt(img, MASTER2)
    k_npcr, k_uaci = npcr_uaci(C, Ck)

    per_ch = {}
    for ch, nm in enumerate('RGB'):
        per_ch[nm] = dict(entropy=entropy(C[:, :, ch]), chi2=chi2(C[:, :, ch]),
                          corr=corr3(C[:, :, ch]),
                          plain_entropy=entropy(img[:, :, ch]))
    res = dict(file=name, size=H, encrypt_time_s=round(te, 2), mse=mse,
               lossless=bool(mse == 0),
               plain_entropy=entropy(img), plain_corr=corr3(
                   np.asarray(Image.fromarray(img).convert('L'))),
               avalanche_npcr=pl_npcr, avalanche_uaci=pl_uaci,
               key_npcr=k_npcr, key_uaci=k_uaci,
               per_channel=per_ch)
    if save_prefix:
        Image.fromarray(img).save(os.path.join(OUT, save_prefix + '_plain.png'))
        Image.fromarray(C).save(os.path.join(OUT, save_prefix + '_cipher.png'))
        Image.fromarray(D).save(os.path.join(OUT, save_prefix + '_decrypt.png'))
    return res


def main():
    results = {}
    # 1) 原生 512x512 真彩: 四幅全测并保存插图
    for name in IMAGES:
        base = name.rsplit('.', 1)[0]
        r = test_one(name, None, save_prefix='color_' + base)
        results['%s@512' % base] = r
        print('%s@512 ent=%.6f/%.6f/%.6f mse=%.0f NPCR=%.3f keyNPCR=%.3f t=%.1fs'
              % (base, r['per_channel']['R']['entropy'], r['per_channel']['G']['entropy'],
                 r['per_channel']['B']['entropy'], r['mse'], r['avalanche_npcr'],
                 r['key_npcr'], r['encrypt_time_s']))
    # 2) 256x256 (与灰度节同规格): baboon
    r = test_one('baboon_original.png', 256)
    results['baboon_original@256'] = r
    print('baboon@256 ent=%.6f/%.6f/%.6f mse=%.0f t=%.1fs'
          % (r['per_channel']['R']['entropy'], r['per_channel']['G']['entropy'],
             r['per_channel']['B']['entropy'], r['mse'], r['encrypt_time_s']))

    with open(os.path.join(OUT, 'color_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('saved color_metrics.json')


if __name__ == '__main__':
    main()
