# -*- coding: utf-8 -*-
"""生成 QPRW-QIE 量子加密结果可视化图（复用 qprw_quantum_sim 的实现）"""
import os, sys, math, hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qprw_quantum_sim as Q

H, W, NPIX = Q.H, Q.W, Q.NPIX
npos = int(round(math.log2(NPIX)))

img = Q.make_test_image()
flat0 = img.reshape(-1).astype(np.uint8)
H_plain = hashlib.sha3_256(flat0.tobytes()).digest()
SEED = Q.MASTER_KEY + H_plain
ks = Q.sha3_keystream(SEED, 4096)
f = Q.bytes_to_floats(ks)
rho = int(f[0] * 8) % Q.NPLANE
dy = (f[1:9] * H).astype(int) % H
dx = (f[9:17] * W).astype(int) % W

flat_qpr, _ = Q.qpr_quantum(flat0, rho, dy, dx, npos)

nblk = NPIX // Q.L_BLOCK
nq_pos = int(round(math.log2(Q.L_BLOCK)))
phi0 = f[17]
seq = flat_qpr.copy()
P0 = None
for b in range(nblk):
    blk = seq[b * Q.L_BLOCK:(b + 1) * Q.L_BLOCK]
    kb = Q.sha3_keystream(SEED + b"|blk|" + b.to_bytes(4, "big") + bytes(blk[:8]), 4096)
    fb = Q.bytes_to_floats(kb)
    th = Q.THETA_LO + (Q.THETA_HI - Q.THETA_LO) * fb[:Q.T_STEPS]
    phi = (phi0 + b * 0.6180339887498949) % 1.0 * 2 * math.pi
    P = Q.dtqw_distribution(th, phi, nq_pos)
    if b == 0:
        P0 = P.copy()
    seq[b * Q.L_BLOCK:(b + 1) * Q.L_BLOCK] = blk[Q.perm_from_scores(P, kb[2048:2048 + Q.L_BLOCK])]

kd = Q.sha3_keystream(SEED + b"|diff", NPIX * 3)
rules = (np.frombuffer(Q.sha3_keystream(SEED + b"|rule", NPIX), dtype=np.uint8) % 8)
cipher = Q.dna_layer(Q.bcd_forward(seq, kd[:NPIX], kd[NPIX:2 * NPIX], int(kd[0]), int(kd[1])),
                     kd[2 * NPIX:3 * NPIX], rules)

fig, ax = plt.subplots(2, 3, figsize=(14, 8))
ax[0, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
ax[0, 0].set_title("Plaintext (32x32)")
ax[0, 1].imshow(cipher.reshape(H, W), cmap="gray", vmin=0, vmax=255)
ax[0, 1].set_title("Ciphertext (QPR+QWP+BCD+DNA)")
ax[0, 2].plot(P0, lw=0.8)
ax[0, 2].set_title("DTQW time-averaged P(x), block 0\n(t=128, avg last 32 steps)")
ax[0, 2].set_xlabel("ring position x"); ax[0, 2].set_ylabel("P(x)")

ax[1, 0].hist(flat0, bins=64, color="steelblue")
ax[1, 0].set_title("Plaintext histogram\nentropy=%.3f" % Q.entropy_bits(flat0))
ax[1, 1].hist(cipher, bins=64, color="indianred")
ax[1, 1].set_title("Ciphertext histogram\nentropy=%.3f" % Q.entropy_bits(cipher))

a = img.astype(np.float64)
c = cipher.reshape(H, W).astype(np.float64)
ax[1, 2].scatter(np.roll(a, 1, axis=1).ravel()[::2], a.ravel()[::2], s=3, alpha=0.35,
                 label="plain", color="steelblue")
ax[1, 2].scatter(np.roll(c, 1, axis=1).ravel()[::2], c.ravel()[::2], s=3, alpha=0.35,
                 label="cipher", color="indianred")
ax[1, 2].set_title("Horizontal adjacent-pixel correlation")
ax[1, 2].legend()
ax[1, 2].set_xlabel("neighbour"); ax[1, 2].set_ylabel("pixel")

for a2 in ax.ravel():
    a2.grid(alpha=0.25)
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qprw_quantum_result.png")
plt.savefig(out, dpi=110)
print("saved", out)
