# -*- coding: utf-8 -*-
r"""
QPRW-QIE v2 全量实验（论文第 5–7 章数据来源）
输出: run_v2_log.txt  +  results_v2/_results.json
"""
import os, sys, math, time, json, hashlib
import numpy as np
from PIL import Image
from qiskit import transpile
import qprw_qie_v2 as V
import nist_suite as NS

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "results_v2")
os.makedirs(OUT, exist_ok=True)
HB = ["rz", "sx", "x", "cx"]
MASTER = b"QPRW-QIE-V2-MASTER-KEY-2026"
IMAGES = ["lena", "baboon", "peppers", "cameraman", "jetplane", "house"]
LOG = []
def log(s=""):
    LOG.append(str(s)); print(s)

def load_images():
    d = {}
    for nm in IMAGES:
        p = os.path.join(BASE, "images", "std", nm + "_512.png")
        im = Image.open(p).convert("L")
        if im.size != (V.H, V.W): im = im.resize((V.H, V.W), Image.BILINEAR)
        d[nm] = np.asarray(im, dtype=np.uint8)
    return d

def circ_stats(qc, olevel=0):
    t = transpile(qc, basis_gates=HB, optimization_level=olevel)
    o = t.count_ops()
    return dict(depth=t.depth(), cx=int(o.get("cx", 0)), total=int(sum(o.values())))

def main():
    t_all = time.time()
    R = {}
    imgs = load_images()
    log("=" * 100)
    log("QPRW-QIE v2 全量实验   图像集: %s   (%dx%d 灰度)" % (", ".join(IMAGES), V.H, V.W))
    log("=" * 100)

    # ============ 1. 行走步数 T 的裁剪实验 ============
    log("\n############ 1. 行走步数 T 的裁剪：安全指标是否保持 ############")
    log("  %-6s %-8s %-8s %-8s %-8s %-8s %-8s %-8s"
        % ("T", "往返", "密文熵", "NPCR%", "UACI%", "相关H", "chi^2", "加密s"))
    sweep = {}
    for T in (8, 16, 32, 64, 128):
        row = {}
        for nm in ("lena", "baboon"):
            im = imgs[nm]
            t0 = time.time()
            ct, hdr, tag = V.encrypt(im, MASTER, t=T, nonce=(b"N%03d" % T) + b"0" * 13)
            te = time.time() - t0
            back, ok = V.decrypt(ct, hdr, tag, MASTER)
            ci = ct.reshape(V.H, V.W)
            n, u = V.npcr_uaci(im, ci)
            row[nm] = dict(ok=bool(np.array_equal(back, im.reshape(-1))), ent=V.entropy_bits(ci),
                           npcr=n, uaci=u, corr=V.adj_corr(ci), chi=V.chi_square(ci), te=te, tag=ok)
        sweep[T] = row
        for nm in ("lena", "baboon"):
            r = row[nm]
            log("  %-6s %-8s %-8.4f %-8.2f %-8.2f %-8.4f %-8.1f %-8.2f"
                % (T if nm == "lena" else "", nm, r["ent"], r["npcr"], r["uaci"],
                   r["corr"][0], r["chi"], r["te"]))
    R["sweep_T"] = sweep
    T_STAR = 32
    log("  -> 选取 T* = %d（指标已饱和，再增大不再改善统计特性）" % T_STAR)

    # ============ 2. 主实验：6 张标准图像 ============
    log("\n############ 2. 标准图像集主实验 (T=%d, 折叠版) ############" % T_STAR)
    log("  %-11s %-9s %-9s %-8s %-8s %-9s %-9s %-8s %-8s %-8s"
        % ("图像", "明文熵", "密文熵", "NPCR%", "UACI%", "相关H", "chi^2", "往返", "标签", "加密s"))
    main_r = {}
    for nm in IMAGES:
        im = imgs[nm]
        t0 = time.time()
        ct, hdr, tag = V.encrypt(im, MASTER, t=T_STAR)
        te = time.time() - t0
        t0 = time.time()
        back, ok = V.decrypt(ct, hdr, tag, MASTER)
        td = time.time() - t0
        ci = ct.reshape(V.H, V.W)
        n, u = V.npcr_uaci(im, ci)
        main_r[nm] = dict(ent_p=V.entropy_bits(im), ent_c=V.entropy_bits(ci), npcr=n, uaci=u,
                          corr_p=V.adj_corr(im), corr_c=V.adj_corr(ci), chi_p=V.chi_square(im),
                          chi_c=V.chi_square(ci), ok=bool(np.array_equal(back, im.reshape(-1))),
                          tag=bool(ok), te=te, td=td)
        r = main_r[nm]
        log("  %-11s %-9.4f %-9.4f %-8.2f %-8.2f %-9.4f %-9.1f %-8s %-8s %-8.2f"
            % (nm, r["ent_p"], r["ent_c"], n, u, r["corr_c"][0], r["chi_c"],
               "OK" if r["ok"] else "FAIL", "OK" if r["tag"] else "FAIL", te))
        Image.fromarray(ci).save(os.path.join(OUT, "ct_%s.png" % nm))
        Image.fromarray(im).save(os.path.join(OUT, "pt_%s.png" % nm))
        np.save(os.path.join(OUT, "ct_%s.npy" % nm), ci)
    log("  相邻像素相关（明文 -> 密文，水平/垂直/对角）:")
    for nm in IMAGES:
        r = main_r[nm]
        log("    %-11s [%.4f %.4f %.4f] -> [%.4f %.4f %.4f]"
            % (nm, r["corr_p"][0], r["corr_p"][1], r["corr_p"][2],
               r["corr_c"][0], r["corr_c"][1], r["corr_c"][2]))
    R["main"] = main_r

    # ============ 3. 敏感性 ============
    log("\n############ 3. 密钥 / nonce / 明文 敏感性 ############")
    im = imgs["lena"]
    nonce0 = b"FIXED-NONCE-0001"
    ct0, hdr0, tag0 = V.encrypt(im, MASTER, t=T_STAR, nonce=nonce0)
    # (a) 主密钥改 1 bit
    mk2 = bytearray(MASTER); mk2[0] ^= 1
    ct_k, _, _ = V.encrypt(im, bytes(mk2), t=T_STAR, nonce=nonce0)
    n, u = V.npcr_uaci(ct0, ct_k)
    log("  主密钥改 1 bit        : NPCR=%.2f%%  UACI=%.2f%%" % (n, u))
    # (b) nonce 改 1 bit（同一明文同一密钥的两次加密）
    nonce1 = bytearray(nonce0); nonce1[0] ^= 1
    ct_n, _, _ = V.encrypt(im, MASTER, t=T_STAR, nonce=bytes(nonce1))
    n2, u2 = V.npcr_uaci(ct0, ct_n)
    log("  nonce 改 1 bit       : NPCR=%.2f%%  UACI=%.2f%%" % (n2, u2))
    # (c) 明文改 1 bit（nonce 固定 -> 纯靠扩散）
    im2 = im.copy().reshape(-1); im2[V.NPIX // 2] ^= 1; im2 = im2.reshape(V.H, V.W)
    ct_p, _, _ = V.encrypt(im2, MASTER, t=T_STAR, nonce=nonce0)
    n3, u3 = V.npcr_uaci(ct0, ct_p)
    log("  明文改 1 bit (nonce固定): NPCR=%.2f%%  UACI=%.2f%%  <- 扩散层独立贡献" % (n3, u3))
    # (d) 解密用错密钥
    back_wrong, ok_wrong = V.decrypt(ct0, hdr0, tag0, bytes(mk2))
    n4, u4 = V.npcr_uaci(im, back_wrong.reshape(V.H, V.W))
    log("  用错误密钥解密        : 与明文 NPCR=%.2f%%  UACI=%.2f%%  标签校验=%s"
        % (n4, u4, ok_wrong))
    R["sens"] = dict(key=(n, u), nonce=(n2, u2), plain=(n3, u3), wrong=(n4, u4))

    # ============ 3b. 扩散层改进对照 ============
    log("\n############ 3b. 扩散层：XOR 双向链 vs 模加+S盒 双向链 (BCD-A) ############")
    rho, dy, dx, iv0, iv1 = V.derive_params(MASTER, nonce0)
    ks0 = V._keystream(MASTER, nonce0)
    s2 = V.qpr_forward(im.reshape(-1), rho, dy, dx)
    s2b = s2.copy(); s2b[1000] ^= 1
    diff_rows = {}
    for name, fn in (("XOR 双向链(v1)", V.bcd_forward_xor), ("模加+S盒双向链(BCD-A)", V.bcd_forward)):
        o1 = fn(s2, ks0["ks1"], ks0["ks2"], iv0, iv1)
        o2 = fn(s2b, ks0["ks1"], ks0["ks2"], iv0, iv1)
        d = 100.0 * (o1 != o2).mean()
        diff_rows[name] = d
        log("  %-22s 输入改 1 字节 -> 输出改变 %.2f%%" % (name, d))
    log("  说明: 纯 XOR 链的差分按奇偶位置交替抵消 (c[i]=r[i]⊕c[i-1])，扩散被截断；"
        "模加+S盒使差分单调传播。")
    R["bcd_cmp"] = diff_rows

    # ============ 4. 攻击测试 ============
    log("\n############ 4. 密文篡改 / 遮挡 / 噪声攻击 ############")
    log("  %-16s %-10s %-10s %-10s %-10s" % ("攻击", "PSNR(dB)", "NPCR%", "标签检出", "备注"))
    att = {}
    def report(name, ct_atk, note=""):
        back, ok = V.decrypt(ct_atk, hdr0, tag0, MASTER)
        bi = back.reshape(V.H, V.W)
        ps = V.psnr(im, bi)
        nn, _ = V.npcr_uaci(im, bi)
        log("  %-16s %-10.2f %-10.2f %-10s %-10s" % (name, ps, nn, "是" if not ok else "否", note))
        att[name] = dict(psnr=ps, npcr=nn, detected=bool(not ok))
    # (a) 单比特翻转
    c = ct0.copy(); c[100] ^= 1; report("单比特翻转", c)
    # (b) 16x16 遮挡（置零）
    c = ct0.copy().reshape(V.H, V.W).copy()
    c[16:32, 16:32] = 0; report("16x16遮挡(1.6%)", c.reshape(-1), "密文块置零")
    # (c) 25% 裁剪
    c = ct0.copy().reshape(V.H, V.W).copy()
    c[32:, :] = 0; report("裁剪25%", c.reshape(-1))
    # (d) 1% 椒盐噪声
    rng = np.random.default_rng(1)
    c = ct0.copy()
    idx = rng.choice(V.NPIX, V.NPIX // 100, replace=False)
    c[idx] = rng.choice(np.array([0, 255], dtype=np.uint8), len(idx))
    report("椒盐噪声1%", c)
    # (e) 5% 椒盐噪声
    c = ct0.copy()
    idx = rng.choice(V.NPIX, V.NPIX // 20, replace=False)
    c[idx] = rng.choice(np.array([0, 255], dtype=np.uint8), len(idx))
    report("椒盐噪声5%", c)
    R["attacks"] = att

    # ============ 5. 已知/选择明文 ============
    log("\n############ 5. 已知明文 / 选择明文 检验 ############")
    special = {"全0图像": np.zeros((V.H, V.W), dtype=np.uint8),
               "全255图像": np.full((V.H, V.W), 255, dtype=np.uint8),
               "棋盘图像": np.indices((V.H, V.W)).sum(axis=0).astype(np.uint8) % 2 * 255}
    kpa = {}
    for nm, im_s in special.items():
        ct_s, _, _ = V.encrypt(im_s, MASTER, t=T_STAR, nonce=nonce0)
        ci = ct_s.reshape(V.H, V.W)
        n, u = V.npcr_uaci(ct0, ci)
        kpa[nm] = dict(ent=V.entropy_bits(ci), npcr_vs_lena=n, uaci_vs_lena=u,
                       corr=V.adj_corr(ci), chi=V.chi_square(ci))
        log("  %-10s 密文熵=%.4f  chi^2=%.1f  相关H=%.4f  与lena密文 NPCR=%.2f%%"
            % (nm, kpa[nm]["ent"], kpa[nm]["chi"], kpa[nm]["corr"][0], n))
    R["kpa"] = kpa

    # ============ 6. NIST 随机性检验 ============
    log("\n############ 6. NIST SP 800-22 随机性检验（聚合比特流） ############")
    target_bits = 2_000_000
    n_ct = int(target_bits // (V.NPIX * 8)) + 1
    buf = bytearray()
    names = IMAGES * ((n_ct // len(IMAGES)) + 1)
    t0 = time.time()
    for i in range(n_ct):
        im_i = imgs[names[i % len(names)]]
        ct_i, _, _ = V.encrypt(im_i, MASTER, t=T_STAR, nonce=(b"NB%05d" % i) + b"0" * 11)
        buf += ct_i.tobytes()
    bits = NS.bits_from_bytes(bytes(buf))
    log("  聚合 %d 个密文 -> %d bit (%.2f Mbit)，生成耗时 %.1f s"
        % (n_ct, bits.size, bits.size / 1e6, time.time() - t0))
    per = 262144
    res, nseq, per_eff = NS.run_800_22(bits, n_per_seq=per, n_seq=bits.size // per)
    log("  序列数=%d, 每序列=%d bit, 显著水平 α=0.01" % (nseq, per_eff))
    log("  %-26s %-10s %-10s %-10s" % ("检验项", "通过/总数", "通过率", "平均p"))
    npass = ntot = 0
    for k, v in res.items():
        if v is None:
            log("  %-26s %-10s" % (k, "N/A")); continue
        log("  %-26s %-10s %-10.3f %-10.4f" % (k, "%d/%d" % (v["pass"], v["n"]),
                                               v["prop"], v["mean_p"]))
        npass += v["pass"]; ntot += v["n"]
    log("  合计: %d/%d 通过 (%.1f%%)" % (npass, ntot, 100.0 * npass / max(ntot, 1)))
    def _nist_item(v):
        if v is None:
            return None
        return dict(npass=int(v["pass"]), n=int(v["n"]), mean_p=float(v["mean_p"]))
    R["nist"] = {k: _nist_item(v) for k, v in res.items()}
    R["nist_total"] = [npass, ntot]

    # ============ 7. 资源核算 ============
    log("\n############ 7. 量子资源核算（硬件基 %s） ############" % HB)
    rng = np.random.default_rng(3)
    th128 = V.THETA_LO + (V.THETA_HI - V.THETA_LO) * rng.random(128)
    th32 = th128[:T_STAR]
    items = {}
    items["QHRNG(20q)"] = circ_stats(V.build_qhrng_circuit(
        [np.zeros(16)] * 3) if False else V.build_qhrng_circuit(
        [V.bytes_to_floats(V.sha3_ctr(b"a", 16)) * math.pi] * 3), 3)
    for lvl in (0, 3):
        items["QWP折叠 T=%d/块(lvl%d)" % (T_STAR, lvl)] = circ_stats(
            V.walk_folded(th32, 1.0, T_STAR, V.NPOS_QW, save=False), lvl)
        items["QWP每步QFT T=128/块(lvl%d)" % lvl] = circ_stats(
            V.walk_perstep(th128, 1.0, 128, V.NPOS_QW, save=False), lvl)
        items["QWP-MCX T=128/块(lvl%d)" % lvl] = circ_stats(
            V.walk_mcx(th128, 1.0, 128, V.NPOS_QW, save=False), lvl)
        items["QPR正/逆(20q,lvl%d)" % lvl] = circ_stats(V.build_qpr_circuit(False), lvl)
    log("  %-28s %-10s %-10s %-10s" % ("电路", "深度", "CX", "总门数"))
    for k, v in items.items():
        log("  %-28s %-10d %-10d %-10d" % (k, v["depth"], v["cx"], v["total"]))

    qpr_cx = items["QPR正/逆(20q,lvl0)"]["cx"]
    qpr_d = items["QPR正/逆(20q,lvl0)"]["depth"]
    fold32 = items["QWP折叠 T=%d/块(lvl0)" % T_STAR]
    ps128 = items["QWP每步QFT T=128/块(lvl0)"]
    mcx128 = items["QWP-MCX T=128/块(lvl0)"]
    tot_new = V.NBLK * fold32["cx"] + qpr_cx            # 加密(行走) + 加/解密 QPR 各一次
    tot_new_dec = qpr_cx                                # 解密侧只需 QPR 逆
    tot_ps = V.NBLK * ps128["cx"] + qpr_cx
    tot_mcx = V.NBLK * mcx128["cx"] + qpr_cx
    log("\n  单张 64x64 图像的两比特门预算（lvl0，结构口径）:")
    log("    v1 原实现  (MCX, T=128, 加解密各跑一次行走): CX = %s"
        % format(2 * V.NBLK * mcx128["cx"] + qpr_cx, ","))
    log("    v1 每步QFT (T=128, 加解密各跑一次行走)     : CX = %s"
        % format(2 * V.NBLK * ps128["cx"] + qpr_cx, ","))
    log("    v2 折叠    (T=%d, 仅加密跑行走)            : CX = %s  (加密 %s + 解密 %s)"
        % (T_STAR, format(tot_new + tot_new_dec, ","),
           format(V.NBLK * fold32["cx"] + qpr_cx, ","), format(qpr_cx, ",")))
    log("    -> 相对 v1 原实现 %.1fx；相对每步QFT 版 %.1fx"
        % ((2 * V.NBLK * mcx128["cx"] + qpr_cx) / (tot_new + tot_new_dec),
           (2 * V.NBLK * ps128["cx"] + qpr_cx) / (tot_new + tot_new_dec)))
    log("  单块最大电路深度: 折叠T=%d -> %d (v1 MCX T=128 -> %d, 每步QFT -> %d)"
        % (T_STAR, fold32["depth"], mcx128["depth"], ps128["depth"]))
    log("  使\"期望错误数<1\"所需的双比特错误率上界: v2 = %.2e, 每步QFT = %.2e, MCX = %.2e"
        % (1.0 / fold32["cx"], 1.0 / ps128["cx"], 1.0 / mcx128["cx"]))
    R["res"] = dict(items=items, tot_new=tot_new + tot_new_dec,
                    tot_ps=2 * V.NBLK * ps128["cx"] + qpr_cx,
                    tot_mcx=2 * V.NBLK * mcx128["cx"] + qpr_cx,
                    fold_depth=fold32["depth"], fold_cx=fold32["cx"],
                    qpr_cx=qpr_cx, qpr_depth=qpr_d)

    # ============ 8. 头部开销 ============
    log("\n############ 8. 头部开销与吞吐 ############")
    log("  密文头 = nonce(%dB) + 封装置换种子(%dB) + MAC(%dB) = %dB, 占 64x64 载荷的 %.2f%%"
        % (V.NONCE_LEN, V.SEED_LEN, V.TAG_LEN, V.HDR_LEN + V.TAG_LEN,
           100.0 * (V.HDR_LEN + V.TAG_LEN) / V.NPIX))
    tes = [main_r[k]["te"] for k in main_r]; tds = [main_r[k]["td"] for k in main_r]
    log("  加密(含量子行走+QHRNG) 平均 %.3f s/图；解密(无需量子行走) 平均 %.3f s/图"
        % (sum(tes) / len(tes), sum(tds) / len(tds)))
    log("  吞吐: 加密 %.1f B/s, 解密 %.1f B/s (纯 Python + 状态矢量模拟)"
        % (V.NPIX / (sum(tes) / len(tes)), V.NPIX / (sum(tds) / len(tds))))
    R["timing"] = dict(enc=sum(tes) / len(tes), dec=sum(tds) / len(tds))

    log("\n总耗时 %.1f s" % (time.time() - t_all))
    with open(os.path.join(BASE, "run_v2_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))
    with open(os.path.join(OUT, "_results.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=1, default=lambda o: (
            o.tolist() if isinstance(o, np.ndarray) else
            float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
    np.save(os.path.join(OUT, "_cipher_bits.npy"), bits[:1_000_000].astype(np.uint8))

if __name__ == "__main__":
    main()
