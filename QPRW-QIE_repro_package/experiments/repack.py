# -*- coding: utf-8 -*-
"""QPRW-QIE / NISQ 量子图像加密 · 复现代码包 重新打包脚本 (idempotent).

产出: <project>/QPRW-QIE_repro_package/  (扁平布局, 与原始工作目录等价, 仅剔除临时/调试脚本)
  - 核心实现 / 安全分析 / 实验驱动 全部置于包根, 便于兄弟模块互相 import
  - 硬编码绝对路径 (D:\\dejunpeng\\...) 改写为按脚本位置推导的 ROOT
  - 从匿名定稿 docx 逐位抽取 10 张图到 figures/
  - 复现结果 results_v3/ 与无版权争议的测试图 images/ 一并归档
  - 终稿论文 (匿名中文定稿 + 英文译本) 置于 paper/
运行: python repack.py
"""
import os
import re
import sys
import json
import shutil
import zipfile

SRC = r"D:\dejunpeng\quantum_image_encryption"
PKG = os.path.join(SRC, "QPRW-QIE_repro_package")
FINAL_DOCX = os.path.join(SRC, "paper_build", "NISQ量子图像加密电路优化论文_中文扩充版_定稿.docx")
EN_DOCX = os.path.join(SRC, "paper_build", "NISQ_QuantumWalk_Image_Encryption_English_Final.docx")

# (源相对路径, 包内相对路径) ; 扁平: 均放包根
CORE = [
    "qprw_qie.py", "qprw_qie_opt.py", "qprw_qie_v2.py",
    "qprw_quantum_sim.py", "qprw_circuit_draw.py", "color_qie.py",
]
SECURITY = [
    "attacks.py", "nist_suite.py", "nist_control.py", "nq_noise.py", "extra_tests.py",
]
EXPERIMENTS = [
    "run_experiments.py", "run_v2.py", "run_v3.py",
    "test_real_images.py", "test_color_real.py",
]
PAPER_SCRIPTS = ["_build_paper_cn.py", "_build_paper_en_final.py"]
FIG_SCRIPTS = ["_figs_v3.py", "qprw_plot.py"]
DOCS = [
    "QPRW-QIE_NISQ优化实现报告.md",
    "QPRW-QIE_量子模拟实现报告.md",
]

FAILS = []

# 路径改写规则 (长字面量优先, 避免被更短规则抢走)
PATH_RULES = [
    # 1) results_real 长后缀 (先于裸根规则)
    (r"r'D:\\dejunpeng\\quantum_image_encryption\\results_real'",
     r"os.path.join(ROOT, 'results_real')"),
    (r'r"D:\\dejunpeng\\quantum_image_encryption\\results_real"',
     r"os.path.join(ROOT, 'results_real')"),
    # 2) 图 4 可视化散落路径
    (r"r'D:\\dejunpeng\\paper_figs_en\\fig4_src_visualization\.png'",
     r"os.path.join(ROOT, 'figures', 'fig4_src_visualization.png')"),
    # 3) 裸根 (sys.path.insert)
    (r"r'D:\\dejunpeng\\quantum_image_encryption'",
     r"ROOT"),
    (r'r"D:\\dejunpeng\\quantum_image_encryption"',
     r"ROOT"),
]
# 论文/图 构建脚本的目录改写 (在拷贝后单独处理, 不走 ROOT)
BUILD_RULES = [
    (r'PAPERS = os.path.join\(BASE, "paper_build"\)',
     'PAPERS = os.path.join(BASE, "paper")'),
    (r'OUTD = os.path.join\(BASE, "figs_dinggao_ordered"\)',
     'OUTD = os.path.join(BASE, "results_v3")'),
]

ROOT_BLOCK = (
    "\nimport os\n"
    "_HERE = os.path.dirname(os.path.abspath(__file__))\n"
    "ROOT = _HERE  # 包根; 所有绝对路径改为按脚本位置推导\n"
)


def needs_root(text):
    return ("ROOT" in text) and ("ROOT = " not in text)


def copy_and_rewrite(rel_src, dest, build_tweak=False):
    src = os.path.join(SRC, rel_src)
    if not os.path.isfile(src):
        FAILS.append("MISSING SRC " + rel_src)
        return
    text = open(src, encoding="utf-8").read()
    # 路径改写
    for pat, rep in PATH_RULES:
        text = re.sub(pat, rep, text)
    if build_tweak:
        for pat, rep in BUILD_RULES:
            text = re.sub(pat, rep, text)
    # 注入 ROOT 定义
    if needs_root(text):
        # 放到首行(编码声明)之后, 否则放最前
        m = re.search(r"^#.*coding[:=].*$", text, flags=re.M)
        if m:
            idx = m.end()
            text = text[:idx] + ROOT_BLOCK + text[idx:]
        else:
            text = ROOT_BLOCK.lstrip("\n") + text
    # 残留绝对路径自检
    if "dejunpeng" in text:
        FAILS.append("RESIDUAL ABS PATH " + rel_src)
    if re.search(r"\bROOT[\\/]['\"]", text):
        FAILS.append("BROKEN ROOT JOIN " + rel_src)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    open(dest, "w", encoding="utf-8").write(text)
    print("  copied+rewrote:", rel_src)


def extract_figures():
    """从定稿 docx 按文档流抽取全部图到 figures/, 命名 fig01..figNN。"""
    import docx
    from docx.oxml.ns import qn
    BLIP = qn("a:blip")                       # DrawingML (PNG/JPG)
    VML = "{urn:schemas-microsoft-com:vml}imagedata"   # 旧式 VML (EMF/WMF)
    RID = qn("r:embed")
    VID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    d = docx.Document(FINAL_DOCX)
    order = []
    for child in d.element.body.iterchildren():
        if child.tag == qn("w:p"):
            for blip in child.findall(".//" + BLIP):
                order.append(blip.get(RID))
            for imd in child.findall(".//" + VML):
                order.append(imd.get(VID))
    out_dir = os.path.join(PKG, "figures")
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for rid in order:
        if not rid:
            continue
        part = d.part.rels[rid].target_part
        ext = part.partname.split(".")[-1].split("/")[-1]
        fn = os.path.join(out_dir, "fig%02d.%s" % (n + 1, ext))
        open(fn, "wb").write(part.blob)
        n += 1
    print("  extracted %d figures -> figures/" % n)
    return n


def main():
    # 覆盖式打包: 仅 makedirs, 不删除已有文件 (避免误删整包)
    os.makedirs(PKG, exist_ok=True)

    print("== core ==")
    for f in CORE:
        copy_and_rewrite(f, os.path.join(PKG, f))
    print("== security ==")
    for f in SECURITY:
        copy_and_rewrite(f, os.path.join(PKG, f))
    print("== experiments ==")
    for f in EXPERIMENTS:
        copy_and_rewrite(f, os.path.join(PKG, f))
    print("== paper build scripts ==")
    for f in PAPER_SCRIPTS:
        copy_and_rewrite(f, os.path.join(PKG, f), build_tweak=True)
    print("== figure scripts ==")
    for f in FIG_SCRIPTS:
        copy_and_rewrite(f, os.path.join(PKG, f))

    # 数据: 测试图 (剔除 lena 版权图) + 复现结果
    print("== images (copyright-free) ==")
    img_dst = os.path.join(PKG, "images")
    os.makedirs(img_dst, exist_ok=True)
    cnt = 0
    for root, dirs, files in os.walk(os.path.join(SRC, "images")):
        for fn in files:
            if "lena" in fn.lower():
                continue
            rel = os.path.relpath(os.path.join(root, fn), os.path.join(SRC, "images"))
            dst = os.path.join(img_dst, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(os.path.join(root, fn), dst)
            cnt += 1
    print("  copied %d images (lena excluded)" % cnt)

    print("== results_v3 ==")
    res_dst = os.path.join(PKG, "results_v3")
    if os.path.isdir(os.path.join(SRC, "results_v3")):
        # dirs_exist_ok: 覆盖式拷贝, 不删除旧文件
        shutil.copytree(os.path.join(SRC, "results_v3"), res_dst, dirs_exist_ok=True)
        print("  copied results_v3/ (%d files)" % len(os.listdir(res_dst)))
    else:
        FAILS.append("MISSING results_v3")

    # 散落量子结果图
    stray = os.path.join(SRC, "qprw_quantum_result.png")
    if os.path.isfile(stray):
        os.makedirs(os.path.join(PKG, "figures"), exist_ok=True)
        shutil.copy2(stray, os.path.join(PKG, "figures", "fig_extra_quantum_result.png"))

    # 论文图抽取
    print("== figures from 定稿 docx ==")
    extract_figures()

    # 终稿论文
    print("== paper ==")
    paper_dst = os.path.join(PKG, "paper")
    os.makedirs(paper_dst, exist_ok=True)
    for src in (FINAL_DOCX, EN_DOCX):
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(paper_dst, os.path.basename(src)))
            print("  copied", os.path.basename(src))
        else:
            FAILS.append("MISSING PAPER " + src)

    # 文档
    print("== docs ==")
    docs_dst = os.path.join(PKG, "docs")
    os.makedirs(docs_dst, exist_ok=True)
    for f in DOCS:
        s = os.path.join(SRC, f)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(docs_dst, f))
    write_readme()
    write_changelog()
    write_requirements()

    # 自包含: 把 repack.py 也放进包根
    shutil.copy2(os.path.abspath(__file__), os.path.join(PKG, "repack.py"))

    # 编译自检
    print("== py_compile ==")
    py_ok = 0
    for root, _, files in os.walk(PKG):
        for fn in files:
            if fn.endswith(".py"):
                p = os.path.join(root, fn)
                r = os.system('%s -m py_compile "%s"' % (sys.executable, p))
                if r == 0:
                    py_ok += 1
                else:
                    FAILS.append("PY_COMPILE_FAIL " + os.path.relpath(p, PKG))
    print("  compiled %d .py files" % py_ok)

    # 清单
    manifest = {}
    for name in ("core", "security", "experiments", "paper", "docs", "figures", "images", "results_v3"):
        d = os.path.join(PKG, name)
        if os.path.isdir(d):
            manifest[name] = sum(len(fs) for _, _, fs in os.walk(d))
    json.dump(manifest, open(os.path.join(PKG, "manifest.json"), "w"),
              ensure_ascii=False, indent=2)

    print("\n=== FAILS ===")
    print(FAILS if FAILS else "全部成功, 0 失败")
    print("package at:", PKG)


def write_readme():
    txt = """# QPRW-QIE 量子图像加密 · NISQ 电路优化 — 复现代码包

> 本包与论文 *NISQ 量子图像加密电路优化论文 (中文定稿)* / *NISQ Quantum-Walk Image
> Encryption — English (Final)* 配套。论文尚在审稿, 方案名已在正文中匿名处理。

## 目录结构

```
QPRW-QIE_repro_package/
├── qprw_qie.py            # 基线 QPRW-QIE 方案 (NEQR 编码 + 量子行走置换 + DNA 扩散)
├── qprw_qie_opt.py        # NISQ 优化 v1 (QFT 帧折叠 / ±1 方向融合 初版)
├── qprw_qie_v2.py         # NISQ 优化 v2 (受控增量器 + 编译期位平面重标记 + 置换种子封装)
├── qprw_quantum_sim.py    # 量子线路态矢量模拟 (Qiskit / NumPy)
├── qprw_circuit_draw.py   # 线路绘制
├── color_qie.py           # 彩色扩展
├── attacks.py             # 抗篡改 / 遮挡 / 噪声攻击
├── nist_suite.py          # NIST SP 800-22 随机性检验套件
├── nist_control.py        # NIST 对照 (理想随机源)
├── nq_noise.py            # NISQ 噪声模型与错误率上界评估
├── extra_tests.py         # 附加统计检验 (相关性 / 卡方 / 熵)
├── run_experiments.py     # 主实验驱动 (无版权图)
├── run_v2.py / run_v3.py  # v1 / v2 优化版实验驱动
├── test_real_images.py    # 真实灰度图验证
├── test_color_real.py     # 真实彩色图验证 (彩色扩展)
├── _figs_v3.py            # 论文图生成 (600 dpi)
├── qprw_plot.py           # 辅助作图
├── _build_paper_cn.py     # 中文定稿生成
├── _build_paper_en_final.py # 英文终稿生成
├── images/                # 无版权标准测试图 (已剔除 Lena)
├── results_v3/            # 复现结果 (6 幅无版权图, 64x64, 逐字节无损)
├── figures/               # 从定稿 docx 逐位抽取的论文图
├── paper/                 # 匿名中文定稿 + 英文终稿 docx
├── docs/                  # 实现报告 (量子模拟 / NISQ 优化)
├── repack.py              # 本打包脚本 (可重跑)
├── manifest.json          # 文件清单
├── README.md / CHANGELOG.md / requirements.txt
```

## 依赖

```
numpy, scipy, matplotlib, Pillow, python-docx, qiskit   # 量子模拟
```
建议: Python 3.10+, 见 requirements.txt。

## 复现步骤 (在包根执行)

1. 生成论文图:        `python _figs_v3.py`        (写入 results_v3/, 600 dpi)
2. 跑主实验:          `python run_v3.py`          (产出 results_v3/_results.json)
3. 统计与安全检验:    `python extra_tests.py` / `python nist_suite.py`
4. 生成论文 docx:     `python _build_paper_cn.py` / `python _build_paper_en_final.py`

> 所有绝对路径已改写为按脚本位置推导的 ROOT, 包可整体迁移到任意目录后直接运行。

## 关键指标 (v2 / Lake 实验)

- 两比特门数 3,809,336 -> 10,480; 允许两比特错误率上界 8.4e-6 -> 1.5e-3
- 6/6 逐字节无损; 密文熵 7.9498-7.9637; NPCR 99.90% / UACI 33.59%
- NIST SP 800-22: 3803/3850 = 98.8%
"""
    open(os.path.join(PKG, "README.md"), "w", encoding="utf-8").write(txt)


def write_changelog():
    txt = """# CHANGELOG

## 2026-10-11  v1 (复现包初版)
- 汇总量子图像加密实现: 基线 (qprw_qie) / NISQ 优化 v1 (qprw_qie_opt) / v2 (qprw_qie_v2)
- 量子模拟 (qprw_quantum_sim)、线路绘制 (qprw_circuit_draw)、彩色扩展 (color_qie)
- 安全分析: 攻击 (attacks)、NIST 套件 (nist_suite/nist_control)、噪声评估 (nq_noise)、附加检验 (extra_tests)
- 实验驱动: run_experiments / run_v2 / run_v3 / test_real_images / test_color_real
- 论文图生成 (_figs_v3, qprw_plot) 与论文构建 (_build_paper_cn / _build_paper_en_final)
- 数据: 无版权测试图 (剔除 Lena) images/ + 复现结果 results_v3/
- 论文图自定稿 docx 逐位抽取至 figures/
- 硬编码绝对路径改写为 ROOT (位置推导), 包可迁移
- 论文 (匿名中文定稿 + 英文终稿) 归入 paper/
"""
    open(os.path.join(PKG, "CHANGELOG.md"), "w", encoding="utf-8").write(txt)


def write_requirements():
    txt = (
        "numpy>=1.24\n"
        "scipy>=1.10\n"
        "matplotlib>=3.6\n"
        "Pillow>=9.2\n"
        "python-docx>=0.8.11\n"
        "qiskit>=0.45\n"
        "qiskit-aer>=0.12\n"
    )
    open(os.path.join(PKG, "requirements.txt"), "w", encoding="utf-8").write(txt)


if __name__ == "__main__":
    main()
