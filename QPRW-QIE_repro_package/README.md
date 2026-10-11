# QPRW-QIE 量子图像加密 · NISQ 电路优化 — 复现代码包

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
