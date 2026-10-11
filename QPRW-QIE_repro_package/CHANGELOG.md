# CHANGELOG

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
