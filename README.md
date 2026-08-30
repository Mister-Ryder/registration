# Learn2Reg MR–CT protocol_300 结果分支

当前分支 `results_l2r_protocol300` 只保存本轮正式的 B00–B12 public-8
多方法对比结果。它不保存训练代码，也不把 V4/V5 自研探索冒充为成功方法。

## 结果入口

- 轻量结果包：[`results/L2R_MRCT_protocol300_20260825`](results/L2R_MRCT_protocol300_20260825)
- 四项指标主表：[`aggregate/HEADLINE_BENCHMARK.md`](results/L2R_MRCT_protocol300_20260825/aggregate/HEADLINE_BENCHMARK.md)
- 机器可读主表：[`aggregate/headline_benchmark_summary.csv`](results/L2R_MRCT_protocol300_20260825/aggregate/headline_benchmark_summary.csv)
- 全部结果索引：[`RESULTS_INDEX.md`](RESULTS_INDEX.md)

该轻量包包含主表、13 个方法的 summary/pair/organ/evaluation 文件、public-8
协议清单、校正记录和完整性报告。每个方法均通过 8/8 pair、8/8 flow、8/8
成功状态检查；32 个器官-病例请求中 30 个可评估。

## 指标边界

Mean Dice、ASSD、HD95 和 fold fraction 均按“缺失器官排除、四器官等权宏平均”
计算。`0002`、`0004` 的 left-kidney reference 缺失，不记为 Dice=0。
完整服务器导出、NIfTI/flow、checkpoint 和 TensorBoard 仍保留在本地原始归档，
没有复制进 GitHub 结果分支。

V4/V5 的失败和诊断路径见 `RESULTS_INDEX.md` 中的边界说明，不属于本分支的主表。
