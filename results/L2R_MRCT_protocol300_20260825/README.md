# Learn2Reg MR–CT protocol_300 结果副本

这是从本地校验归档中抽出的轻量、可审阅版本，覆盖 B00–B12 共 13 个方法。
原始完整归档仍保留在：

`E:\00-Code\00-Chat-code\KidneyCT_Registration_Response\results\L2R_MRCT_protocol300_20260825`

## 内容

- `aggregate/HEADLINE_BENCHMARK.md`：四项指标主表；
- `aggregate/headline_benchmark_summary.csv`：机器可读主表；
- `aggregate/pair_metrics_all.csv`：全部 pair 级指标；
- `aggregate/organ_metrics_all.csv`：全部器官-病例指标；
- `aggregate/verification_report.json`：13/13 方法完整性通过；
- `aggregate/METRIC_CORRECTION.json`：缺失左肾标注的校正审计；
- `methods/<method_id>/`：每个方法的 `evaluation.json`、`summary.csv`、
  `pair_metrics.csv`、`organ_metrics.csv`；
- `manifests/`：本轮 public-8 pair 和 label 清单；
- `archive_checksums.sha256`：原始服务器传输包的校验值。

轻量副本不包含大体积 NIfTI、flow、checkpoint 和 TensorBoard 文件；这些仍在上方
完整归档中。缺失 left-kidney reference 的 `0002`、`0004` 两项按协议排除，而不是
记为 Dice=0。

四项指标均是校正后的正式值：Mean Dice、ASSD、HD95 和 fold fraction。每个方法
均为 8/8 pair、8/8 flow、8/8 成功状态；32 个请求器官病例中 30 个可评估。
