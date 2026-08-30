# 本轮实验结果索引

## 结论边界

当前最有论文价值的是 **Learn2Reg MR–CT CT→MR public-8 protocol_300 多方法
对比**。它是唯一按统一 8 对病例、统一四器官指标和统一缺失标签规则完成的
完整对比结果。B10/PRA-CM 虽然作为 `ours` 行保留在表中，但结果是负向对照，
不能宣称实现了目标创新。

## 正式多方法结果

完整归档的绝对路径：

`E:\00-Code\00-Chat-code\KidneyCT_Registration_Response\results\L2R_MRCT_protocol300_20260825`

主要文件：

- `aggregate\HEADLINE_BENCHMARK.md`：可读主表；
- `aggregate\headline_benchmark_summary.csv`：13 行机器可读主表；
- `aggregate\pair_metrics_all.csv`：13×8 个 pair 级结果；
- `aggregate\organ_metrics_all.csv`：13×32 个器官-病例结果；
- `aggregate\verification_report.json`：完整性检查，13/13 通过；
- `aggregate\METRIC_CORRECTION.json`：缺失左肾标注的校正记录；
- `archive_checksums.sha256`：服务器传输包 SHA-256；
- `extracted\server1` / `extracted\server2`：逐方法 evaluation、flow、status、
  训练运行记录、TensorBoard 和可用 checkpoint；
- `downloads\server1` / `downloads\server2`：原始服务器归档包。

服务器分工：server1 为 SSH 33601，保存 B00、B02、B04–B10；server2 为 SSH
46608，保存 B01、B03、B11、B12。两台服务器现已关闭，以下路径是本地归档路径，
不是仍在线的服务器路径。

### 主表（校正后）

| ID | 方法 | Mean Dice ↑ | ASSD (mm) ↓ | HD95 (mm) ↓ | Fold fraction ↓ | 来源 |
|---|---|---:|---:|---:|---:|---|
| B00 | Identity | 0.370861 | 15.631536 | 39.433018 | 0.000000 | 33601 |
| B01 | ANTs-SyN + MI | 0.574181 | 10.317572 | 28.637102 | 0.000398 | 46608 |
| B02 | ConvexAdam + MIND-SSC | 0.735027 | 6.296807 | 20.377894 | 0.015578 | 33601 |
| B03 | FireANTs | 0.471178 | 13.510936 | 36.014302 | 0.000000 | 46608 |
| B04 | DINO-Reg | 0.782112 | 4.962758 | 23.348518 | 0.041090 | 33601 |
| B05 | MASR/DNS + IO | 0.380776 | 15.041330 | 38.751084 | 0.003903 | 33601 |
| B06 | SynMSE | 0.394881 | 14.956584 | 38.531678 | 0.014497 | 33601 |
| B07 | Locor | 0.627334 | 11.811766 | 30.928099 | 0.000000 | 33601 |
| B08 | DGMIR-U | 0.361762 | 16.335685 | 42.208488 | 0.005154 | 33601 |
| B09 | M2M-Reg | 0.394367 | 13.989197 | 38.449040 | 0.000039 | 33601 |
| B10 | PRA-CM v3（ours） | 0.412384 | 15.299623 | 39.364444 | 0.000647 | 33601 |
| B11 | TransMorph + MIND-SSC | 0.397730 | 13.912598 | 36.358340 | 0.040049 | 46608 |
| B12 | CorrMLP + MIND-SSC | 0.499682 | 13.033869 | 36.711645 | 0.024432 | 46608 |

每个方法均有 8 个 pair 调用、8 个成功状态和 8 个 flow；四器官×八病例共 32
个请求器官病例，其中 `0002`、`0004` 的 left-kidney reference 缺失，因此 30
个可评估、2 个明确排除。ASSD、HD95 和 Dice 均遵循同一器官等权宏平均规则。

## 自研方法的非主结果

这些记录保留用于追溯和解释失败原因，不应与正式多方法表合并：

1. `E:\00-Code\00-Chat-code\KidneyCT_Registration_Response\results\PRA_CM_v4_L2R_corrected_20260825\HEADLINE_V4_CORRECTED.csv`
   - V4-A relational：Mean Dice `0.408678`；
   - V4-B hierarchical：Mean Dice `0.377378`；
   - V4-Full interim best：Mean Dice `0.403136`。
2. `E:\00-Code\00-Chat-code\KidneyCT_Registration_Response\results\v4_corefix_20260826\solver_diagnostics.json`
   只包含合成平移/非刚体 QA，不是 public-8 论文结果。
3. `E:\00-Code\00-Chat-code\KidneyCT_Registration_Response\releases\v4_core_0p663361_frozen_20260827`
   是冻结的 V4-Core 负向基线（Mean Dice `0.663362`），不能与 B04 的外部 anchor
   或 V5 路由结果混称为独立自研结果。

## 无关或历史结果

`results\plc_r_paper_v1` 属于另一条 PLC-R 研究线，不属于本次 MR–CT
protocol_300 对比；`experiments`、`tmp` 和根目录部署包均为过程材料，已在索引中
与正式结果分离。
