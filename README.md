# Registration 方法版本仓库

本仓库用独立 Git 分支保存三个不同的方法身份。`main` 只负责版本导航，
不代表任何一次训练、推理或实验结果；查看和复现实验时，请先切换到对应分支。

| 分支 | 方法身份 | 主代码目录 | 冻结 public-8 结果 |
|---|---|---|---|
| [`registration_v4_pracm`](https://github.com/Mister-Ryder/registration/tree/registration_v4_pracm) | 原始 3-D PRA-CM V4 开发线 | `code/registration_v4_pracm` | 历史开发版本，不作为当前最佳结果 |
| [`registration_v4_core`](https://github.com/Mister-Ryder/registration/tree/registration_v4_core) | 忠实 DSIR / V4-Core 冻结线 | `code/registration_v4_final` | Dice `0.663361521` |
| [`registration_v5`](https://github.com/Mister-Ryder/registration/tree/registration_v5) | DINO anchor + dense-corefix + 无标签路由的完整版本 | `code/registration_v5` | Dice `0.785949746` |
| [`results_l2r_protocol300`](https://github.com/Mister-Ryder/registration/tree/results_l2r_protocol300) | Learn2Reg MR–CT B00–B12 正式对比结果 | `results/L2R_MRCT_protocol300_20260825` | 13 方法，8/8 完整性通过 |

每个方法分支的根 `README.md` 都只介绍该分支；完整的代码说明、运行入口、
指标和边界条件继续保存在对应方法目录中。分支关系与保存原则见
[`GIT_BRANCHES.md`](GIT_BRANCHES.md)。

数据集、服务器完整导出、临时探针、大型归档和外部基础模型权重不纳入 Git。
冻结 checkpoint 或结果快照仅在相应分支明确列出，并通过 manifest 与 SHA-256
固定身份，不能跨版本混用。

正式多方法对比的轻量结果分支为
[`results_l2r_protocol300`](https://github.com/Mister-Ryder/registration/tree/results_l2r_protocol300)，
本地完整归档和四项指标索引见 `deliverables\RESULTS_INDEX.md`。
