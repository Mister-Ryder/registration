# Registration V5（当前最佳完整版本）

当前分支为 `registration_v5`。它把本轮最佳 MR--CT 配准方案整理为一个完整、
可审计的方法包，而不是继续依赖多个增量实验目录。

## 方法身份

V5 由三部分形成一条明确的信息链：

1. 冻结的官方 B04 DINO-Reg 产生全局可靠 anchor；
2. 冻结 V4-Core DSIR descriptor 与 dense-corefix 产生非刚性候选和 QA；
3. 无标签 capture/topology/forward-backward router 在整例层面选择候选，并按原字节
   保存最终 flow。

V5 自有代码全部位于 [`code/registration_v5`](code/registration_v5)，运行入口、
flow 约定、路由条件和外部依赖说明见
[`code/registration_v5/README.md`](code/registration_v5/README.md)。

## 冻结 public-8 结果

| 指标 | 数值 |
|---|---:|
| Mean Dice ↑ | `0.7859497458` |
| ASSD (mm) ↓ | `4.577276` |
| HD95 (mm) ↓ | `19.858701` |
| Fold fraction ↓ | `0.026441595` |

dense-corefix 用于病例 `0004`、`0014`，其余六例保留字节一致的 DINO anchor。
该结果是当前已完成版本中的最佳值，但仍明确低于 `0.80` 和项目目标 `0.85`。

## 代码与文档

- 完整代码：[`code/registration_v5`](code/registration_v5)
- 方法说明 TeX：[`code/registration_v5/docs/registration_v5_method_detail_v1.tex`](code/registration_v5/docs/registration_v5_method_detail_v1.tex)
- 编译版 PDF：[`code/registration_v5/docs/registration_v5_method_detail_v1.pdf`](code/registration_v5/docs/registration_v5_method_detail_v1.pdf)
- 版本身份：[`code/registration_v5/VERSION_MANIFEST.json`](code/registration_v5/VERSION_MANIFEST.json)
- 分支清单：[`git_manifests/registration_v5.md`](git_manifests/registration_v5.md)

官方 DINO-Reg 上游代码及 DINOv2 权重属于外部依赖，不能将其描述为 V5 新训练模型；
冻结 V4-Core checkpoint 也通过 SHA-256 固定，不能由新训练结果静默替换。

历史版本请切换到
[`registration_v4_pracm`](https://github.com/Mister-Ryder/registration/tree/registration_v4_pracm)
或 [`registration_v4_core`](https://github.com/Mister-Ryder/registration/tree/registration_v4_core)。

