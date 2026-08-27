# Registration V4-Core（冻结版本）

当前分支为 `registration_v4_core`，保存第一份成功并冻结的忠实 DSIR / V4-Core
实现及其 public-8 结果。主代码是 `code/registration_v4_final`；
`code/registration_v4_pracm` 仅作为明确记录的内部依赖保留。

## 方法身份

V4-Core 采用输入分辨率的 Figure-9 风格特征提取、direct + dilated DNS、
24 通道 feature-squeezing DSIR，以及 descriptor-agnostic ConvexAdam 求解链。
完整实现说明见
[`code/registration_v4_final/README.md`](code/registration_v4_final/README.md)。

## 冻结 public-8 结果

| 指标 | 数值 |
|---|---:|
| Mean Dice ↑ | `0.6633615210` |
| ASSD (mm) ↓ | `8.234422` |
| HD95 (mm) ↓ | `27.164855` |
| Fold fraction ↓ | `0.000059382` |

- 最佳验证 checkpoint：epoch `217`
- 完整训练：epoch `299/299`
- 冻结 checkpoint SHA-256：`6ba1c54ab260f4fb830b019caeaaf8414c1b45aae435adb4ff9eb68592d5bb70`

冻结源代码、checkpoint、服务器路径和哈希清单见
[`releases/v4_core_0p663361_frozen_20260827`](releases/v4_core_0p663361_frozen_20260827)。
该目录是不可变基线，不应被后续实验覆盖。

## 主要入口

- 主代码：[`code/registration_v4_final`](code/registration_v4_final)
- 内部依赖：[`code/registration_v4_pracm`](code/registration_v4_pracm)
- 分支清单：[`git_manifests/registration_v4_core.md`](git_manifests/registration_v4_core.md)
- 冻结发布说明：[`releases/v4_core_0p663361_frozen_20260827/README.md`](releases/v4_core_0p663361_frozen_20260827/README.md)

本分支不代表 V5 的 DINO anchor、dense-corefix 路由或 `0.785949746` 结果。
当前最佳完整版本请切换到
[`registration_v5`](https://github.com/Mister-Ryder/registration/tree/registration_v5)。

