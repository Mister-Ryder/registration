# Registration V4 PRA-CM

当前分支为 `registration_v4_pracm`，独立保存原始 3-D PRA-CM V4 开发线。
它不是 V4-Core，也不是后续 V5；三者的模型结构、候选空间、checkpoint 和结果身份
不能混用。

## 方法身份

本版本围绕以下对象构建原生三维配准：

- structural / response 双分支表征；
- 显式 `(2r+1)^3` 三维候选对应分布与循环搜索；
- 由候选后验直接得到的熵、峰值概率和位移方差；
- 多期 CT 星形协议与概率关系闭合，以及独立的 MR--CT 适配协议。

完整的方法约定、flow 语义、数据协议和命令见
[`code/registration_v4_pracm/README.md`](code/registration_v4_pracm/README.md)。

## 主要入口

- 主代码：[`code/registration_v4_pracm`](code/registration_v4_pracm)
- 配置：[`code/registration_v4_pracm/configs`](code/registration_v4_pracm/configs)
- 训练：`python -m registration_v4_pracm.scripts.train`
- 推理：`python -m registration_v4_pracm.scripts.infer`
- 回归测试：[`code/registration_v4_pracm/tests`](code/registration_v4_pracm/tests)
- 分支清单：[`git_manifests/registration_v4_pracm.md`](git_manifests/registration_v4_pracm.md)

## 结果边界

这是保留用于追溯设计来源的历史开发分支，不把 V4-Core 或 V5 的 public-8 指标、
checkpoint 或 flow 归入本方法。外部数据、服务器运行目录和基础模型权重均不在 Git 中。

其他版本请切换到
[`registration_v4_core`](https://github.com/Mister-Ryder/registration/tree/registration_v4_core)
或 [`registration_v5`](https://github.com/Mister-Ryder/registration/tree/registration_v5)。

