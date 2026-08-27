# KidneyCT Registration Response

本项目包含多期相 CT 配准、增强响应分析及历史基线。论文升级方法已经独立实现为
`code/registration_v3`，当前 package version 为 `0.4.0`。

## 当前主入口

- 新方法说明与正式命令：[`code/registration_v3/README.md`](code/registration_v3/README.md)
- 正式 PLC 训练：`code/registration_v3/scripts/train_plc.py`
- 单对 NIfTI 推理：`code/registration_v3/scripts/infer_nifti.py`
- Evaluation-only 标签传播：`code/registration_v3/scripts/evaluate_labels.py`
- 冻结 PLC 审计契约：`data/PLC_PreRegistration_Analysis_v1`
- 历史 `DualREG_PatentExperts`：只保留为 baseline，不与 v3 checkpoint/flow cache 混用

Registration v3 已具备显式候选对应、response-conditioned matching、可选 P/A/V
relational training、三视图体推理、严格 flow provenance、checkpoint/resume、QA 与
独立标签评价。代码与合成/CUDA 回归已闭环，但尚未完成 PLC/WAW 全量训练，不能把当前
smoke 结果写成论文性能结果。

当前目录没有 `.git`；正式运行会冻结配置、输入文件哈希、split/exclusion identity、
源码树 SHA-256、RNG 及 last/best checkpoint。历史迁移清单见
[`PROJECT_INVENTORY.md`](PROJECT_INVENTORY.md)。

