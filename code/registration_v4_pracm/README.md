# PRA-CM v3：原生 3-D 相位响应感知 correspondence 配准

这是从 `code/registration_v3` v0.5.2 独立演进的新目录。旧代码和旧 checkpoint
保持不变；二者架构、候选空间和 checkpoint 均不兼容。

本实现落实引用讨论中的四个核心方向：

1. **Phase-response-aware structural representation**：共享 3-D encoder 后显式分成
   structural `S` 与 response `R`。`S` 由多尺度三维邻域自相似关系生成；`R` 通过强度
   重建和 phase 分类保留强化响应。结构分支使用 gradient reversal 去除可直接判别期相的
   信息，并以跨分支协方差约束避免 `S/R` 重新塌缩为同一表示。
2. **Explicit recurrent 3-D correspondence search**：不再逐切片预测。每一级在当前 flow
   周围枚举 `(2r+1)^3` 候选，得到 `p(Δ|x)`，再移动搜索中心重复搜索。卷积 GRU 只能学习
   soft-argmax 更新的增益和有界亚体素偏置，不能绕过 correlation 自由回归 flow。
3. **Correspondence uncertainty**：从候选分布直接输出归一化熵、峰值概率和位移方差。
   不确定性控制循环更新幅度，进入训练校准，并随 flow/uncertainty 一起导出；不再依靠
   “flow 越平滑越可信”的三平面启发式融合。
4. **Four-phase star protocol 与 probabilistic relation**：P 是独立的平扫期，绝不再作为
   C1 的别名。真实 pair 训练仅使用三个论文主方向 `P<-C1`、`P<-C2`、`P<-C3`；关系训练
   轮换 `(P,C1,C2)` 与 `(P,C2,C3)`，每个三角形同时约束组合 flow 的均值、对角方差和
   联合熵。这是 correspondence distribution 的矩闭合，不只是普通 flow cycle。推理仍
   只需一对 CT。

## 明确没有实现的三项

- Foundation feature distillation；
- Transformer；
- Diffusion。

代码中也没有把普通 image-level InfoNCE 接在 bottleneck 后。对比监督来自已知合成
形变 `J(y)=T_intensity(I(y+g(y)))`：`g` 是 scaling-and-squaring 得到的光滑采样形变，
因此每个 voxel 的 positive correspondence 精确已知；候选 NLL 使用连续位移的三线性
soft label，相似但位置错误的候选直接作为 hard negatives。

## 唯一 flow 约定

```text
warped_moving(z,y,x) = moving(z+dz, y+dy, x+dx)
flow: [B,3,D,H,W], component order=(dz,dy,dx)
unit: current/common tensor-grid voxels
mapping: fixed/output grid -> moving/input sampling position
```

`compose_flows(A_to_B, B_to_C)` 返回 `A_to_C`。NIfTI 推理先用 affine 将 moving
重采样到 fixed 的物理网格，导出的 flow 因而处于 fixed common grid，而不是 native
moving voxel 单位。

## 目录

| 部分 | 代码 |
|---|---|
| 3-D `S/R` encoder 与 DNS | `model/encoder.py` |
| 候选 response compatibility | `model/response.py` |
| 显式 3-D correlation distribution | `model/correlation.py` |
| correspondence-constrained recurrence | `model/update.py` |
| coarse-to-fine PRA-CM | `model/pracm.py` |
| 合成精确 correspondence | `training/augmentation.py` |
| 全部配准/解耦/关系损失 | `losses/objective.py` |
| pair/triplet 训练图 | `training/module.py` |
| PLC-R 体 patch 数据 | `data/dataset.py` |
| 训练与断点恢复 | `scripts/train.py` |
| pairwise NIfTI 推理 | `scripts/infer.py` |

训练前向和损失不读取 segmentation label。冻结验证集的 liver mask 可仅用于 checkpoint
选择，选择分数同时考虑 DSC、折叠率与有效支持率；这一用途会写入 run manifest，且标签
不会进入梯度。PLC-R 中 replicated-component CT 必须继续使用旧版本已经验证并带
provenance 的 scalar cache；新 loader 对 4-D 输入 fail closed，不会自行选通道。

数据协议强制校验最终 250 例及固定划分 `172/29/49`，并用冻结 manifest 的源 hash 校验
四期 inventory。任何病例缺少 P/C1/C2/C3、P 被错误映射、清单变化或恢复训练时数据协议
变化都会直接报错。旧 147 例 P/C2 队列和 demo 的 392 个方向不属于本版本训练/主测试。

### Learn2Reg MR–CT 适配的边界

`pracm_l2r_mrct_v1.yaml` 使用相同的 3-D structural/response encoder、显式
correlation distribution、recurrent search 和 uncertainty 输出，但严格区分监督来源：

- auxiliary MR/CT 是**未配对域数据**。loader 分别在各自 NIfTI 原生网格取 patch，
  不把任意 MR×CT cross-product 当成真实解剖配准对；
- 每个域独立生成已知 diffeomorphic flow，在 MR 内与 CT 内提供精确 voxel positive、
  hard negative、candidate NLL、flow 和 uncertainty calibration；共享 `S` 分支的域对抗
  去除 MR/CT 可判别信息，`R` 分支仍重建并判别 acquisition domain；
- 未配对数据无法辨识 candidate-specific MR/CT appearance compatibility。因此 L2R 使用
  `response_gate_mode: neutral`，候选排序由 structural correlation 决定；PLC-R 继续使用
  `learned` gate；
- public-8 paired image/label 只用于最终推理评价，不进入训练或 checkpoint 选择。

PLC 的 `PHASES=(P,C1,C2,C3)` 保持冻结；MR/CT 使用独立、checkpointed
`acquisition_identities=(MR,CT)`。两者不会 alias。同一算法代码支持两种问题，但分别训练
checkpoint。

本轮正式 L2R 协议标记为 `controlled_200`：固定 200 epochs、每轮 85 个 PRA-CM
logical samples、10 epochs warmup，并让学习率在第 200 轮走完完整衰减。120/40 的
minimum/patience 只记录“可能已收敛”的诊断，不提前截断统一预算；末 20 轮若仍明显改善，
结果标记为可继续训练而不声称 paper-full。每 50 epochs 保留一个完整恢复点，并持续原子
更新 `last.pt` 与 `best.pt`。恢复点包含 optimizer、scheduler、AMP scaler、
logical/optimizer/global step、随机数状态以及配置/源码/数据身份哈希。

## 配置

复制 `configs/pracm_plc_r_v1.yaml` 后，只修改：

```yaml
data:
  manifest: data/PLC_R_v1/PLC_R_manifest_v1.json
  phase_inventory: data/PLC_PreRegistration_Analysis_v1/case_inventory.csv
  data_root: /actual/PLC-CECT
  scalar_cache: /actual/PLC_R_scalar_cache
```

`data_root` 和 `scalar_cache` 下都保持冻结 manifest 的相对路径，例如
`ct_files/P0001_ct_C1.nii.gz`。默认采用 PLC observed `uint8/255`；真实 HU 数据必须把
`intensity_mode` 改为 `hu_window` 并明确窗口。

## 训练

在项目根目录执行：

```powershell
$env:PYTHONPATH = 'E:\00-Code\00-Chat-code\KidneyCT_Registration_Response\code'
$python = 'C:\Users\JIA\anaconda3\envs\irsiv3-reg\python.exe'
& $python -m registration_v3_pracm.scripts.train `
  --config code\registration_v3_pracm\configs\pracm_plc_r_v1.yaml `
  --output-dir '<项目外实验目录>\pracm_v3' `
  --device cuda
```

恢复训练：

```powershell
& $python -m registration_v3_pracm.scripts.train `
  --config '<同一配置>' --output-dir '<同一输出目录>' --device cuda `
  --resume '<同一输出目录>\checkpoints\last.pt'
```

默认优化上限为 200 epochs，含 10 epochs warmup、余弦衰减、至少 60 epochs 以及 25 次
验证无改进才 early stop，不再使用 demo 的短时限训练。验证集 29 例的三个 `P<-Ci`
方向各评估一次，共 87 个固定、liver-centered 的 patch 样本（不是未经声明的全体积
验证）。checkpoint 保存模型、optimizer、scheduler、AMP
scaler、全部 RNG、配置/源码 hash 和四期数据身份；任一项改变时拒绝精确恢复。

## 成对推理

```powershell
& $python -m registration_v3_pracm.scripts.infer `
  --config code\registration_v3_pracm\configs\pracm_plc_r_v1.yaml `
  --checkpoint '<run>\checkpoints\best.pt' `
  --fixed '<P.nii.gz>' --moving '<C2.nii.gz>' `
  --fixed-phase P --moving-phase C2 `
  --output-flow '<case>.pracm.flow.npz' `
  --output-warped '<case>.warped.nii.gz' `
  --output-qa '<case>.qa.json' `
  --device cuda
```

NPZ 同时保存 flow、方差、熵、最大候选概率、response gate、candidate coverage、endpoint
validity、fixed affine 与完整约定。QA 报告 evidence domain 中的 Jacobian 和 uncertainty。

## 资源与边界

- 默认训练 patch 为 `96×160×160`，batch size 固定为 1；P/C1/C2/C3 由 patient-balanced
  volume sampler 产生。为控制 3-D graph 峰值显存，pair step 负责精确 synthetic
  correspondence，triplet step 负责三条真实边的概率关系闭合；两种训练制度按配置概率
  交替，而不是在同一步保留四套体网络计算图。
- 推理超过 tile 时采用带重叠 Hann evidence blending。tile overlap 应不小于预期最大
  位移；特别大的器官位移应增大 tile/overlap 或使用整幅推理。
- candidate sampling 按块进行，但训练保留概率 volume 以计算精确 NLL；不要把训练
  `candidate_chunk_size` 误解为候选数。
- 这是可训练、可推理的方法代码，不包含新实验结果或可用于论文结论的 checkpoint。
