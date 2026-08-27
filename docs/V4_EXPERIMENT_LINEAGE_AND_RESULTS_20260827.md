# V4 实验谱系、验证过程与四指标完整汇总

生成日期：2026-08-27  
范围：本地 43 个 **code/registration_v4*** 目录，以及理解这些目录所必需的 registration_v3、B02、B04、B05 和 Identity 对照。  
组织方式：按“当前表现最好、最接近目标 → 有效增量 → 冻结基线 → 机制修正 → 训练修正 → 失败架构与未完成候选”排列，不按创建时间排列。

## 1. 结论先行

1. 当前已完成的最佳完整 public-8 是 **V4 DINO-anchor capture-router**：Mean Dice 0.785949746、ASSD 4.577276 mm、HD95 19.858701 mm、Fold fraction 0.026441595。它高于 B04 DINO-Reg 的 0.782111717，但仍低于 0.80，更未达到既定 0.85 目标。
2. 0.78595 不是“V4 描述子单独取得”的结果。路由器在 8 例中对 0004、0014 选用 dense corefix，其余 6 例保留 B04。它证明了“强锚点 + 只在超出捕获范围时替换”的选择性组合有效，但增量只有 0.003838。
3. **B04 → 重新编码 → 官方 Adam 连续残差**完成了完整 public-8：Mean Dice 0.783104826、ASSD 4.846534 mm、HD95 23.328032 mm、Fold 0.040684255。四项都比 B04 略好，但 Dice 仅增加 0.000993，说明它是可吸收的细化组件，不是独立终点。
4. 冻结训练型 **V4-Core** 的完整 public-8 是 0.663361521 / 8.234422 / 27.164855 / 0.000059382。它必须保留为不可覆盖的可复现基线，但不是最近探索的最高结果。
5. Solver 修复能把 V4-Core 同三例 Dice 从 0.634999 提到 0.694435，却仍低于同三例 B04 的 0.727893；把 V4-Core 描述子接到官方 B02 solver 也只有 0.644639。由此可见 solver 有问题，但 correspondence/descriptor 才是更主要的性能上限。
6. batch-one、固定学习率、mask、appearance view、取消 gradient clipping、Jacobian 回溯均修复了真实偏差，但最佳训练分支 public-3 仍只有 0.613269。增加 epoch 或继续围绕训练边角修补，没有证据能自然补到 0.8–0.85。
7. 当前最清晰的信息对象链应固定为：

   **raw descriptor d(x) → candidate cost Cx(k) → full-amplitude MAP u*(x) 与独立 confidence w(x) → spatial residual r(x) → pullback-composed final flow**

   过去的关键错误，是在不该消除信息的对象上做了归一化、池化、后验均值和置信度缩幅。

## 2. 指标口径与可比性

- **Mean Dice ↑**：canonical-v2 的四器官等权宏平均。0002、0004 缺失 left-kidney 标注，不再当作 Dice=0；完整 public-8 共有 30 个可评价器官病例。
- **ASSD (mm) ↓**：平均对称表面距离。
- **HD95 (mm) ↓**：95% Hausdorff 距离。
- **Fold fraction ↓**：最终位移场负 Jacobian 的比例。
- **public-8** 与 **public-3** 不能直接混排。public-3 只包含固定的 0004、0010、0002，用于快速辨别方向。
- 表中“未生成/未同步”表示本地没有对应正式落盘值；不会从日志印象或其他面板推算。
- 评价修正证据：**results/L2R_MRCT_protocol300_20260825/aggregate/METRIC_CORRECTION.json**。

## 3. 完整 public-8 结果

| 排名 | 方法/版本 | Mean Dice ↑ | ASSD mm ↓ | HD95 mm ↓ | Fold fraction ↓ | 性质与结论 |
|---:|---|---:|---:|---:|---:|---|
| 1 | V4 DINO-anchor capture-router | 0.785949746 | 4.577276 | 19.858701 | 0.026441595 | 当前最佳；B04 作为默认锚点，仅 0004、0014 换成 dense corefix |
| 2 | B04 → re-encode → Adam residual | 0.783104826 | 4.846534 | 23.328032 | 0.040684255 | 四项相对 B04 同向小幅改善；连续细化组件 |
| 3 | B04 DINO-Reg | 0.782111717 | 4.962758 | 23.348518 | 0.041089906 | 最强稳定锚点 |
| 4 | B02 ConvexAdam + MIND-SSC | 0.735027397 | 6.296807 | 20.377894 | 0.015578249 | 无训练强基线；说明任务上限并非 0.4 |
| 5 | 冻结 V4-Core | 0.663361521 | 8.234422 | 27.164855 | 0.000059382 | 当前正式训练型 V4 基线；不可覆盖 |
| 6 | PRA-CM v3 / B10 | 0.412383524 | 15.299623 | 39.364444 | 0.000647142 | v3 正式结果；对应信息未被有效利用 |
| 7 | V4-A relational | 0.408677545 | 15.281151 | 39.511429 | 0.007211749 | 旧 V4 分支；接近 v3 |
| 8 | V4-Full epoch175 | 0.403136190 | 14.981566 | 38.648462 | 0.037425380 | 旧 interim checkpoint；未因训练变充分而跃升 |
| 9 | B05 MASR/DNS + IO | 0.380775506 | 15.041330 | 38.751084 | 0.003902902 | 接近 Identity；复刻链存在根本对应问题 |
| 10 | V4-B hierarchical | 0.377377902 | 15.393336 | 39.154447 | 0.000000000 | 位移过保守，几乎没有有效配准 |
| 11 | Identity | 0.370861151 | 15.631536 | 39.433018 | 0.000000000 | 配准前基准 |

补充：DINO spacing-6 residual 曾记录完整 public-8 Mean Dice 0.7807871，但本地没有同步该次 ASSD、HD95、Fold 的正式 summary，因此不把它混入“四指标完整排名”。其结论仍成立：它低于 B04，没有通过保留门。

## 4. 固定 public-3 与诊断结果

下表使用固定 0004、0010、0002。相同面板内可以比较；它们不能替代 public-8。

| 实验 | Mean Dice ↑ | ASSD mm ↓ | HD95 mm ↓ | Fold fraction ↓ | 主要验证问题 | 结论 |
|---|---:|---:|---:|---:|---|---|
| B04 → re-encode → Adam | 0.736267636 | 6.650741 | 27.168882 | 0.027772748 | B04 后重新编码再做连续细化是否有净增量 | 通过；随后 full-8 也有小增量 |
| B04 reference | 0.727892919 | 6.973402 | 27.297255 | 0.028182814 | 同三例锚点 | 强基线 |
| B04-init + V4-Core residual | 0.724192986 | 7.692016 | 32.412361 | 0.033740234 | 保留 B04，只让 V4-Core 估残差是否更好 | 总体退化；局部病例可能有用 |
| V4 solver corefix | 0.694434990 | 7.681638 | 34.314247 | 0.000169599 | 修复 dense coarse、IC、composition 后 solver 能补多少 | 明显高于同三例 V4-Core，但仍低于 B04 |
| V4-Core + official B02 solver | 0.644639321 | 10.960408 | 26.844195 | 0.000916771 | 强 solver 能否挽救 V4 描述子 | 不能；descriptor 是主要瓶颈之一 |
| Frozen V4-Core reference | 0.634999441 | 10.535448 | 30.906348 | 0.000068156 | 同三例冻结基线 | 用于 solver 增量对照 |
| V4 cost-decoupled solver | 0.630034784 | 10.356311 | 30.798317 | 0.000042951 | descriptor cost 与 coverage 分离是否改善 | 拓扑更好但 Dice 下降；不保留为主线 |
| DNS maskfix + jacbacktrack，v4_final e25 | 0.613268723 | 12.197225 | 41.814443 | 0.000132073 | mask/background 与训练语义修复是否形成跃升 | 该训练族最好三例值，但仍远低于 B04 |
| DNS noclip + jacbacktrack，paper e5 | 0.596082006 | 12.641171 | 45.453691 | 0.000150158 | 取消全局 gradient clipping 是否立即改善 | 无决定性增量 |
| DNS noclip + jacbacktrack，legacy e5 | 0.595414428 | 12.743095 | 43.772169 | 0.000158409 | legacy exposure 控制 | 无决定性增量 |
| DNS noclip + jacbacktrack，v4_final e5 | 0.590087831 | 12.952751 | 45.461532 | 0.000180506 | 同上，V4 训练路由 | 无决定性增量 |
| DNS viewfix + jacbacktrack，legacy e5 | 0.590909765 | 12.926488 | 45.224478 | 0.000145919 | 第二 appearance view 的背景是否被错误抹除 | 修复真实语义，但仍低 |
| DNS viewfix + jacbacktrack，v4_final e5 | 0.579859541 | 13.252304 | 46.197200 | 0.000210458 | 同上 | 未形成跃升 |
| DNS viewfix + jacbacktrack，paper e5 | 0.574451441 | 13.332351 | 44.423389 | 0.000173894 | 同上 | 未形成跃升 |
| MIND + audited residual solver | 0.524284654 | 15.720168 | 49.205348 | 0.000135295 | 把审计 solver 接到 MIND 是否自动变强 | 该接法失败；强基线不是“任意 descriptor + 任意 solver” |
| MIND + audited solver，无 input-L2 | 0.523608667 | 15.713031 | 49.192343 | 0.000139703 | 输入 L2 是否是主要错误 | 几乎不变，排除为主因 |
| correspondence-corefix e120 | 0.360686363 | 未生成/未同步 | 未生成/未同步 | 未生成/未同步 | 120 epoch、10200 步后原生 3D correspondence 是否完成跨模态迁移 | 失败；停止用增加 epoch 解释差距 |
| 旧 V4-Full e175 + corrected IO probe | 0.348939040 | 18.141179 | 44.193488 | 0.050486190 | 只修正旧 checkpoint 的多尺度 descriptor IO 是否能复活 | 不能；旧表示本身无效 |

## 5. 关键实验过程：每一步为什么做、验证了什么

### 5.1 先修正评价器，而不是继续解释错误数值

**问题。** 0002、0004 没有 left-kidney 标注，旧评价器把它们当作 Dice=0。  
**修正。** canonical-v2 只在实际存在的 30 个器官病例上计算四器官等权宏平均。  
**作用。** B04 从旧口径 0.728945 修正为 0.782112，B02 从 0.687177 修正为 0.735027，PRA-CM v3 从 0.390973 修正为 0.412384。  
**结论。** 绝对值被系统性低估，但方法排序和 V4 与 B04 的巨大差距没有消失。

### 5.2 修正 flow IO、单位、分量顺序与复合

**问题。** 残差与已有 flow 不是简单逐点相加。正确对象是固定网格到 moving 采样位置、分量 dzyx、单位 native voxel。  
**正确复合。**

phi_final(x) = r(x) + phi_anchor(x + r(x))

**验证。** 非立方体、常量平移、仿射场、零残差 bit-exact 回退、final-flow Jacobian 均加入合成门。  
**结果。** 真实错误被修复，B04-init 残差能够在部分病例改善，但三例总体 0.724193 仍低于 B04 0.727893。  
**结论。** IO 错误必须修，但它不是 0.4→0.85 的唯一解释。

### 5.3 把 solver 与 descriptor 交叉拆开

**实验 A。** 冻结 V4-Core descriptor，只替换为修复后的 dense spacing12→spacing6 solver：0.634999→0.694435。  
**实验 B。** 冻结 V4-Core descriptor，接官方 B02 solver：0.644639。  
**实验 C。** MIND descriptor 接当前 audited residual solver：约 0.524。  
**判别目的。** 判断主要瓶颈在 descriptor、solver，还是二者接口。  
**结论。** 当前 solver 的确丢失了性能，但强 solver 不能让弱 descriptor 达到 B04；同样，强 descriptor 也不能随意接到一个语义不匹配的 solver。真正问题是 descriptor—cost—candidate—residual 的对象契约必须一致。

### 5.4 从 identity 重求改为保留 B04 锚点

**问题。** dense corefix 从 identity 重求整场 flow，会修复困难病例，却破坏 B04 已经配好的简单病例。  
**实验。** B04 先预变形 CT，V4 只估残差并用 pullback composition 合成。  
**结果。** public-3 总体略退化，但 0004/0014 等困难病例存在互补收益。  
**吸收方式。** 不再要求一个残差场替换全部 B04，而用无标签 capture gate 只在 coarse p95 超过 24 voxel、拓扑和双向链安全时选择 dense flow。  
**最终结果。** public-8 选择 0004、0014 dense，其余 6 例 B04，得到当前最佳 0.785950。

### 5.5 纠正 DINO PCA24 的处理对象

**问题。** 24 是 joint pairwise PCA 坐标轴，不是候选轴、尺度轴或空间轴。此前接口对 PCA24 做通道 L2 归一化，改变约 46.19% 的候选 argmin；spacing-6 池化又改变约 51.87% 的 argmin。  
**官方语义。** raw PCA24 沿 24 通道求和 SSD；K=729 才是位移候选轴，后验和 MAD 只能沿 K 处理。  
**验证。** raw-PCA 分支与官方候选 cost 做逐候选数值等价，最大绝对误差 0、argmin mismatch 0。  
**结果。** 候选对象被修正后，硬后验和软均值分支仍退回 B04，因为真实 CT–MR 后验高度多峰、margin 很小。  
**结论。** 修正 raw PCA 是必要条件，但“cost 正确”不等于“对应足够判别”。

### 5.6 发现后验均值和 evidence 缩幅在错误位置消除了信息

**观察。**

- posterior entropy 约 0.883–0.891；
- top-1 probability 仅约 1.74%–2.25%；
- top-3/top-5 质量约 5.75%/8.74%；
- posterior mean 经常不落在任何 top-5 mode；
- 双向一致率约 6.25%–28.13%，相对 margin 约 1.2%–2.7%；
- 把约 1e-4 evidence 乘到位移上，会把 1–数 voxel 的候选压到约 0.01 voxel。

**结论。** confidence 应控制数据项在空间拟合中的权重，不能改变候选位移幅值。于是产生 raw-PCA MAP residual 分支：先保留完整 argmin 位移 u*，再让 w 只进入空间目标的数据项。

### 5.7 验证连续细化的实际边际

**目的。** 避开候选和 posterior，直接问：B04 后重新编码是否还含可被官方 Adam 利用的连续残差。  
**实现。** B04 预变形 CT；重新运行官方 DINO case_inference；raw PCA24 直接进入官方 convex_adam_3d_param；disp_init 为非空零场，从而跳过 spacing-6 离散阶段。  
**结果。** public-3 Dice 0.727893→0.736268；public-8 0.782112→0.783105，且 ASSD、HD95、Fold 同向改善。  
**结论。** 连续细化是真实但很小的增量，应作为 MAP 或强锚点之后的最后一层，而非新的主架构。

### 5.8 系统排除训练协议为主要根因

围绕 DNS/V4-Core 依次隔离了：

1. batch size 1、Adam 固定 1e-4、无 warmup/decay；
2. appearance 每个 anchor 必做，geometry 只作额外更新；
3. CT 连通背景与内部 0 HU 的 mask 语义；
4. nonlinear/inversion 后第二 appearance view 不再重复抹去背景；
5. 完全取消 global gradient clipping；
6. raw Adam 导致折叠时，用最大安全 alpha 做 Jacobian 回溯；
7. fp16 非有限时成对回退 bf16，禁止 fixed/moving 混合精度。

这些修复提高了协议忠实度和可复现性，但 public-3 始终约 0.57–0.61，未接近 B04。它们应保留为代码正确性修复，不再被当作达到 0.85 的主方向。

### 5.9 器官级先验是最新未完成分支

**触发原因。** 同一全局 flow 往往修好 kidney 却损伤 liver，spleen 又被所有全局 solver 丢失，错误表现是器官局部而非整例一致。  
**设计。** registration_v4_atlas_organ 只使用与 public-8 完全分离的 CHAOS MR / BCV CT 辅助标签建立四器官概率 atlas，再以 label-free 区域权重限制局部残差。  
**当前证据。** 本地已有对象契约、atlas 构建、先验验证和 refinement gate 代码；没有同步正式 public-3/public-8 结果。  
**状态。** 属于未完成探索，不进入性能排名，也不能声称有效。

## 6. 43 个本地 V4 目录逐项说明

### A. 当前最佳、B04 锚定与 DINO 信息对象

| 目录 | 来源与建立原因 | 实际测试/要验证的命题 | 结果与当前状态 |
|---|---|---|---|
| **code/registration_v4_solver_capture_router_dino_anchor_public8** | 从 dino-anchor router 生成的完整 public-8 包装；需要在八例上冻结选择规则并复制被选 flow | 验证 public-3 的无标签捕获规则能否泛化到全部八例 | 8/8 完成；0.785949746 / 4.577276 / 19.858701 / 0.026441595；当前最佳，保留 |
| **code/registration_v4_solver_capture_router_dino_anchor** | 从 capture-router 改成以 B04 为默认锚点、dense corefix 为备选；因为 Core 默认锚点弱于 B04 | coarse body p95>24 voxel 且 topology/FB 安全时才用 dense，是否能保留简单病例并修困难病例 | 支撑当前最佳；0004、0014 选 dense，其余选 B04；算法目录本身与 public8 包装不是两个方法 |
| **code/registration_v4_solver_capture_router** | 以冻结 V4-Core 和 dense corefix 为两个候选的路由前身 | 不看标签，仅按捕获范围、拓扑和双向链选择，能否避免 dense 对容易病例的破坏 | 本地无独立正式四指标 summary；低于随后 B04-anchor 版本，作为历史诊断保留 |
| **code/registration_v4_solver_crossobjective_router** | B04 默认锚点与 dense corefix 的另一种路由；以官方 B02/MIND objective 判断 | 跨 descriptor objective 是否比 capture-range 规则更可靠 | 本地只有 manifest、router 与 public3 runner，没有同步完成 summary；未形成更优正式结果 |
| **code/registration_v4_corefix_b04anchor** | 组合冻结 B04、冻结 V4-Core descriptor 和 audited solver；修复“从 identity 丢掉强锚点” | 预变形后只估残差、B04-relative fold ceiling、正确 pullback 是否可稳定增益 | 6/6 合成门通过；直接相关 public3 残差实验 0.724193 / 7.692016 / 32.412361 / 0.033740234，低于 B04；仅作为困难病例候选组件 |
| **code/registration_v4_dino_reencoded_fullres_residual** | 从 B04 和官方 DINO extractor 出发，完全绕开 candidate/posterior；用于测量连续残差上限 | B04 后重新编码 raw PCA24，再运行官方 Adam 是否有可泛化净增量 | 7 项合成门通过；public3 0.736268；public8 0.783105 / 4.846534 / 23.328032 / 0.040684255；保留为连续细化组件 |
| **code/registration_v4_dino_rawpca_map_residual** | 从 raw-PCA posterior 分支继续；修复 posterior mean 与 w×u 缩幅 | 完整离散 MAP u* 与 confidence w 分离后，空间正则能否传播而不抹除位移 | 6/6 CPU 合成门通过；本地没有同步 GPU/public3 结果；候选有效性尚未成立 |
| **code/registration_v4_dino_posterior_rawpca** | 从 posterior core 分叉；因为通道 L2 归一化改变官方 raw PCA24 候选排序 | C=24 与 K=729 的对象契约、官方 channel-sum SSD 数值等价、候选轴 posterior | 6/6 合成门；cost 最大误差 0、argmin mismatch 0；真实分支回到 B04，无独立性能增量；保留为契约/证据代码 |
| **code/registration_v4_dino_posterior_core** | 从 dino_core + B04 建立的硬 posterior/evidence 残差 | mutual consistency、MAD improvement 和 top1/top2 separation 能否筛出可信位移 | 6/6 合成门；真实后验弱，残差接近零，结果退回 B04；排除为最终流 |
| **code/registration_v4_dino_posterior_soft_core** | posterior_core 的软均值对照 | 不用硬 argmin，posterior mean 是否能在多峰情况下更稳定 | 6/6 合成门；真实多峰相消，仍退回 B04；排除 |
| **code/registration_v4_dino_core** | 把官方 DINO-Reg case_inference 的 pairwise PCA24 捕获到 V4 solver 接口 | official DINO descriptor 与 V4 residual solver 的交叉兼容，以及 identity/B04-prewarp 三种初始化 | extractor 被后续分支继续复用；spacing-6 full8 Dice 0.7807871，其他三项未同步，低于 B04；保留 extractor，淘汰该 solver 配置 |
| **code/registration_v4_atlas_organ** | 由“全局 flow 对不同器官利弊相反”的病例分析产生；使用 public-8 之外的 CHAOS/BCV 标签做 atlas | label-free 器官区域 prior 能否把残差限制在 liver/spleen/kidney 各自区域 | 代码、对象契约和 refinement gate 已存在；无本地正式 public3/public8 指标；未完成探索 |

### B. 冻结 V4-Core 正式谱系

| 目录 | 来源与建立原因 | 实际测试/要验证的命题 | 结果与当前状态 |
|---|---|---|---|
| **code/registration_v4_final** | 替代失败的 V4-A/B/Full；回到 Figure-9 特征、direct+dilated DNS、24-channel DSIR 与 descriptor-agnostic ConvexAdam | 忠实训练型 V4 在 protocol-300 下能达到什么水平 | 300-epoch 正式谱系；public8 0.663361521 / 8.234422 / 27.164855 / 0.000059382；冻结，不得覆盖 |
| **code/registration_v4_final_main** | registration_v4_final 的稳定 benchmark train/infer v2 入口 | 防止服务器脚本依赖内部模块路径，保证同一方法入口可复现 | 纯入口 alias；共享同一 checkpoint 和 0.66336 结果，不是独立方法 |
| **code/registration_v4_final_release** | registration_v4_final 的 audited release inference/train 入口 | 固定正式部署路径和 solver 入口 | 纯 release alias；共享同一 0.66336 结果，不应在结果表重复计数 |

### C. Solver 对象与阶段拆解

| 目录 | 来源与建立原因 | 实际测试/要验证的命题 | 结果与当前状态 |
|---|---|---|---|
| **code/registration_v4_solver_corefix** | 冻结 V4-Core descriptor，只重写 solver；修复 sparse/global translation、fixed denominator、coarse/residual composition 与 acceptance | solver 错误能解释多少性能差距 | 3 个组件门通过；public3 0.694435 / 7.681638 / 34.314247 / 0.000169599；有明显增量但低于 B04，作为 dense 候选保留 |
| **code/registration_v4_solver_corefix_release** | corefix 的稳定推理入口 | 确保服务器调用固定 solver 实现 | release alias；无独立结果，复用 corefix 指标 |
| **code/registration_v4_solver_costdecoupled** | 从 corefix 分叉；发现 coverage penalty 与 descriptor mismatch 被错误加成同一 cost | 把 descriptor cost 与 coverage hard constraint 分开，是否能同时保持捕获与拓扑 | 3 个组件门；public3 0.630035 / 10.356311 / 30.798317 / 0.000042951；拓扑改善但丢掉 0004 大位移收益，不保留主线 |
| **code/registration_v4_solver_costdecoupled_release** | costdecoupled 的固定推理入口 | 复现负结果 | release alias；共享同一负结果 |
| **code/registration_v4_solver_stage_decoupled** | 从 corefix/costdecoupled 继续，把 spacing12 coarse 与 spacing6 residual 的目标职责分开 | coarse 负责捕获、residual 只优化 mutual descriptor cost，是否避免同一 penalty 在不同阶段含义漂移 | 代码、manifest、public3 runner 存在；本地没有同步完成 summary；未形成保留结果 |
| **code/registration_v4_solver_stage_decoupled_release** | stage-decoupled 稳定入口 | 固定诊断版本 | release alias；无独立指标 |
| **code/registration_v4_solver_local_consensus** | 从冻结 Core 与 dense corefix 两个完整 flow 出发，尝试局部而非整例选择 | relative cost、两候选 entropy/margin、FB cycle 与局部 evidence 能否组合成可靠空间权重 | 合成规则与 public3 runner 存在；本地无完成 summary；未形成更优正式结果 |
| **code/registration_v4_solver_local_consensus_release** | local-consensus 固定推理入口 | 复现实验代码 | release alias；无独立指标 |

### D. DNS、搜索范围与训练协议修正

| 目录 | 来源与建立原因 | 实际测试/要验证的命题 | 结果与当前状态 |
|---|---|---|---|
| **code/registration_v4_searchfix** | 从冻结 registration_v4_final 复制的最小 solver-only 版本；原 spacing6×radius4 只能捕获 ±24 voxel，而困难病例约 32–41 voxel | 增加 spacing12 ±48 全局平移，再接 spacing6 ±24 residual，合计约 ±72，能否解决捕获范围不足 | 6 个非立方/35–42 voxel 合成门通过；无独立正式 public8；历史最小修复副本 |
| **code/registration_v4_searchfix_release** | searchfix 稳定 inference alias | 固定最小搜索修复的部署入口 | release alias；无独立指标 |
| **code/registration_v4_dnsfix** | 从 final/searchfix 扩展；除层级搜索外，修正 shared squeezing、24 DNS relation、full-FOV foreground 和 invalid penalty=8 | 同时修复 descriptor 忠实度与捕获范围能否补足差距 | 组件契约通过，但没有高于冻结基线的正式 full8；成为后续训练修正父分支 |
| **code/registration_v4_dnsfix_release** | dnsfix 的 stable train/infer alias | 服务器正式调用 | release alias；不是独立方法 |
| **code/registration_v4_dnsfix_residual_release** | dnsfix 的 identity-initialized residual-only 控制；明确绕过全局平移前端 | 判断层级 global translation 本身是否是性能来源 | 有 identity、非立方 translation、uint8 mask regression 代码；本地无独立正式四指标结果；属于控制，不是普通 release 快照 |
| **code/registration_v4_dnsfix_batch1** | 从 dnsfix 分叉；修正为单 GPU、batch1、Adam 固定 1e-4，并恢复每个 anchor 的 appearance 暴露 | 训练 batch/LR/exposure 偏差是否是主要瓶颈 | parent manifest 无独立指标；后续 alias 的 public3 仍低，说明不是主要根因 |
| **code/registration_v4_dnsfix_batch1_release** | batch1 稳定 train/infer 入口 | 固定训练与推理命令 | release alias；不重复计结果 |
| **code/registration_v4_dnsfix_batch1_maskfix** | 从 batch1 分叉；修复 CT 外部 exact-zero 连通背景，同时保留内部 0 HU；修正 checkpoint selection | 错误背景支持是否让 descriptor 学到 FOV 轮廓而非解剖 | 对应 e25 + jacbacktrack public3 v4_final 为 0.613269 / 12.197225 / 41.814443 / 0.000132073；有改善但远低于 B04 |
| **code/registration_v4_dnsfix_batch1_maskfix_release** | maskfix 稳定 train/infer 入口 | 固定部署 | release alias；不独立计数 |
| **code/registration_v4_dnsfix_batch1_viewfix** | 从 maskfix 分叉；第二 nonlinear/inversion appearance view 不再重复 foreground mask | 两个视图是否因共同 FOV silhouette 形成伪对应 | e5 三变体约 0.574–0.591；修复真实语义但没有跃升 |
| **code/registration_v4_dnsfix_batch1_viewfix_release** | viewfix 稳定 train/infer 入口 | 固定部署 | release alias；不独立计数 |
| **code/registration_v4_dnsfix_batch1_noclip** | 从 viewfix 分叉；唯一训练变化是彻底取消 global gradient clipping | clipping 是否把需要的大梯度/表达变化压掉 | e5 三变体约 0.590–0.596；没有决定性改善 |
| **code/registration_v4_dnsfix_batch1_noclip_release** | noclip 稳定 train/infer 入口 | 固定部署 | release alias；不独立计数 |
| **code/registration_v4_dnsfix_jacbacktrack** | 与 batch1 训练分支隔离的 solver sibling；raw Adam 可降低 descriptor objective 却制造折叠 | 在 discrete→raw refined 线段上二分最大安全 alpha，能否同时保留优化与拓扑 | 合成折叠案例 alpha=0.65747 后 fold 归零；这是 QA 修复，不是独立描述子结果 |
| **code/registration_v4_dnsfix_jacbacktrack_maskfix_alias** | 将 maskfix checkpoint 严格接入 jacbacktrack，不修改训练或 solver | e25 maskfix 在统一安全 solver 下的真实表现 | 三 checkpoint 契约通过；v4_final public3 0.613269 / 12.197225 / 41.814443 / 0.000132073；非独立方法，是评估入口 |
| **code/registration_v4_dnsfix_jacbacktrack_viewfix_alias** | 将 viewfix checkpoint 接入同一 jacbacktrack | 隔离 appearance-view 修正的影响 | e5 完成三变体；最好 legacy 0.590910 / 12.926488 / 45.224478 / 0.000145919；未保留主线 |
| **code/registration_v4_dnsfix_jacbacktrack_noclip_alias** | 将 noclip checkpoint 接入同一 jacbacktrack | 隔离取消 clipping 的影响 | e5 完成三变体；最好 paper Dice 0.596082，完整四项见第4节；未形成跃升 |
| **code/registration_v4_dnsfix_jacbacktrack_noclip_precision_safe_alias** | 从 noclip alias 增加 inference precision 安全；fp16 任一侧非有限则 fixed/moving 成对改用 bf16 | 非有限或两侧混合精度是否污染 cost | manifest 指向 e50 paper_control，但本地无 verification/summary；未完成，不计性能 |

### E. 训练型 correspondence 重构与早期 V4

| 目录 | 来源与建立原因 | 实际测试/要验证的命题 | 结果与当前状态 |
|---|---|---|---|
| **code/registration_v4_correspondence_corefix** | registration_v4_pracm 的隔离继续；修复 zero coverage、MR/CT OOD response gate、candidate absolute-flow 语义及 global solver 丢失最后 updater | 原生 3D explicit correspondence 图修正后，充分训练能否完成跨模态迁移 | 11 个核心/继承门和 16 项选择面板通过；e120/10200步 public3 Dice 0.360686，其余指标未同步；跨模态训练失败，停止加 epoch |
| **code/registration_v4_pracm** | 从 registration_v3 v0.5.2 独立演进；S/R 解耦、显式 recurrent 3D candidates、posterior uncertainty、概率关系闭合 | 用原生 3D correspondence 取代 planar/tri-view 能否解决旧 v3 的表示问题 | 作为早期 V4 架构和 correspondence_corefix 的父目录；没有独立高于 B04 的正式结果；历史方法来源 |

## 7. registration_v3 到 V4 的来源关系

这两个目录不属于上述 43 个 V4 目录，但解释 V4 的起点：

- **code/registration_v3**：planar DNS、候选分布和三视图融合的正式 v3；对应 B10 public8 0.412383524 / 15.299623 / 39.364444 / 0.000647142。它证明旧链相对 Identity 只有很小改善。
- **code/registration_v3_pracm**：把 v3 思想转成原生 3D S/R、显式候选与不确定性的中间架构；registration_v4_pracm 从这里继续演进。
- **code/registration_v4_pracm**：早期 V4 主链；之后 correspondence_corefix 尝试修正其核心 correspondence 图。
- **code/registration_v4_final**：放弃旧 V4-A/B/Full checkpoint，重建 paper-aligned DNS/DSIR + ConvexAdam，形成 0.66336 冻结基线。
- 后续 solver、DINO-anchor、raw-PCA 和 reencoded 分支均围绕“保留冻结资产、隔离一个对象错误”展开。

## 8. 已经排除或显著降级的解释

| 被检验的解释 | 证据 | 当前判断 |
|---|---|---|
| 只是评价器算错 | 修正后所有值上升，但 V4 与 B04 的差距仍大 | 不是主因 |
| 只是 flow IO/复合错 | 合成门和 B04 residual 已修；总体仍未超过 B04 | 必须修，但不足 |
| 只是搜索范围太小 | ±72 层级搜索通过大位移合成门，真实性能未跃升 | 不是唯一主因 |
| 只是 solver 太弱 | corefix public3 0.694；official B02 solver + V4 descriptor 0.645 | solver 有损失，descriptor 更关键 |
| 只是输入通道 L2 | raw PCA 候选顺序大幅改变且已修；后验仍弱 | 必要修正，不足 |
| 只是训练 epoch 不够 | correspondence-corefix e120/10200步仍 0.361；旧 V4 e175 仍约0.403 | 不应继续盲目增加 epoch |
| 只是 batch/LR/mask/view/clip | 多个隔离分支最好 public3 0.613 | 真实偏差，但非决定因素 |
| 只要 posterior 平均更平滑 | posterior mean 常落在 top-5 mode 之外，残差被相消 | 错误信息对象，已排除 |
| confidence 应直接缩小位移 | 约1e-4 evidence 把有效位移压至约0.01 voxel | 错误；confidence 只能作数据项权重 |
| 任意强 descriptor + 强 solver 都能组合 | MIND+audited solver 与 V4+official B02 solver 均失败 | 接口对象与目标语义必须一致 |

## 9. 当前应保留、不应混淆的资产

1. **当前最佳实验结果**  
   代码：code/registration_v4_solver_capture_router_dino_anchor_public8  
   本地结果摘要：tmp/capture_router_summary.csv  
   数值：0.785949746 / 4.577276 / 19.858701 / 0.026441595。

2. **可吸收的连续细化组件**  
   代码：code/registration_v4_dino_reencoded_fullres_residual  
   public3 门：tmp/actor_parserfix_PUBLIC3.json  
   public8 门：tmp/actor_parserfix_PUBLIC8_GATE.json  
   数值：0.783104826 / 4.846534 / 23.328032 / 0.040684255。

3. **冻结训练型 V4-Core**  
   代码：code/registration_v4_final  
   冻结发布：releases/v4_core_0p663361_frozen_20260827  
   数值：0.663361521 / 8.234422 / 27.164855 / 0.000059382。

4. **最强稳定外部锚点 B04**  
   结果：results/L2R_MRCT_protocol300_20260825/extracted/server1/evaluation/protocol_300/B04_dino_reg  
   数值：0.782111717 / 4.962758 / 23.348518 / 0.041089906。

5. **release 与 alias 的含义**  
   名称含 release 的目录通常只是稳定命令入口或不可变部署包装；名称含 alias 的目录负责严格 checkpoint 契约和 solver 兼容。除 dnsfix_residual_release 这个显式控制外，它们都不应被当成额外论文方法或额外结果行。

## 10. 当前节点的技术判断

过去探索并非“一个更好版本都没有”。已经获得两个超过 B04 或局部超过 B04 的有效节点：

- 完整 public-8 当前最佳 0.78595 的选择性 DINO-anchor router；
- 完整 public-8 0.78310、四项同向改善的 re-encoded Adam 连续细化。

但这两个增量都很小。它们共同说明：B04 已经提供了大部分可用位移，V4 目前只能在少量困难病例或连续残差上补一点；V4 自身 descriptor/correspondence 尚未形成足以独立超过 B04 的强证据。

因此后续若继续讨论，必须从信息对象本身出发，而不是再展开训练协议树：

1. 哪一层/哪种特征仍保存 CT–MR 的局部定位信息；
2. 哪一步在通道、候选、空间或尺度轴上过早消除了信息；
3. 如何让完整位移 mode 被保留，同时把 uncertainty 仅用于“是否相信”，而不是“把位移压小”；
4. 如何在 B04 锚点上累计已经验证的正增量，而不是让新分支重置强基线。

达到 0.85 的目标当前仍未完成。本报告是一次完整实验节点记录，不把未完成代码、合成门通过或 alias 包装误写成性能成功。

## 11. 主要证据路径

- 完整 B00–B12 四指标：results/L2R_MRCT_protocol300_20260825/aggregate/headline_benchmark_summary.csv
- 评价修正：results/L2R_MRCT_protocol300_20260825/aggregate/METRIC_CORRECTION.json
- 旧 V4 A/B/Full：results/PRA_CM_v4_L2R_corrected_20260825/HEADLINE_V4_CORRECTED.csv
- 冻结 V4-Core：releases/v4_core_0p663361_frozen_20260827
- 当前最佳 router：tmp/capture_router_summary.csv、tmp/capture_router_pair_metrics.csv、tmp/capture_router_decisions.json
- re-encoded Adam：tmp/actor_parserfix_PUBLIC3.json、tmp/actor_parserfix_PUBLIC8_GATE.json
- B04-init V4 residual：benchmark_l2r_mrct/diagnostics/b04_init_v4_core_residual_public3_20260827
- solver corefix：benchmark_l2r_mrct/diagnostics/v4_solver_corefix_public3_20260827
- cost-decoupled：benchmark_l2r_mrct/diagnostics/v4_solver_costdecoupled_public3_20260827
- V4 descriptor + official B02 solver：benchmark_l2r_mrct/diagnostics/v4_core_official_b02_solver_public3_20260827
- MIND + audited solver：benchmark_l2r_mrct/diagnostics/mind_audited_solver_public3_20260827
- correspondence-corefix：benchmark_l2r_mrct/diagnostics/v4_correspondence_corefix_interim_best_public3_20260827
- raw PCA 对象契约：code/registration_v4_dino_posterior_rawpca/OBJECT_CONTRACT.json
- MAP 对象契约：code/registration_v4_dino_rawpca_map_residual/OBJECT_CONTRACT.json
- re-encoded 对象契约：code/registration_v4_dino_reencoded_fullres_residual/OBJECT_CONTRACT.json
- 器官 atlas 对象契约：code/registration_v4_atlas_organ/OBJECT_CONTRACT.json
