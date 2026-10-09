# CU-MOBO 改进方向（文献调研 + D1–D4 program）

> 2026-06-25 调研。目标:回答用户两问 —— (Q1) CU-MOBO 能否和 diffusion 等更先进生成模型联用?
> (Q2) CU-MOBO 算法本身怎么优化? 结论:两者都"能",且**文献里的改进恰好对应我们这次实测到的弱点**。

## 0. 我们实测到的弱点 → 文献解法（核心映射）

| 我们的实测弱点（来自本项目实验） | 文献解法 | 落点 |
|---|---|---|
| **高维 BO 输给 NSGA-II**（Path B 16 维, BO 全线被 NSGA-II 压) | **信赖域 BO**:TuRBO（局部 GP, 治过度探索 + 天然异方差, 与 σ-noise 协同）、**LOL-BO**（隐空间信赖域 + 联合训练, 比 SOTA 隐空间 BO 快 **20×**） | **D1** |
| **VAE 后验坍缩 / 隐空间饱和**（60 actives encode→decode 仅 1 唯一; 38% active 语料隐空间全活性 → 无 gap） | **联合隐空间训练**:LOL-BO / **CoBO**（让隐空间距离与目标相关）/ **PG-LBO**（伪标签 + GP 引导） | **D3** |
| **σ-as-objective 在稀有 active 反噬**（最小化 σ 把搜索从新颖 active 引开; 误差校准也没救回） | **自适应可信度**:**DyRAMO**（Nat Commun 2025, 同一 reward-hacking 命题）—— 动态调每目标可信度等级 + 可信域(AD)门控, 不死板最小化 σ | **D2** |
| **RF 树方差 σ 校准粗糙**（limitations 写的前提:σ 须真校准） | **深度集成**（LaMBO-2 用 partial deep ensembles）/ **conformal** / OOF 残差（已加 `sigma="calibrated"`) | **D1/D2** |
| **想接更强生成器（diffusion/flow）** | **采集梯度引导扩散采样**:**LaMBO-2**（离散 diffusion + NEHVI + 集成 UQ, 蛋白）、**NF-BO**（normalizing flow 可逆隐空间）、**Diff-BBO** | **D4** |

## 1. Q1 — 和 diffusion / flow 联用:能,已被验证

- **LaMBO-2 / NOS（Gruver, Stanton et al., NeurIPS 2023, 蛋白设计）= 最直接证据**:
  - 生成器 = 离散去噪 diffusion; 采集 = 多目标 **NEHVI**（=我们的 qNEHVI); UQ = **partial deep ensembles**（共享 encoder 的多个判别头）。
  - **采集如何引导扩散**:每个去噪步内用 **Langevin 动力学沿采集值(EHVI 估计)上升 + KL 惩罚**保持分子合理;
    用**采集梯度的 saliency** 选要改的位置（高价值位优先腐蚀/重采）。= "CU-MOBO 换 diffusion 生成器"。
  - **对我们的意义**:扩散**不需要可导航的欧氏隐空间** → 直接绕开我们这次踩的 VAE 坍缩/饱和。
- **NF-BO（ICLR 2025）**:用 normalizing flow 的**可逆**隐空间做 BO, 消除 VAE 重构/坍缩问题。
- **Diff-BBO**、**gradient guidance for diffusion（arxiv 2404.14743, optimization 视角）**:扩散作生成器, 目标梯度引导。
- **我们手上已有** molgen 的 Flow Matching（`latent_flow_matching.py` 的 `VelocityField`）→ 接 guided sampling 顺路。

## 2. Q2 — 算法本身优化:文献杠杆

- **信赖域 BO（D1)**:TuRBO（Eriksson 2019, `arxiv 1910.01739`）局部 GP + 信赖域伸缩/重启, 治高维退化;
  局部 GP 还天然处理异方差 → 与我们 σ-as-noise 协同。**LOL-BO**（Maus 2022, `arxiv 2201.11872`）把信赖域搬进隐空间 + 联合 VAE/GP 训练, 解"隐空间信赖域 ≠ 结构空间信赖域"的错配, 6 个真实基准 **20×**。
- **联合/相关隐空间（D3)**:**CoBO**（`arxiv 2310.20258`)让隐空间距离与目标相关;**PG-LBO**（AAAI 2024, `arxiv 2312.16983`）伪标签 + GP 引导提升高维隐空间 BO。
- **自适应可信度（D2) = DyRAMO**（`nature.com/articles/s41467-025-57582-3`）:
  - 可信域 AD = 与训练集**最大 Tanimoto 相似度 ≥ ρ**（ρ=可信度等级);每目标独立 AD, 交集=可行设计域。
  - reward 仅当**所有**目标 MTS≥ρᵢ 否则 0（硬约束, 防外推);DSS = ∏ Scalerᵢ(ρᵢ)^(1/n) × top10%reward（几何均值防忽略弱目标)。
  - **用 BO 在 ρ 组合空间里搜**最优可信度组合（EI, 10 随机 init + 30 BO 步 ×5 episode)。
  - 验证:设计 EGFR 抑制剂, **重发现 gefitinib**(从训练集剔除后仍找回), 不产生不合理分子。
  - **vs 我们固定 σ-目标**:单一固定 σ 不能按目标独立调可信度; DyRAMO 的 Scaler 每目标可调 σ + BO 自动调 → 正好治我们 Path B 的 σ-objective 反噬。

## 3. 关键文献（sources）

- LaMBO-2 / Protein Design with Guided Discrete Diffusion (NeurIPS 2023): https://arxiv.org/abs/2305.20009
- LaMBO (原版, DAE+BO): https://arxiv.org/abs/2203.12742
- LOL-BO (Local Latent Space BO): https://arxiv.org/abs/2201.11872
- CoBO (Correlated Latent Space): https://arxiv.org/abs/2310.20258
- PG-LBO (AAAI 2024): https://arxiv.org/html/2312.16983v1
- NF-BO (Latent BO via Normalizing Flows, ICLR 2025): https://proceedings.iclr.cc/paper_files/paper/2025/file/df4f371f1f89ec8ba5014b3310578048-Paper-Conference.pdf
- DyRAMO (Nat Commun 2025, reward hacking via reliability): https://www.nature.com/articles/s41467-025-57582-3
- TuRBO (Scalable Global Opt via Local BO, NeurIPS 2019): https://arxiv.org/abs/1910.01739
- Diff-BBO (diffusion inverse modeling for BBO): https://arxiv.org/html/2407.00610v1
- Gradient Guidance for Diffusion (optimization perspective): https://arxiv.org/abs/2404.14743

## 4. D1–D4 增量计划

见 `HANDOFF_CU_MOBO.md` §8。节奏:每步**最小可跑原型 + 合成 oracle-gap(隐藏真值)验证**再叠。
代码:`cu_mobo.py`(trust_region/reliability_mode 已加)、`benchmark/run_trust_region.py`、
`benchmark/run_reliability.py`、`molecule/train_latent_aware.py`(D3)、`molecule/flow_guided_d4.py`(D4)、
`molecule/kcat_surrogate.py`(sigma="calibrated")。

## 5. D1–D4 实测结果（诚实, 2026-06-25）

**这是探索性"算法改进"研究, 不是都成功 —— 如实记录正面/负面/有张力的结果。**

### D1 信赖域 + 校准 σ —— 结论: regime-敏感, 校准 σ 是真交付
- 信赖域(TuRBO-faithful 局部 GP, σ-discounted 选心)在 `UncertainOracleProblem`:n_var=8 **帮**(0.536 vs 0.38),
  n_var=16 **伤**(0.193 vs 0.393)。→ **不是可靠默认增益**, 默认关(`trust_region=False`), 仅大预算+调参用。
- **校准 σ**(`KcatSurrogate.fit(sigma="calibrated")`, OOF 残差)= 提供了"σ 追踪误差"的选项, 但**实测诚实**(见下 §6 校准审计):
  真实 kcat 上 OOF-σ 排序校准 Spearman 0.180 **反不如**树方差 0.213 → OOF 不是免费午餐, σ 校准本身才是真瓶颈(→ 更强 UQ / E2 自适应)。

### D2 自适应可信度（DyRAMO 式）—— 结论: 程序里最强的正面成果 ✅
`reliability_mode="adaptive"`: μ_eff = μ − λ·relu(σ−τ), τ=σ 的 q 分位; 只罚超阈极不可信(排陷阱), 留中等 σ 新颖真最优。
测试床 `problems/uncertain_optimum.py`(真最优 σ 中等、安全诱饵 σ 低、陷阱 σ 极高), `benchmark/run_reliability.py`。
**3-seed(n_var6/budget48, λ=3 q=0.75)**:

| method | top-K 真实活性 (mean±std) | 触达真最优 frac |
|---|---|---|
| naive | 0.736 ± **0.251** | 0.120 |
| sigma-obj | 0.730 ± 0.091 | 0.112 |
| cu-mobo(含 σ-obj) | 0.674 ± 0.093 | 0.048 |
| **d2-adaptive** | **0.779 ± 0.077** | **0.175** |
| nsga2 | 0.372 ± 0.208 | 0.067 |

→ d2-adaptive **活性最高 + 触达真最优最频 + 方差最小(最稳)**, 且**胜过静态 σ 方法**(sigma-obj/cu-mobo)。
naive 均值接近但方差 3×(碰运气, 时而落陷阱)。**= 自适应可信度修 σ-objective 反噬, 直接回应并修复 Path B 的负面发现。**

### D3 属性感知隐空间（LOL-BO/CoBO 式）—— 结论: 机制成立, 但暴露真实张力 ✅⚠️
`molecule/train_latent_aware.py`: VAE 隐 + 属性头 g(z)→活性, 联合训练。**z→活性 线性探针 AUC(5 split)**:

| latent | probe AUC |
|---|---|
| **property-aware (w_prop=5)** | **1.000 ± 0.000** |
| vanilla(纯重构) | 0.630 ± 0.011 |
| 原始 Morgan FP(1024, 上限参照) | 0.975 ± 0.005 |

→ **纯重构 16 维隐空间把活性信号几乎丢光**(0.63 « 0.975 结构上限)= CoBO/LOL-BO 的核心抱怨;属性监督把它找回(0.63→1.0)。
**诚实警告**:AUC=1.0 是**因为隐空间被直接监督在这个标签上**(留出但同标签)→ 证明隐**能**携带属性, 不等于下游 BO 自动变好。
**关键张力 = 重构保真 vs 属性组织(实测, w_prop 扫描)**:属性权重越大, probe 越好但解码器越塌——

| latent | z→活性 probe AUC | N(0,1) 解码有效率 | 唯一 |
|---|---|---|---|
| vanilla (w_prop=0) | 0.635 | 174/200 (87%) | 50 |
| **D3 w_prop=1（甜点）** | **1.000** | **129/200 (65%)** | 40 |
| D3 w_prop=5 | 1.000 | 33/200 (17%) | 9 |

→ **w_prop=1 是甜点**:满属性组织(AUC 1.0 vs vanilla 0.635), 解码器仅小幅代价(65% vs 87%);w_prop=5 把解码器砸塌(17%)。
**= 张力可管理**(正是 CoBO/PG-LBO 平衡的东西), 用作 D4 的健康隐空间。诚实保留:`encode→decode` 忠实重构仍弱(7/80 唯一)
——属性监督重排隐空间, 利于"按活性导航"、不利于"精确复原单分子";对 D4(从先验采样)无碍。

### D4 流匹配 + 梯度引导采样（LaMBO-2 式）—— 结论: 正面, 且生成式先验=防 reward-hacking 的正则 ✅
`molecule/flow_guided_d4.py`: 在 D3 隐(平衡 w_prop=1, 解码器健康)上训整流流 v(z,t)(复用 molgen VelocityField),
引导采样 `z+=dt·(v+s·∇_z g(z))`。诚实三方:unguided(纯流)/ guided / grad-only(去流, 故意 off-manifold)。
**独立评测=真实 DRD2 oracle**(非引导头, 避免自证)。**首跑曾在 w_prop=5 塌解码器上跑→不公平(有效率 0.02), 故重训平衡 D3 后重跑。**

**引导强度 guide_scale 扫描(n=96, 60-ep flow, 真实 DRD2 oracle)**:

| guide_scale | 解码有效率 | 引导头自评 | **真实 DRD2(均)** | active 比例 |
|---|---|---|---|---|
| 0.0（纯流) | 0.85 | 0.11 | 0.108 | 0.11 |
| 0.3 | 0.34 | 0.74 | 0.174 | 0.17 |
| 0.6 | 0.19 | 0.99 | **0.247** | **0.20** |
| ∞（grad-only, 去流) | **0.00** | 1.00 | **0.000** | 0.00 |

**三条诚实结论**:
1. **引导抬高了"真实"活性**(独立 oracle, 非自评):mean_DRD2 0.108→0.247(2.3×), active 11%→20%。
2. **生成式流先验是关键正则**:去掉流(grad-only)→引导头自评满格 1.00 但**真实 DRD2=0、有效率=0** = 隐空间里的
   **reward-hacking / off-manifold 崩塌**(正呼应全篇 oracle-gap 主题)。**流先验=把梯度引导从 reward-hacking 拉回的那道闸**。
3. **诚实代价/可调旋钮**:guide_scale 越大活性越高但有效率越低(0.85→0.19)= 活性↔有效率 Pareto, 由 guide_scale(=LaMBO-2 的
   KL 正则强度)调。实用区在中等 scale(如 0.3 兼顾)。**= "CU-MOBO 可与 flow/扩散生成器联用提分", 但需调引导强度。**

→ 直接回答用户 Q1(能否和 diffusion/flow 联用提分):**能**——梯度引导流采样在独立 oracle 上把真实活性抬到 2.3×,
且暴露了"无生成先验=reward-hacking"这一与本项目核心命题一致的机理。完整 qNEHVI 多目标引导(替单属性头)是顺延工作。

## 6. σ 校准审计（论文 §4.4, 诚实负面 → 反增信）

文献调研一致指出: 方法名 "Calibrated-Uncertainty" 但旧稿从没**测过**校准 = 审稿人头号必杀。补 `benchmark/run_calibration.py`,
在**真实 kcat**(4306 唯一底物, 留出 25%, log10kcat 跨 ~13 个数量级)上测同一 RF 代理两种 σ 的回归校准:

| σ 源 | Spearman ρ(σ,\|err\|)↑ | ENCE↓ | miscal area↓ | NLL↓ |
|---|---|---|---|---|
| RF 树方差 | **0.213**(p=1e-12) | **0.258** | **0.017** | **1.908** |
| OOF 残差("calibrated") | 0.180(p=3e-9) | 0.404 | 0.022 | 1.964 |

**意外但重要的诚实发现**:① 两种 σ 都**只弱校准**(ρ≈0.2);② OOF(专门训来追误差的)在这噪声极大的 oracle 上**反而更差**
(|残差| 本身从指纹近乎学不出来)。→ **在真实噪声 oracle 上"拿到校准 σ"本身才是瓶颈**, 不是默认就有。
合成基准 σ 按构造完美校准(ρ=1.0)是理想区; 真实 ρ=0.2 划出了 σ 机制"保证有效 vs 须先修 UQ"的边界。
**这反而增信**(自我审计 + 数据驱动), 且(a)印证 §4.3 用树方差 σ 没错;(b)指向更强 UQ(深集成/conformal)与 **E2 自适应**
(只罚超阈 σ, 对弱校准鲁棒)。图 `results/calibration.png`(reliability diagram + 误差校准曲线)。

**过度优化曲线(论文 E4, COMs 式)**:`benchmark/plot_overoptimization.py` 把 D4 的 guide_scale 扫描重画成
"真值 vs 优化压力": 压力↑(0→0.3→0.6→∞), 头自评单调 0.11→1.00, 真实活性先升后崩(0.108→0.247→**0**),
reward-hack gap 0.003→**1.0**。= COMs(Trabucco ICML2021 Fig3)"优化越狠真值越掉"招牌图的我方独立-oracle 版。`results/overoptimization.png`。

> 完整调研(4 路: 分子优化 benchmark / 抗 reward-hacking / 校准 / 富集) + 可应用清单 + 成本标注 见 `docs/validation_protocol_research.md`。
> 已做 cheap 三项(校准审计 §4.4、过度优化曲线 E4、related-work 对位 COMs/Design-Bench/DyRAMO); 其余(EF/BEDROC+骨架划分、
> 推荐集多样性、qEHVI 基线、seeds≥5、Design-Bench 锚点)进论文 §6 roadmap, 诚实标注未跑。
