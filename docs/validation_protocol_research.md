# 验证流程文献调研 → 论文强化清单（2026-06-25）

> 调研同类论文的**验证/评估协议**, 判断哪些能落到 CU-MOBO 这篇上。4 路并行调研(分子优化 benchmark、
> 抗 reward-hacking、不确定性校准、回顾性富集)。下面=可应用清单 + 成本标注 + 落点。

## 0. 一句话结论
现两腿验证(合成 oracle-gap + 真实 kcat 富集)在**质量/富集**上有竞争力, 但缺三样审稿人视为**必备**的:
**① σ 的校准证据**(方法名就叫 Calibrated-Uncertainty 却从没测过校准)、**② 推荐分子集的多样性/新颖性**、
**③ PMO 式 min-max AUC-Top-K + qEHVI 基线 + COMs 式"过度优化曲线"**。补齐 ①②③ 的 cheap 项即可从
"自建 benchmark"升到"会议可信"。

## 1. 抗 reward-hacking 的验证协议(最核心, 直接对应本文命题)
- **Design-Bench(Trabucco ICML2022, arXiv:2202.08450)+ COMs(ICML2021, arXiv:2107.06882)= "proxy 高但真值低"的标准协议**:
  train proxy → 优化 proxy → 用**独立 true oracle**(湿实验/模拟器/精确查表/独立 holdout)评分 found designs;
  报 **100th + 50th 百分位归一化真值**(`(y−y_min)/(y_max−y_min)`), **N=128 候选, 8 trials, bootstrap CI(rliable/IQM)**。
  **关键基线 = naive gradient-ascent on proxy(= "the reward hacker")**; 关键消融 = **"优化越狠真值越掉"的退化曲线**(COMs Fig3)。
  → **我们的合成基准天然满足"隐藏真值 oracle"要求, 直接采用其评测语法**。我们的 **D4 guide_scale 扫描已是这条曲线的样本**
  (0/0.3/0.6/∞ → 真实活性 0.108/0.174/0.247/**0.000**), 把它画成 COMs 式曲线即可。[cheap]
- **DyRAMO(Nat Commun 2025)**:reward-hacking = 设计漂离训练域(低 max-Tanimoto-similarity MTS)→ 用**适用域 AD(MTS≥ρ)门控**, ρ 用 BO 调。
  招牌证明 = **gefitinib 从训练集移除后仍被重发现**。→ 我们应(a)显式对位(我们=软校准异方差 σ, DyRAMO=硬 AD 门, COMs=对抗惩罚);
  (b)加 MTS 诊断: 证明 μ-only 选解漂离流形(低相似、oracle 不可信处), σ-目标把解留在 AD 内。[medium]
- **LaMBO-2(NeurIPS2023)**:抗钓机制内建于优化器(KL 正则把引导拉回扩散先验 + 自然度阈值约束 + 编辑预算);
  真证据=**374 设计 3 轮湿实验, 99% 表达/40% 结合**。→ 我们 D4 的"去流先验=reward-hack 崩塌"正是同一思想的合成版。

## 2. 分子优化 benchmark 标准(样本效率腿)
- **PMO(Gao NeurIPS2022, arXiv:2206.12411)**:**10k oracle 预算上限**; 主指标 **AUC-Top-K(K=1/10/100), 全部 min-max 归一到[0,1]**
  (奖励"早发现"=样本效率); **23 任务, 5 seeds 评测 + 3 seeds 在 2 个 held-out 任务上调 HP**(防"在测试任务上调参")。
- **GuacaMol(JCIM2019)/MOSES**:分布学习 5 指标(validity/uniqueness/novelty/FCD/KL); **IntDiv_p(集合内多样性)、SNN(到训练集最近邻相似=新颖性)、Scaff(Bemis-Murcko 骨架分布)**。
- **Pareto-aware BO(arXiv:2507.13704)**:HV、R² indicator、**#Circles 多样性**(Tanimoto 阈下两两不相似分子数); 3 seeds 均值±std + **Cohen's d/Cliff's Delta 效应量**。
- → **应加**:① 推荐集 **IntDiv₁ + #Circles + 骨架数 + SNN/novelty vs 训练集**(最大缺口, 见 §4); ② AUC-Top-1/10/100 min-max; ③ **qEHVI/qNEHVI 基线**(现只有 qParEGO/NSGA-II, 缺最强 MOBO 基线很扎眼); ④ seeds≥5。

## 3. 不确定性校准评估(方法名的命门, 现在只断言未测量)
两种校准都要(本文都依赖):**平均/概率校准**(reliability diagram, 区间覆盖=对应 σ²-as-train_Yvar)
+ **误差/离散度/排序校准**(ENCE, Spearman(σ,|err|)=对应 σ-as-Pareto-objective; 常数 σ 可"平均校准"却对排序无用)。
- **指标(从 (y_true,μ,σ) 数组秒算)**:
  - **Spearman ρ(σ,|err|)** = 最直接的"σ 能否排序误差"(scale-free, 头号, 直接撑 σ-目标)。
  - **Reliability diagram + Miscalibration area**: 标准化残差 z=(y−μ)/σ; 对覆盖度 p, 观测 `mean(|z|≤norm.ppf(0.5+p/2))` vs p; 面积=∫|obs−p|。
  - **ENCE(N=10~15 bin)**: 按 σ 分箱, **RMV_j=√mean(σ²), RMSE_j=√mean((y−μ)²)**, ENCE=mean|RMV−RMSE|/RMV; 画 **RMSE_j vs RMV_j 误差校准曲线**(分子 UQ 标准图)。
  - 辅助: NLL(高斯)、sharpness=√mean(σ²)、Cv=std(σ)/mean(σ)(近 0 则 σ 无信息=毁本文前提)、conformal 覆盖(分布无关)。
- **参考**: Kuleshov ICML2018; Levi ENCE(Sensors2022); Hirschfeld JCIM2020(分子模板: ECE 区间形 + ENCE); Scalia JCIM2020; Tran MLST2020 + Uncertainty Toolbox; Pernot arXiv:2305.11905(ENCE 偏置 caveat)。
- → **加一张 reliability diagram + 误差校准曲线 + Spearman/ENCE 表, 对比 tree-var σ vs OOF-校准 σ**(arrays 现成, 秒级)。这是审稿人**必看**那张图。[cheap, 最高性价比]

## 4. 回顾性富集(§4.3 kcat)的严格化
- **指标**: **EF(top 1%/5%) + BEDROC(α=80.5, 现 ML-VS 标准) + ROC-AUC**(`rdkit.ML.Scoring.Scoring`); AUC 被明确**贬为次要**(对早识别盲, Truchon&Bayly JCIM2007)。
- **现 §4.3 三处会被攻击**(诚实):
  - **(a) Tanimoto≥0.4 太松**: 随机 ZINC 对≈0.1–0.2, 0.4 只是"粗结构沾边"非"重发现"; GuacaMol/PMO 重发现以 **Tc=1.0 精确匹配**为金标, 有意义阈值 **≥0.7(近似物 0.85)**。
  - **(b) 随机划分有 analogue-bias**: 留出 HIGH 的近邻仍在训练集→"recover@0.4"近乎记忆近重复(LIT-PCBA arXiv:2507.21404: 无参数 1-NN Tanimoto 基线就能打平 SOTA)。应**Bemis-Murcko 骨架划分** + **去近邻控制(剔训练集中 Tc≥0.6/0.85 到任一留出 active 的分子)再测**。
  - **(c) recovery_CU/recovery_random 基线弱**: 应加 **1-NN Tanimoto 记忆基线** + **uncalibrated-MOBO 消融**(证明是"校准"而非"优化器/VAE 先验"在起效) + bootstrap CI/置换检验(Ash&Hughes-Oliver JCheminf2022)。3 seeds/60 active 功效不足→≥5 seeds。
- **重写 §4.3 框架(一段)**: 按 acquisition 分排全库 → 报 EF_1%/5% + BEDROC(80.5) + AUC, 在随机划分 + 骨架划分双设定下, 带 Tc≥0.6 去近邻控制; recovery@τ(τ∈{0.4,0.7,exact})作互补相似视角; bootstrap CI + 置换检验 ≥5 seeds; 基线 = random + 1-NN + uncalibrated-MOBO。

## 5. 落地优先级
- **[✅ 已做]** ① σ 校准评估(`run_calibration.py`, 真实 kcat, tree-var vs OOF, reliability+ENCE+Spearman 表+图)= 论文 §4.4。**诚实负面**:两 σ 都弱校准(ρ≈0.2), OOF 反不如树方差。
- **[✅ 已做]** ② D4 重构成 COMs 式"过度优化曲线"(`plot_overoptimization.py`, gap 0.003→1.0)= 论文 E4。
- **[✅ 已做]** ③ 论文 §2 对位 DyRAMO/COMs/Design-Bench/PMO + §6 验证路线图(诚实标 done/planned)。
- **[✅ 已做]** ④ §4.3 严格化(`run_retrospective_kcat_hardened.py`):**EF/BEDROC/AUC + 随机 vs 骨架划分 + 1-NN 记忆基线 + bootstrap CI(5 seed)**。
  结果(诚实): surrogate 随机 AUC 0.799[.78,.82]→骨架 0.651[.62,.69](CI 排除 0.5, 真泛化但 modest); 随机→骨架大跌量化了 analogue-bias;
  AUC 两划分都胜 1-NN, 但随机早富集 BEDROC 与 1-NN 相当(0.49 vs 0.51)= 正说明骨架划分+记忆基线非可选。已写入 paper §4.3。
- **[✅ 已做]** ⑤ 推荐集多样性/新颖性(`run_diversity.py`, 3 seed)= 论文 §4.5。**诚实双发现**:CU-MOBO **不塌多样性**
  (IntDiv 0.86>naive 0.84, #Circles 19>16, 骨架 5.7>4.0, 质量也更高)→ 反驳"靠贴训练流形作弊";**但新颖性更低**
  (SNN 0.76 vs naive 0.66 vs random 0.60; 新颖比例 9.5% vs 12% vs 21%)= σ 偏向低不确定(近训练可信)区的**必然代价** → 新颖性↔可信度张力, 指向 E2/降权 σ。
- **[✅ 已做]** ⑥ 外部锚点 = 标准 TDC oracle(QED-only, `run_tdc_anchor.py`, 3 seed)= 论文 §4.6。surrogate σ="calibrated"(OOF)→ **同时**测 σ-目标+校准异方差噪声两半, 无循环(代理≠裁判)。
  两个标注预算: QED-rich(600, 平滑/无 gap → CU-MOBO 无悔+温和获益, topk 0.894 最高/gap 0.056 最低/方差最小); QED-scarce(120, 仍学得好仍无大 gap → 胜 naive δ+0.56 但 σ-noise 略胜)。
  **稳定结论: CU-MOBO oracle-gap 最低(0.056)/并列最低(0.078) vs naive 0.068/0.078**, 活性不降反略升 = 无悔。诚实点: QED 太指纹可学(近双射)造不出大 gap → 只能测"无悔"非"抗gap" → 故 §4.1 合成 + §4.3 kcat 才是决定性证据。**DRD2 最终移除出 §4.6(用户 2026-06-29 拍板, 只留 QED)**, 代码留存不入稿。
  **配图(2026-06-29)**: 主图 TikZ(`docs/figures/main_figure.tex`→`results/fig1_main.png`)+ §4.6 图(`benchmark/plot_tdc_anchor.py`→`results/tdc_anchor.png`); paper 插 8 图; LaTeX 环境备在 `docs/figures/figures_latex.tex`。
- **[受阻, 诚实记录]** Design-Bench TF-Bind-8 外部锚点: design-bench 已装(UTF-8 setup 修复)+ 可 import(deepchem stub), 但上游数据托管 GCS bucket `design-bench` 已失效(NoSuchBucket, 2026-06-26 直连确认), 无镜像 → 无法获取 shard。判为外部基础设施死亡, 改由上面 TDC 锚点(§4.6)承担。GSK3B/JNK3 因 pickled RF oracle 的 sklearn 版本不兼容亦略。
- **[roadmap, medium/expensive 未做]** qEHVI 基线、生成腿 Tc≥0.7 阈值、MTS-AD 诊断。(合成 seeds≥5+Cliff's δ、生成腿 Tc≥0.6 去近邻 均已并入 §4.1/§4.3。)

## 6. 关键引用(投稿用)
PMO arXiv:2206.12411 · GuacaMol arXiv:1811.09621 · MOSES Front.Pharmacol.2020 · Pareto-aware-BO arXiv:2507.13704 ·
Design-Bench arXiv:2202.08450 · COMs arXiv:2107.06882 · LaMBO-2 arXiv:2305.20009 · DyRAMO 10.1038/s41467-025-57582-3 ·
Kuleshov ICML2018 · Levi ENCE(Sensors2022, arXiv:1905.11659) · Hirschfeld JCIM2020(10.1021/acs.jcim.0c00502) ·
Scalia JCIM2020(arXiv:1910.03127) · Pernot arXiv:2305.11905 · Truchon&Bayly JCIM2007(10.1021/ci600426e) ·
Ash&Hughes-Oliver JCheminf2022(10.1186/s13321-022-00629-0) · Tossou scaffold-split arXiv:2406.00873 · LIT-PCBA leakage arXiv:2507.21404。
