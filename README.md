# CU-MOBO

**校准不确定性多目标贝叶斯优化(Calibrated-Uncertainty Multi-Objective Bayesian
Optimization)** —— 面向**昂贵 + 不确定 oracle** 的、把**校准不确定性作为一等目标**的约束多目标
逆向设计。**原理驱动(贝叶斯优化),不含元启发式**。

方法细节见 `docs/cu_mobo_method.md`,改进与校准审计见 `docs/cu_mobo_improvements.md`,
论文初稿见 `docs/paper_draft.md`。

---

## 1. 问题设定

逆向设计在搜索空间 `x ∈ X`(如生成模型隐空间 `z`)上找高性能候选。每次评估调用一个
**昂贵**且**不确定**的 oracle(如 KcatMoE 给酶动力学 kcat、Boltz 给结构),它返回:

- 预测活性 `μ(x)`(point estimate);
- **校准**不确定性 `σ(x)`(如集成/conformal,σ 与真实误差正相关);
- 其它确定性目标(QED 成药性、SA 可合成性);
- 约束(SMILES 有效、Lipinski、PAINS、尺寸)。

真正想要的是**真实活性 `a_true`**,而非预测 `μ`。

## 2. 失效模式:oracle-gap

朴素 `μ`-贪婪逆向设计会**系统性地利用 oracle 过自信但错误的区域**:在某些 `x`, `μ` 被高估
(看起来高活性)但 `a_true` 低。**校准 σ 在这些区域会升高**(oracle 自己示警),但 μ-贪婪忽略
σ → 落入陷阱 → 预测高、真实低(oracle-gap 大)。CU-MOBO 就是为消除该 gap 而设计。

## 3. 三个原理性机制(方法贡献)

1. **σ 作一等目标**:Pareto 优化 `[μ↑, −σ↑(可信度), QED↑, −SA↑]`。把 `μ−λσ` 标量选择升级成
   **显式前沿** —— 无 λ 超参,风险偏好在前沿上事后取点。
2. **校准 σ 作异方差观测噪声(核心算法新意)**:活性 GP 代理用 oracle 的 `σ²(x)` 作
   **per-point 观测噪声**(`train_Yvar`)。GP 后验**自动下采信高 σ 的不可靠观测、上采信低 σ 的
   可靠观测** → 后验均值是去噪的真实活性估计 → 直接抵抗 oracle-gap。标准 MOBO(qEHVI/qNEHVI)
   假设无噪声或同方差,丢掉了 oracle 给的这份校准信息。
3. **风险感知批量采集 + 约束**:`qNEHVI` 在上述目标上批量(q 点/轮)选点,样本高效;约束用
   可行性加权。

> 2×2 消融(σ-作目标 × σ-作噪声)可独立验证机制 1、2 的贡献。

内核在 `algorithms/cu_mobo.py`,通过 `use_sigma_objective` / `use_sigma_noise` 两个 flag 支持
`naive` / `sigma-obj` / `sigma-noise` / `cu-mobo` 四种组合。

## 4. 验证协议:打破"自评循环"的 in-silico 版

逆向设计论文的头号风险是**循环论证**(用 oracle 引导生成、又用同一 oracle 评估)。本方法用
**合成不确定 oracle 基准**做可控、可证伪的验证:

- 已知**真实活性 `a_true(x)`**,但优化器**看不到**;
- oracle 给 `μ = a_true + 陷阱bump(x)`、`σ = σ0 + γ·bump(x)`(**校准**:σ 跟踪误差);
- 优化器只用 `(μ, σ, qed, sa)` 搜索;
- 最后用隐藏的 `a_true` 算指标:**true-activity↑、oracle-gap = |μ−a_true|↓、陷阱占比↓**。

### 4.1 三块验证证据

| 证据 | 文件 | 说明 |
|---|---|---|
| **① 合成 oracle-gap(机制证明)** | `benchmark/run_oracle_gap.py`、`problems/uncertain_oracle.py` | 隐藏真值 `a_true`,2×2 消融。CU-MOBO true 0.867±0.054 vs naive 0.164(5.3×)、gap 16×↓;qParEGO(μ-only)最差 → σ 机制才是关键 |
| **② 真实 kcat 回顾性富集(真实标签)** | `benchmark/run_retrospective_kcat.py`(`--seeds N`) | 真实实验 kcat 标签留出富集,enrichment 1.26±0.11×、特异性 1.74±0.31× |
| **③ σ 校准审计(诚实负面)** | `benchmark/run_calibration.py` | 真实 kcat 上两个 σ 都只弱校准(Spearman ρ≈0.2),划出"σ 须先真校准"的边界;`results/calibration.png` |

合成基准 σ 按构造完美校准(ρ=1.0)是理想区;真实 ρ=0.2 说明"拿到校准 σ"本身才是瓶颈,指向更强
UQ(深集成/conformal)与自适应惩罚。

## 5. 安装依赖

```powershell
python -m pip install numpy scipy pandas matplotlib pymoo `
    torch botorch gpytorch
```

分子隐空间验证还需 `rdkit` 与本地 `molgen_baseline` 的 `SmilesVAE` 资产。

## 6. 快速运行

```powershell
# ① 合成 oracle-gap + PMO 样本效率(2×2 消融 + qParEGO + NSGA-II)
python -m puridiff_optim.benchmark.run_oracle_gap --seeds 3 --budget 80 --plot
# ② 真实 kcat 回顾性富集(3 seed 误差棒)
python -m puridiff_optim.benchmark.run_retrospective_kcat --seeds 3 --budget 120 --tau 0.4 --plot
# ③ σ 校准审计
python -m puridiff_optim.benchmark.run_calibration --test-frac 0.25 --seed 0 --plot
# 分子隐空间用 CU-MOBO
python -m puridiff_optim.optimize_molecules --method cu_mobo --budget 160 --plot
```

输出:`results/oracle_gap_*.csv/png`、`results/retro_kcat_*.csv/png`、`results/calibration.png`、
`results/mol_cumobo*.csv`。

## 7. 目录结构(CU-MOBO 相关)

```
algorithms/
├── cu_mobo.py            # ★ CU-MOBO 内核(qNEHVI + σ目标 + σ噪声, 2×2 消融 flags)
├── qparego.py            # qParEGO 基线(μ-only, 忽略 σ)
└── mobo.py               # 标准 MOBO 参照
problems/
├── uncertain_oracle.py   # 合成不确定 oracle(隐藏真值 a_true + 陷阱)
└── uncertain_optimum.py  # D2 自适应可信度测试床
surrogate/
├── gp.py                 # GP 代理
└── surrogate_problem.py  # 代理辅助 Problem 封装
molecule/                 # 分子隐空间接口(VAE 解码 + 校准 kcat 代理)
mode_b/                   # 真实人在环异步闭环(propose / ingest)
benchmark/                # CU-MOBO 验证 runner 与绘图
docs/
├── cu_mobo_method.md       # 方法细节
├── cu_mobo_improvements.md # D1–D4 改进 + σ 校准审计
└── paper_draft.md          # 论文初稿(纯 in-silico)
```

## 8. 改进方向(D1–D4,见 `docs/cu_mobo_improvements.md`)

| | 内容 | 结论 |
|---|---|---|
| **D1** | 信赖域 + 校准 σ(类似 TuRBO/LOL-BO) | regime-敏感:小维帮、高维伤,默认关;校准 σ 是真交付 |
| **D2** | 自适应可信度(DyRAMO 式,单边 σ 惩罚) | **最强正面成果**:修 σ-objective 反噬,3-seed 活性最高且最稳 |
| **D3** | 属性感知隐空间(CoBO/LOL-BO 式) | 机制成立(probe AUC 0.63→1.0),但暴露重构↔属性张力 |
| **D4** | 流匹配 + 梯度引导采样(LaMBO-2 式) | 正面:真实 DRD2 活性 2.3×;生成先验=防 reward-hacking 的正则 |

复现命令:

```powershell
python -m puridiff_optim.benchmark.run_trust_region --n-var 16 --width 0.5 --seeds 3 --budget 120 --plot
python -m puridiff_optim.benchmark.run_reliability --n-var 8 --seeds 3 --budget 100 --plot
python -m puridiff_optim.molecule.train_latent_aware --corpus tdc_data/drd2_corpus_hard.csv --prop-col is_drd2_active --out puridiff_optim/molecule/assets/vae_drd2_d3/vae.pt --vanilla puridiff_optim/molecule/assets/vae_drd2_hard/vae.pt --epochs 40 --w-prop 1.0
python -m puridiff_optim.molecule.flow_guided_d4
```

## 9. 诚实边界

- 贡献 = **校准不确定性的一等目标化 + 异方差去噪 + oracle-gap 验证协议**在逆向设计上的原理性
  实例化与实证,属**应用/方法学**贡献,**不主张新优化器原理**;内层 BO/qNEHVI 是昂贵 MOO 的
  强基线。
- 合成基准证明机制有效;真实闭环仍需把候选送**真实 KcatMoE/Boltz**,且 true-hit 用**独立
  oracle** 验证。
- σ 必须**真校准**才有效;未校准 σ 当噪声/目标无益甚至有害(应作对照)。
