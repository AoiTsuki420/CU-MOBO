# puridiff_optim

重组小分子蛋白纯化策略的**多目标优化器内核**。研究主线:

> **方向 2 兜底**(成熟 MOBO/qEHVI)+ **方向 1 创新**(代理辅助多目标 ALA)。

本包是**独立模块**,不依赖任何私有平台仓库(`backend/frontend`)或 `CADET_DPNN`
可微主线。真实评估通过可插拔的 `Problem` / `Simulator` 接口注入,所以现在可以脱离
那两部分单独开发、benchmark 与发表对照。

---

## 1. 目录结构

```
puridiff_optim/
├── core/
│   ├── problem.py      # Problem 抽象 (统一最小化, 支持约束 g(x)<=0)
│   ├── dominance.py    # Pareto / 约束支配, 非支配排序, 拥挤度
│   ├── archive.py      # 外部有界 Pareto 档案 (拥挤度截断 + leader 选择)
│   └── operators.py    # Lévy 飞行 (Mantegna), 边界裁剪
├── algorithms/
│   ├── ala.py          # ALA 单目标 (忠实复现官方 MATLAB)
│   ├── mo_ala.py       # ★ MO-ALA 多目标 (本项目创新点)
│   └── mopso.py        # MOPSO 多目标基线
├── problems/
│   └── benchmark.py    # ZDT/DTLZ 适配 (pymoo, 提供真实 PF)
├── benchmark/
│   └── run_benchmark.py# 统一 runner: 多算法 x 多问题 x 多 seed -> HV/IGD+
└── results/            # 输出: CSV 汇总 + 收敛曲线/前沿图
```

## 2. 安装依赖

```powershell
python -m pip install numpy scipy pandas matplotlib pymoo
```
(本机 base env 已具备 numpy/scipy/pandas/matplotlib,补装 `pymoo` 即可。)

## 3. 快速跑 benchmark

```powershell
python -m puridiff_optim.benchmark.run_benchmark `
    --problems zdt1 zdt2 zdt3 dtlz2 --algos mo_ala mopso nsga2 `
    --seeds 5 --max-fes 10000 --pop 100 --plot
```
输出:
- `results/benchmark_raw.csv` —— 每 (问题,算法,seed) 的 HV / IGD+ / 前沿规模 / 耗时
- `results/benchmark_summary.csv` —— 按 (问题,算法) 聚合的均值±std
- `results/convergence_<problem>.png` —— HV 随函数评估次数收敛曲线
- `results/front_<problem>.png` —— 2 目标问题最终 Pareto 前沿对比

## 4. MO-ALA 是什么(创新点)

原始 **Artificial Lemming Algorithm (ALA)** (Xiao 等, *Artificial Intelligence
Review*, 2025) 是**单目标**算法,目前文献中尚无多目标版本。本包把它扩展为多目标:

| 单目标 ALA | MO-ALA (本包) |
|---|---|
| 全局唯一最优 `Position` | 每个体从**外部 Pareto 档案**按拥挤度二元锦标赛选 `leader` |
| 标量 `f1 < f2` 贪婪比较 | **Deb 约束支配**比较 |
| 无前沿概念 | 有界外部档案(非支配 + 拥挤度截断)即输出前沿 |
| 能量切换 / 四行为 / Brownian / Lévy | **完整保留** |

四行为公式忠实复现官方 `ALA.m`:
- 能量:`θ = 2·atan(1 − FEs/MaxFEs)`,`E = 2·ln(1/rand)·θ`;`E>1`→探索,否则→开发
- 探索:`rand<0.3` 长距迁徙(Eq 3.2)/ 否则挖洞(Eq 3.6)
- 开发:`rand<0.5` 觅食(螺旋, Eq 3.8)/ 否则躲避天敌(Lévy, Eq 3.11)
- 预算单位 = **函数评估次数 FEs**(非迭代数)→ 与"昂贵评估"代理框架天然对齐

### 4.1 MO-ALA+ (M1.x 增强, `mo_ala_plus`)

base MO-ALA 在 ZDT2 偶发早熟坍缩、DTLZ2(3目标)偏弱。MO-ALA+ 在**不改 ALA 核心
行为**的前提下加两个策略(类比 CALA 给 ALA 加策略):

1. **停滞重启 + 反向学习(OBL)再注入**:监测种群归一化多样性, 坍缩时围绕档案
   leader 取反向点 + LHS 重注入部分种群 → 根治早熟坍缩;
2. **ε-支配档案**(`EpsilonArchive`):ε 网格保持前沿均匀, 改善多目标分布。

base 版 `mo_ala` 保持忠实不变, 作为消融对照。

## 5. 接层析问题(下一步 M3)

实现一个 `Problem` 子类即可接入,把"纯度↑/收率↑"取负转成最小化:

```python
class ChromatographyProblem(Problem):
    def __init__(self, simulator):
        # x = [IEX_pH, load_mg_mL, HIC_salt_m, gradient_cv, ...]
        super().__init__(n_var=..., n_obj=3, xl=..., xu=...,
                         n_constr=2, name="iex_hic")
        self.sim = simulator
    def _evaluate(self, X):
        purity, yield_, cost, hcp, risk = self.sim(X)   # 可插拔后端
        F = np.column_stack([-purity, -yield_, cost])   # 主目标 (最小化)
        G = np.column_stack([hcp_threshold - hcp, risk - risk_max])  # g<=0 可行
        return F, G
```
`simulator` 后端三选一(规则评分 / CADET 机理 / 可微混合模型),互不影响内核。

## 6. 路线图

- [x] **M1** 优化器内核 + MO-ALA + 标准 MOO benchmark 对比 NSGA-II/MOPSO
  - MO-ALA 在 ZDT1/ZDT3 优于 NSGA-II, 收敛快 ~5×
- [x] **M1.x** MO-ALA+ 算法增强 (`mo_ala_plus`): 停滞重启(OBL再注入) + ε-支配档案
  - ZDT2 早熟坍缩根治 (seed 崩溃 HV 0.11→0.54); DTLZ2(3目标) ε-档案 HV 0.572→0.615
- [x] **M2** 代理辅助闭环(GP 代理)+ 样本效率对比 MOBO/qEHVI
  - ZDT1-6D / 120 真实评估: SA-MO-ALA HV=0.869 > MOBO 0.621 > 纯MO-ALA 0.491
- [x] **M3** 合成层析问题端到端 + Pareto 策略推荐 (`optimize_strategies.py`)
  - 约束层析问题: SA-MO-ALA 160 评估 ≈ MO-ALA 8000 评估 (HV 0.463 vs 0.445), ~50× 样本效率
- [ ] **M4** 接 CADET 机理 / 可微混合模型作为 Simulator (替换 `chromatography.simulate`)
- [ ] **M5** 3 个重组小分子蛋白案例 + 论文图表 + `POST /strategies/optimize` FastAPI 封装

### 复现命令

```powershell
# M1 标准 benchmark
python -m puridiff_optim.benchmark.run_benchmark --seeds 5 --max-fes 10000 --plot
# M2 样本效率对比
python -m puridiff_optim.benchmark.run_surrogate_comparison --problem zdt1 --n-var 6 --budget 120 --seeds 5 --plot
# M3 层析策略 Pareto 推荐
python -m puridiff_optim.optimize_strategies --method sa_mo_ala --budget 160 --plot
```

## 7. 分子逆向设计集成(与 molgen_baseline 耦合)

把优化器接到 SMILES 生成模型(`molgen_baseline` 的 `SmilesVAE` 连续隐空间),做
**隐空间多目标分子逆向设计**(`puridiff_optim/molecule/` + `optimize_molecules.py`):

```
latent z --VAE.generate--> SMILES --(kcat 代理 + RDKit)--> (−kcat μ, σ, SA, −QED)
        + 约束(有效/Lipinski/PAINS/最小尺寸) --> MO-ALA+(从真实分子隐编码 warm-start)
        --> 分子 Pareto 前沿(活性↑ vs 不确定性↓ vs 可合成↑ vs 成药↑)
```

- `VaeDecoder` 复用 molgen 的 `SmilesVAE`(z↔SMILES,greedy 确定性解码)
- `KcatSurrogate` 从 `ranked_candidates.csv` 训 Morgan-FP 随机森林 → (kcat μ, σ)
- `MoleculeLatentProblem`:z∈R⁴⁸ → 4 目标 + 4 约束
- 直接服务 molgen 论文主张:把 `μ−λσ` **标量选择**升级成 **(μ,σ) 显式 Pareto**

```powershell
python -m puridiff_optim.optimize_molecules --max-fes 4000 --n-agents 80 --plot
```

> ⚠️ 本地 demo:VAE 为 CPU 小模型(随机隐 ~66% 有效),kcat 用本地代理(KcatMoE 的
> proxy,仅 100 分子)。**真实研究闭环(Mode B)**:每批候选送真实 KcatMoE/Boltz,
> true-hit 用独立 oracle(见 molgen `PAPER_PLAN_inverse_design.md` §3);优化 μ 又用同源
> 信号评估存在循环论证,σ 作显式目标是缓解而非消除。此时换 `sa_mo_ala`(样本高效)。

## 8. CU-MOBO:校准不确定性多目标贝叶斯优化(主线方法)

面向**昂贵 + 不确定 oracle** 的、把**校准不确定性作为一等目标**的约束多目标逆向设计。
**原理驱动(贝叶斯优化),不含元启发式**。方法细节见 `docs/cu_mobo_method.md`。

三个机制:
1. **σ 作一等 Pareto 目标**(去掉 `μ−λσ` 的 λ);
2. **校准 σ² 作活性 GP 的异方差观测噪声**(`train_Yvar`)→ 去噪、下采信不可靠观测,
   抗 oracle-gap(核心算法新意);
3. **qNEHVI 批量风险感知采集** + 约束。

验证(打破自评循环的 in-silico 版):合成不确定 oracle(隐藏真值 `a_true`,陷阱区 μ
虚高 + 校准 σ 升高),用隐藏 `a_true` 算 true-activity / oracle-gap,做 **2×2 消融**
(σ-作目标 × σ-作噪声)+ NSGA-II 参照。

```powershell
# 合成 oracle-gap 验证(2x2 消融)
python -m puridiff_optim.benchmark.run_oracle_gap --seeds 3 --budget 80 --plot
# 分子隐空间用 CU-MOBO
python -m puridiff_optim.optimize_molecules --method cu_mobo --budget 160 --plot
```

> 诚实定级:不是新优化器原理,而是 BO 内的原理性实例化 + oracle-gap 验证协议(应用/
> 方法学贡献)。ALA 系列(`mo_ala*` / `sa_mo_ala`)保留作历史基线,主线不再使用。

### 8.1 三块验证证据 + 论文初稿

论文初稿:`docs/paper_draft.md`(英文,投 JCIM/Digital Discovery 档,纯 in-silico)。两块验证证据:

| 证据 | 文件 | 说明 |
|---|---|---|
| **① 合成 oracle-gap(机制证明)** | `benchmark/run_oracle_gap.py`、`problems/uncertain_oracle.py` | 隐藏真值 `a_true`,2×2 消融。CU-MOBO true 0.867±0.054 vs naive 0.164(5.3×)、gap 16×↓;qParEGO(μ-only)最差 → σ 机制才是关键 |
| **② 真实 kcat 回顾性富集(真实标签)** | `benchmark/run_retrospective_kcat.py`(`--seeds N`) | 真实实验 kcat 标签留出富集,enrichment 1.26±0.11×、特异性 1.74±0.31× |

```powershell
# ① 合成 oracle-gap + PMO(3 seed)
python -m puridiff_optim.benchmark.run_oracle_gap --seeds 3 --budget 80 --plot
# ② 真实 kcat 富集(3 seed 误差棒)
python -m puridiff_optim.benchmark.run_retrospective_kcat --seeds 3 --budget 120 --tau 0.4 --plot
```

> **真实 DRD2 oracle 压力测试(探索, 不入论文)**:`benchmark/run_drd2_gap.py` + `problems/drd2_oracle.py`
> 已建并跑过,但在「16 维 + 小预算 + 稀有 active」regime 下 CU-MOBO 不及 NSGA-II、σ 机制无稳定收益
> → 砍出论文,代码留存。配套基础设施:`molecule/_tdc_compat.py`(PyTDC `--no-deps` + `rdkit.six` 垫片,
> 现代 rdkit/Windows 可用)、`molecule/train_drug_vae.py`(抗后验坍缩 drug-like VAE: word-dropout + free-bits)。

## 9. 参考文献

- Y. Xiao, H. Cui, R. Abu Khurma, P. A. Castillo. *Artificial lemming algorithm: a
  novel bionic meta-heuristic technique for solving real-world engineering
  optimization problems.* Artificial Intelligence Review, 2025.
- C. A. Coello Coello, G. T. Pulido, M. S. Lechuga. *Handling multiple objectives
  with particle swarm optimization.* IEEE TEC, 2004.
- K. Deb et al. *A fast and elitist multiobjective genetic algorithm: NSGA-II.*
  IEEE TEC, 2002.
