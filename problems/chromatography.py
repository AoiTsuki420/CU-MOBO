"""合成层析策略优化问题 (IEX -> HIC), 对接原型方案第 5/6/7 节.

这是一个**解析代理**(analytic surrogate) 的工艺评分模型, 用来在没有 CADET /
可微混合模型时把整条 "参数 -> 多目标 + 约束 -> Pareto 推荐" 闭环跑通.
之后 M4 只需把 `simulate()` 换成 CADET 机理或可微混合模型后端, 接口不变.

决策变量 x (物理量纲, 6 维):
    IEX_pH         离子交换洗脱 pH        [4.5, 7.5]
    IEX_load_mg_mL 上样负载                [5,   50]
    IEX_grad_cv    IEX 梯度长度 (CV)       [5,   30]
    HIC_salt_M     HIC 起始盐浓度          [0.5, 2.0]
    HIC_grad_cv    HIC 梯度长度 (CV)       [5,   30]
    pool_width     收集窗口宽度 (归一化)    [0.2, 1.0]

目标 (内部统一最小化, F = [-purity, -yield, cost]):
    purity   纯度        (越大越好)
    yield    收率        (越大越好)
    cost     时间+缓冲    (越小越好)

约束 g(x) <= 0:
    hcp_min - hcp_removal <= 0   (HCP 去除率下限)
    risk    - risk_max    <= 0   (风险上限)

体现的工艺权衡 (符合层析常识):
    - 梯度越长 -> 分离度越高 (纯度↑) 但时间/缓冲↑、收率略↓;
    - 收集窗口越窄 -> 纯度↑、收率↓ (经典纯度-收率矛盾);
    - 负载越高 -> 产率/通量↑ 但过载时纯度↓、风险↑;
    - pH / 盐 偏离最优 -> 选择性↓.
"""

from __future__ import annotations

import numpy as np

from ..core.dominance import non_dominated_mask
from ..core.problem import Problem

_VAR_NAMES = ["IEX_pH", "IEX_load_mg_mL", "IEX_grad_cv",
              "HIC_salt_M", "HIC_grad_cv", "pool_width"]
_XL = np.array([4.5, 5.0, 5.0, 0.5, 5.0, 0.2])
_XU = np.array([7.5, 50.0, 30.0, 2.0, 30.0, 1.0])


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def simulate(X):
    """解析工艺仿真. X: (N, 6) 物理量纲. 返回 dict of (N,) 数组."""
    X = np.atleast_2d(np.asarray(X, float))
    pH, load, gcv_iex = X[:, 0], X[:, 1], X[:, 2]
    salt, gcv_hic, pool = X[:, 3], X[:, 4], X[:, 5]

    # 梯度分离度 (越长越好, 饱和)
    R_iex = 1.0 - np.exp(-gcv_iex / 12.0)
    R_hic = 1.0 - np.exp(-gcv_hic / 12.0)
    # pH / 盐 选择性 (最优点附近高斯)
    sel_pH = np.exp(-((pH - 6.0) / 1.2) ** 2)
    sel_salt = np.exp(-((salt - 1.2) / 0.5) ** 2)
    # 过载对纯度的惩罚 (负载高于 ~35 后下降)
    overload = _sigmoid(-(load - 35.0) / 6.0)
    # 综合分离质量 [0,1]
    sep = R_iex * R_hic * (0.5 + 0.5 * sel_pH) * (0.5 + 0.5 * sel_salt) * overload

    # 纯度: 窗口越宽纳入越多杂质 -> 纯度越低
    purity = 1.0 - (1.0 - sep) * (0.4 + 0.6 * pool)
    purity = np.clip(purity, 0.0, 0.999)
    # 收率: 窗口越宽收率越高; 负载提升质量回收; 梯度损失略减
    yield_ = (0.3 + 0.7 * pool) * (0.6 + 0.4 * (load / 50.0)) * (1.0 - 0.15 * (R_iex + R_hic) / 2.0)
    yield_ = np.clip(yield_, 0.0, 0.999)
    # HCP 去除率 ~ 分离质量, 宽窗口略降
    hcp = np.clip(1.0 - (1.0 - sep) * (0.5 + 0.5 * pool), 0.0, 0.999)
    # 成本: 时间 + 缓冲, 主要来自梯度长度 (归一化到 ~[0,1])
    time_min = (gcv_iex + gcv_hic) * 2.0 + 20.0
    buffer_L = (gcv_iex + gcv_hic) * 0.1 + 1.0
    cost = time_min / 160.0 + buffer_L / 7.0
    # 风险: 高负载 / 偏离最优 pH、盐
    risk = 0.5 * _sigmoid((load - 40.0) / 4.0) + 0.3 * (1.0 - sel_pH) + 0.2 * (1.0 - sel_salt)
    risk = np.clip(risk, 0.0, 1.0)

    return {"purity": purity, "yield": yield_, "hcp_removal": hcp,
            "cost": cost, "risk": risk}


class ChromatographyStrategyProblem(Problem):
    """IEX->HIC 策略多目标优化 (合成解析后端)."""

    def __init__(self, hcp_min=0.80, risk_max=0.55, simulator=None):
        super().__init__(n_var=6, n_obj=3, xl=_XL, xu=_XU,
                         n_constr=2, name="iex_hic_strategy")
        self.var_names = list(_VAR_NAMES)
        self.obj_names = ["purity", "yield", "cost"]
        self.hcp_min = float(hcp_min)
        self.risk_max = float(risk_max)
        # 可插拔后端: 默认解析 simulate, 之后可换 CADET / 可微混合模型
        self.simulator = simulator if simulator is not None else simulate

    def _evaluate(self, X):
        s = self.simulator(X)
        F = np.column_stack([-s["purity"], -s["yield"], s["cost"]])  # 最小化
        G = np.column_stack([self.hcp_min - s["hcp_removal"], s["risk"] - self.risk_max])
        return F, G

    # -- 把解向量翻译成可读策略 (供 Pareto 推荐 / 前端展示) ------------------
    def decode(self, x):
        x = np.asarray(x, float).ravel()
        s = self.simulator(x[None, :])
        params = {n: float(v) for n, v in zip(self.var_names, x)}
        kpis = {k: float(v[0]) for k, v in s.items()}
        return {"params": params, "kpis": kpis}

    def reference_front(self, n_samples=200000, seed=0):
        """蛮力采样近似真实 (可行) Pareto 前沿, 用于 IGD/可视化参照."""
        rng = np.random.default_rng(seed)
        X = self.xl + rng.random((n_samples, self.n_var)) * (self.xu - self.xl)
        F, G = self.evaluate(X)
        feas = np.all(G <= 0, axis=1)
        Ff = F[feas]
        if len(Ff) == 0:
            return None
        return Ff[non_dominated_mask(Ff)]
