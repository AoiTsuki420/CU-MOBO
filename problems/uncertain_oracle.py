"""合成"不确定 oracle"基准 —— 用于可控验证 CU-MOBO 抗 oracle-gap.

已知真实活性 a_true(x) (优化器看不到). oracle 返回:
    μ(x)  = a_true(x) + 陷阱bump(x)        预测活性 (陷阱区被高估)
    σ(x)  = σ0 + γ·陷阱bump(x)              校准不确定性 (σ 跟踪误差, 陷阱区升高)
    qed(x), sa(x)                          其它确定性目标

优化器只用 (μ, σ, qed, sa) 搜索; 评估时用隐藏的 a_true 算 true-hit / oracle-gap,
从而打破"自评循环"(molgen PAPER_PLAN §3 的 in-silico 版).

oracle 输出 schema (CU-MOBO 约定): dict(mu, sigma, qed, sa), 均为 (N,) 数组.
    mu  越大越好 (活性, 带校准噪声 sigma)
    sigma 越小越好 (可信度)
    qed 越大越好
    sa  越小越好
"""

from __future__ import annotations

import numpy as np


class UncertainOracleProblem:
    def __init__(self, n_var=6, A_reliable=1.0, A_trap=1.6, width=0.35,
                 sigma0=0.03, gamma=0.5, baseline=0.05):
        self.n_var = int(n_var)
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)
        # r* 与 t* 取对称交替向量: 二者 mean(x) 相同、到 q* 距离相同
        # => SA、QED 在两处完全相等, 唯一区别是 μ(t* 虚高) 和 σ(t* 高) 与真实活性.
        alt = np.array([0.8 if i % 2 == 0 else 0.2 for i in range(self.n_var)])
        self.r_star = alt                          # 可靠的真实活性最优
        self.t_star = 1.0 - alt                    # 陷阱 (μ 虚高, a_true 低)
        self.q_star = np.full(self.n_var, 0.5)     # qed 最优
        self.A_reliable = A_reliable
        self.A_trap = A_trap
        self.width = width
        self.sigma0 = sigma0
        self.gamma = gamma
        self.baseline = baseline
        self.obj_names = ["mu", "sigma", "qed", "sa"]

    # -- 隐藏真值 (仅供评估) --------------------------------------------------
    def true_activity(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        d2 = np.sum((X - self.r_star) ** 2, axis=1)
        return self.A_reliable * np.exp(-d2 / (2 * self.width ** 2)) + self.baseline

    def _trap(self, X):
        d2 = np.sum((X - self.t_star) ** 2, axis=1)
        return self.A_trap * np.exp(-d2 / (2 * self.width ** 2))

    # -- oracle (优化器可见) --------------------------------------------------
    def evaluate_oracle(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        a = self.true_activity(X)
        trap = self._trap(X)
        mu = a + trap                         # 陷阱区 μ 虚高
        sigma = self.sigma0 + self.gamma * trap   # 校准: σ 跟踪误差(=trap)
        dq = np.sum((X - self.q_star) ** 2, axis=1)
        qed = np.exp(-dq / (2 * 0.3 ** 2))
        sa = 1.0 + 4.0 * np.mean(X, axis=1)   # 高 x → 高合成成本 (与 r* 形成张力)
        return {"mu": mu, "sigma": sigma, "qed": qed, "sa": sa}

    # 便于报告: 陷阱判定 (σ 高 = 在陷阱/不可信区)
    def in_trap(self, X, sigma_thresh=0.15):
        return self.evaluate_oracle(X)["sigma"] > sigma_thresh
