"""合成"高σ最优"基准 —— 用于验证 D2 自适应可信度修 σ-objective 反噬.

复刻我们 Path B 实测到的失效结构: **真实最优本身处于高不确定区**(稀有/新颖 active),
而"安全但平庸"区 σ 低。于是:
  - naive(max μ)            → 落进 **陷阱 t_star**(μ 虚高、真实低、σ 极高);
  - σ-objective(min σ)      → 躲进 **安全区 safe**(σ 低但真实平庸)→ **反噬**(避开了真最优);
  - **D2 自适应可信域约束**  → 只排除"极高 σ"的陷阱、保留"中等 σ"的真最优 → 触达 r_star。

三处(均在 [0,1]^n):
  r_star = 真最优      : 真实活性高,  σ 中等(γ_r), μ≈真实;
  safe   = 安全诱饵    : 真实活性平庸, σ 低,        μ≈真实;
  t_star = oracle 陷阱 : 真实活性低,  σ 极高(γ_t), μ 虚高(+trap)。

oracle 输出 {mu, sigma, qed, sa}(同 UncertainOracleProblem 约定), true_activity 隐藏。
"""

from __future__ import annotations

import numpy as np


class UncertainOptimumProblem:
    def __init__(self, n_var=8, width=0.45, A_opt=1.0, A_safe=0.45, A_trap=1.4,
                 sigma0=0.03, gamma_r=0.25, gamma_t=0.6, baseline=0.05):
        self.n_var = int(n_var)
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)
        alt = np.array([0.8 if i % 2 == 0 else 0.2 for i in range(self.n_var)])
        self.r_star = alt                       # 真最优(高真实, σ 中等)
        self.t_star = 1.0 - alt                 # 陷阱(μ 虚高, 真实低, σ 极高)
        self.safe_star = np.full(self.n_var, 0.5)   # 安全诱饵(σ 低, 真实平庸)
        self.q_star = np.full(self.n_var, 0.5)
        self.width = width
        self.A_opt, self.A_safe, self.A_trap = A_opt, A_safe, A_trap
        self.sigma0, self.gamma_r, self.gamma_t, self.baseline = sigma0, gamma_r, gamma_t, baseline
        self.obj_names = ["mu", "sigma", "qed", "sa"]

    def _bump(self, X, c):
        d2 = np.sum((X - c) ** 2, axis=1)
        return np.exp(-d2 / (2 * self.width ** 2))

    # -- 隐藏真值(仅评估) --
    def true_activity(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        return (self.A_opt * self._bump(X, self.r_star)
                + self.A_safe * self._bump(X, self.safe_star) + self.baseline)

    # -- oracle(优化器可见) --
    def evaluate_oracle(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        a = self.true_activity(X)
        trap = self.A_trap * self._bump(X, self.t_star)
        mu = a + trap                                          # 陷阱处 μ 虚高
        sigma = (self.sigma0 + self.gamma_r * self._bump(X, self.r_star)   # 真最优处 σ 中等(关键!)
                 + self.gamma_t * self._bump(X, self.t_star))              # 陷阱处 σ 极高
        dq = np.sum((X - self.q_star) ** 2, axis=1)
        qed = np.exp(-dq / (2 * 0.3 ** 2))
        sa = 1.0 + 4.0 * np.mean(X, axis=1)
        return {"mu": mu, "sigma": sigma, "qed": qed, "sa": sa}

    def at_optimum(self, X, tol=0.25):
        X = np.atleast_2d(np.asarray(X, float))
        return np.linalg.norm(X - self.r_star, axis=1) / np.sqrt(self.n_var) < tol
