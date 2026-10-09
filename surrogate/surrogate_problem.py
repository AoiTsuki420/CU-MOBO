"""把 GP 代理包成 Problem, 供内层 MO-ALA 在"便宜"代理上搜索.

目标: 返回乐观估计 mean - kappa*std (最小化语义下, 越不确定越被鼓励探索).
约束: 若代理学了约束, 返回约束后验均值, 让内层搜索具备可行性意识 (Deb 约束支配).
kappa=0 即纯均值 (开发); kappa>0 偏探索.
"""

from __future__ import annotations

import numpy as np

from ..core.problem import Problem


class SurrogateProblem(Problem):
    def __init__(self, base_problem, surrogate, kappa=0.0, use_constraints=True):
        n_constr = base_problem.n_constr if (use_constraints and surrogate.cmodel is not None) else 0
        super().__init__(
            n_var=base_problem.n_var, n_obj=base_problem.n_obj,
            xl=base_problem.xl, xu=base_problem.xu,
            n_constr=n_constr, name=f"surrogate({base_problem.name})",
        )
        self.surrogate = surrogate
        self.kappa = float(kappa)

    def _evaluate(self, X):
        mean, std = self.surrogate.predict(X)
        F = mean - self.kappa * std
        G = self.surrogate.predict_constraints(X) if self.n_constr > 0 else None
        return F, G
