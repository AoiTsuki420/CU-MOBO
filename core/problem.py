"""Problem 抽象.

约定 (统一为最小化):
    F  形状 (N, n_obj)   —— 所有目标都最小化 (越小越好)
    G  形状 (N, n_constr)—— 不等式约束, g(x) <= 0 为可行; 无约束时为 None

层析场景里 "纯度↑/收率↑" 这类越大越好的目标, 在 Simulator 适配层取负号后
再喂进来, 保证内核统一按最小化处理.
"""

from __future__ import annotations

import numpy as np


class Problem:
    """最小化问题基类. 子类实现 `_evaluate(X) -> (F, G)`."""

    def __init__(self, n_var, n_obj, xl, xu, n_constr=0, name="problem"):
        self.n_var = int(n_var)
        self.n_obj = int(n_obj)
        self.n_constr = int(n_constr)
        self.xl = np.broadcast_to(np.asarray(xl, float), (self.n_var,)).copy()
        self.xu = np.broadcast_to(np.asarray(xu, float), (self.n_var,)).copy()
        self.name = name

    # -- 公共入口: 始终接受 (N, n_var), 返回 (F, G) --------------------------
    def evaluate(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        F, G = self._evaluate(X)
        F = np.asarray(F, float).reshape(len(X), self.n_obj)
        if self.n_constr > 0 and G is not None:
            G = np.asarray(G, float).reshape(len(X), self.n_constr)
        else:
            G = None
        return F, G

    def _evaluate(self, X):
        raise NotImplementedError

    def pareto_front(self, n_points=200):
        """真实 Pareto 前沿 (若已知), 用于计算 IGD+. 未知时返回 None."""
        return None

    def __repr__(self):
        return (f"Problem(name={self.name!r}, n_var={self.n_var}, "
                f"n_obj={self.n_obj}, n_constr={self.n_constr})")
