"""标准多目标 benchmark 适配 (ZDT / DTLZ), 通过 pymoo 提供真实 Pareto 前沿.

把 pymoo 问题包成本包统一的 `Problem` 接口, 这样 MO-ALA / MOPSO 和 pymoo
的 NSGA-II 跑在**同一个**问题对象上, HV / IGD+ 口径完全一致.
"""

from __future__ import annotations

import numpy as np
from pymoo.problems import get_problem

from ..core.problem import Problem


class PymooProblem(Problem):
    """把 pymoo 的多目标测试问题适配为本包 Problem."""

    def __init__(self, name, **kwargs):
        p = get_problem(name, **kwargs)
        super().__init__(
            n_var=p.n_var, n_obj=p.n_obj, xl=p.xl, xu=p.xu,
            n_constr=0, name=name,
        )
        self._p = p

    def _evaluate(self, X):
        out = self._p.evaluate(X, return_values_of=["F"])
        return np.asarray(out, float), None

    def pareto_front(self, n_points=200):
        try:
            return self._p.pareto_front()
        except Exception:
            return None


# 默认 benchmark 套件: ZDT 系列 (2目标) + DTLZ2 (3目标), 配各自 HV 参考点.
_DEFAULT_SUITE = {
    "zdt1": {"ref_point": [1.1, 1.1]},
    "zdt2": {"ref_point": [1.1, 1.1]},
    "zdt3": {"ref_point": [1.1, 1.1]},
    "dtlz2": {"kwargs": {"n_var": 12, "n_obj": 3}, "ref_point": [1.1, 1.1, 1.1]},
}


def get_benchmark(name):
    """返回 (problem, ref_point). name 见 _DEFAULT_SUITE."""
    if name not in _DEFAULT_SUITE:
        raise KeyError(f"unknown benchmark {name!r}; choices={list(_DEFAULT_SUITE)}")
    spec = _DEFAULT_SUITE[name]
    prob = PymooProblem(name, **spec.get("kwargs", {}))
    return prob, np.asarray(spec["ref_point"], float)


def suite_names():
    return list(_DEFAULT_SUITE)
