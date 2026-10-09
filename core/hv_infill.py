"""基于超体积增益的批量 infill 选点 (贪婪, 乐观估计).

给定当前真实档案前沿 archive_F 与一批候选 (代理 Pareto 集) 的预测 mean/std,
按"乐观超体积增益" (用 mean - kappa*std 作为候选目标值) 贪婪挑 q 个点,
每选一个就把它并入工作前沿再算下一个, 避免扎堆.
"""

from __future__ import annotations

import numpy as np
from pymoo.indicators.hv import HV


def greedy_hvi_batch(archive_F, cand_mean, cand_std, ref_point, q, kappa=1.0):
    """返回所选候选的索引列表 (长度 <= q)."""
    ref = np.asarray(ref_point, float)
    ind = HV(ref_point=ref)
    val = np.asarray(cand_mean, float) - kappa * np.asarray(cand_std, float)  # 乐观值
    n = len(val)
    if n == 0:
        return []

    working = []
    if archive_F is not None and len(archive_F):
        working = [np.asarray(f, float) for f in archive_F]

    selected = []
    remaining = list(range(n))
    for _ in range(min(q, n)):
        base = ind(np.array(working)) if working else 0.0
        best_gain, best_i = -np.inf, None
        for i in remaining:
            test = np.array(working + [val[i]])
            gain = ind(test) - base
            if gain > best_gain:
                best_gain, best_i = gain, i
        if best_i is None:
            break
        selected.append(best_i)
        working.append(val[best_i])
        remaining.remove(best_i)
    return selected
