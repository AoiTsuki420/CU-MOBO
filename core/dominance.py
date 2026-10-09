"""Pareto 支配 / 约束支配 / 非支配排序 / 拥挤度.

约束支配采用 Deb 规则:
    1. 可行解支配不可行解;
    2. 两个不可行解, 约束违反量小者支配;
    3. 两个可行解, 按标准 Pareto 支配比较.
"""

from __future__ import annotations

import numpy as np


def constraint_violation(G):
    """总约束违反量 = sum(max(0, g)). G 为 None 时返回 None."""
    if G is None:
        return None
    return np.sum(np.maximum(np.asarray(G, float), 0.0), axis=1)


def dominates(f_a, f_b, cv_a=0.0, cv_b=0.0):
    """a 是否(约束)支配 b (单点对单点)."""
    a_feas = cv_a <= 0.0
    b_feas = cv_b <= 0.0
    if a_feas and not b_feas:
        return True
    if b_feas and not a_feas:
        return False
    if not a_feas and not b_feas:
        return cv_a < cv_b
    f_a = np.asarray(f_a, float)
    f_b = np.asarray(f_b, float)
    return bool(np.all(f_a <= f_b) and np.any(f_a < f_b))


def non_dominated_mask(F):
    """向量化非支配筛选 (仅用于一组可行解). 返回布尔 mask, True = 非支配.

    复杂度 O(n^2 * n_obj), n<=几百时足够快.
    """
    F = np.asarray(F, float)
    n = len(F)
    if n == 0:
        return np.zeros(0, dtype=bool)
    # dominated_by[i, j] : F[j] 支配 F[i]
    leq = np.all(F[:, None, :] >= F[None, :, :], axis=2)   # F[j] <= F[i] 全维
    less = np.any(F[:, None, :] > F[None, :, :], axis=2)   # F[j] <  F[i] 某维
    dominated_by = leq & less
    np.fill_diagonal(dominated_by, False)
    return ~np.any(dominated_by, axis=1)


def fast_non_dominated_sort(F, CV=None):
    """NSGA-II 快速非支配排序 (约束感知). 返回若干 front, 每个是索引列表."""
    F = np.asarray(F, float)
    n = len(F)
    cv = np.zeros(n) if CV is None else np.asarray(CV, float)
    S = [[] for _ in range(n)]
    ndom = np.zeros(n, dtype=int)
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(F[p], F[q], cv[p], cv[q]):
                S[p].append(q)
            elif dominates(F[q], F[p], cv[q], cv[p]):
                ndom[p] += 1
        if ndom[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                ndom[q] -= 1
                if ndom[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
    return fronts


def crowding_distance(F):
    """NSGA-II 拥挤距离. F: (m, n_obj). 边界点为 inf."""
    F = np.asarray(F, float)
    m = len(F)
    if m == 0:
        return np.zeros(0)
    if m <= 2:
        return np.full(m, np.inf)
    n_obj = F.shape[1]
    dist = np.zeros(m)
    for k in range(n_obj):
        order = np.argsort(F[:, k])
        fk = F[order, k]
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        span = fk[-1] - fk[0]
        if span == 0:
            continue
        dist[order[1:-1]] += (fk[2:] - fk[:-2]) / span
    return dist
