"""外部 Pareto 档案 (有界, 拥挤度截断, leader 选择).

MO-ALA / MOPSO 都用它保存当前非支配前沿, 并据此为每个个体挑选 leader.
设计要点:
    - 只保留可行 (CV<=0) 的非支配解; 全不可行时退化为保留最小违反者.
    - 截断时保留拥挤距离**大**的点 (边界 + 稀疏), 维持前沿分布.
    - leader 选择用二元锦标赛, 偏好拥挤距离大的点 (推动稀疏区探索, MOPSO 经典做法).
"""

from __future__ import annotations

import numpy as np

from .dominance import constraint_violation, crowding_distance, non_dominated_mask


class ParetoArchive:
    def __init__(self, max_size=200, dedup_decimals=8):
        self.max_size = int(max_size)
        self.dedup_decimals = dedup_decimals
        self.X = None     # (M, n_var)
        self.F = None     # (M, n_obj)

    def __len__(self):
        return 0 if self.X is None else len(self.X)

    def add(self, X, F, CV=None):
        X = np.atleast_2d(np.asarray(X, float))
        F = np.atleast_2d(np.asarray(F, float))
        if CV is not None:
            CV = np.asarray(CV, float)
            feas = CV <= 0.0
            if np.any(feas):
                X, F = X[feas], F[feas]
            else:
                return  # 本批全不可行, 不入档
        if len(X) == 0:
            return
        if self.X is None:
            self.X, self.F = X.copy(), F.copy()
        else:
            self.X = np.vstack([self.X, X])
            self.F = np.vstack([self.F, F])
        self._dedup()
        self._filter()
        self._truncate()

    def _dedup(self):
        _, idx = np.unique(
            np.round(self.F, self.dedup_decimals), axis=0, return_index=True
        )
        idx = np.sort(idx)
        self.X, self.F = self.X[idx], self.F[idx]

    def _filter(self):
        mask = non_dominated_mask(self.F)
        self.X, self.F = self.X[mask], self.F[mask]

    def _truncate(self):
        if len(self.X) <= self.max_size:
            return
        cd = crowding_distance(self.F)
        keep = np.argsort(cd)[::-1][: self.max_size]  # 拥挤距离降序, 保留稀疏/边界
        self.X, self.F = self.X[keep], self.F[keep]

    def leader(self, rng):
        """二元锦标赛选 leader, 偏好拥挤距离大者."""
        m = len(self)
        if m == 0:
            return None
        if m == 1:
            return self.X[0]
        cd = crowding_distance(self.F)
        i, j = rng.integers(0, m, size=2)
        return self.X[i] if cd[i] >= cd[j] else self.X[j]
