"""ε-支配外部档案.

用 ε-网格保持前沿均匀分布: 每个被占据的 box 只保留一个"最靠近理想角"的代表点.
相比纯拥挤距离截断, ε 网格在 3+ 目标上分布更均匀, 也能避免档案退化.
自适应调节网格分辨率, 使保留点数接近 target_size. 接口同 ParetoArchive.
"""

from __future__ import annotations

import numpy as np

from .dominance import crowding_distance, non_dominated_mask


def _box_representatives(F, ideal, span, div):
    """给定网格划分 div, 返回每个占据 box 的代表点索引 (最靠近理想角者)."""
    coord = (F - ideal) / span * div
    box = np.minimum(np.floor(coord).astype(np.int64), div - 1)
    frac = coord - box
    dist = np.sqrt((frac ** 2).sum(axis=1))
    rep = {}
    for i in range(len(F)):
        key = tuple(box[i])
        if key not in rep or dist[i] < rep[key][1]:
            rep[key] = (i, dist[i])
    return [v[0] for v in rep.values()]


class EpsilonArchive:
    def __init__(self, target_size=100, dedup_decimals=8):
        self.target_size = int(target_size)
        self.dedup_decimals = dedup_decimals
        self.X = None
        self.F = None

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
                return
        if len(X) == 0:
            return
        if self.X is None:
            self.X, self.F = X.copy(), F.copy()
        else:
            self.X = np.vstack([self.X, X])
            self.F = np.vstack([self.F, F])
        self._dedup()
        mask = non_dominated_mask(self.F)
        self.X, self.F = self.X[mask], self.F[mask]
        self._eps_thin()

    def _dedup(self):
        _, idx = np.unique(np.round(self.F, self.dedup_decimals), axis=0, return_index=True)
        idx = np.sort(idx)
        self.X, self.F = self.X[idx], self.F[idx]

    def _eps_thin(self):
        m = len(self.F)
        if m <= self.target_size:
            return
        ideal = self.F.min(0)
        span = np.maximum(self.F.max(0) - ideal, 1e-12)
        # 网格数 div 越大保留越多; 二分找最大的 div 使保留数 <= target_size
        lo, hi = 2, max(4, 8 * self.target_size)
        best = _box_representatives(self.F, ideal, span, lo)
        while lo <= hi:
            mid = (lo + hi) // 2
            keep = _box_representatives(self.F, ideal, span, mid)
            if len(keep) <= self.target_size:
                best = keep
                lo = mid + 1
            else:
                hi = mid - 1
        idx = np.array(sorted(best), dtype=int)
        self.X, self.F = self.X[idx], self.F[idx]

    def leader(self, rng):
        m = len(self)
        if m == 0:
            return None
        if m == 1:
            return self.X[0]
        cd = crowding_distance(self.F)
        i, j = rng.integers(0, m, size=2)
        return self.X[i] if cd[i] >= cd[j] else self.X[j]
