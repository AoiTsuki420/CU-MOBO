"""初始试验设计采样 (拉丁超立方)."""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc


def latin_hypercube(n, xl, xu, seed=None):
    """在 [xl, xu] 盒约束内做 n 点拉丁超立方采样, 返回 (n, dim)."""
    xl = np.asarray(xl, float)
    xu = np.asarray(xu, float)
    dim = len(xl)
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    unit = sampler.random(n)          # [0,1]^dim
    return xl + unit * (xu - xl)
