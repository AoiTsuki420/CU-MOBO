"""通用搜索算子: Lévy 飞行, 边界处理.

Lévy 飞行严格复现原始 ALA 的 MATLAB 实现 (beta=1.5, Mantegna 算法).
"""

from __future__ import annotations

import math

import numpy as np


def levy_flight(dim, rng, beta=1.5):
    """Mantegna 算法生成 Lévy 步长 (与 ALA.m 的 Levy(d) 一致)."""
    sigma = (
        math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = rng.standard_normal(dim) * sigma
    v = rng.standard_normal(dim)
    return u / np.abs(v) ** (1 / beta)


def clip_to_bounds(X, xl, xu):
    """越界裁剪到 [xl, xu] (对应 ALA.m 的 Boundary correction)."""
    return np.clip(X, xl, xu)
