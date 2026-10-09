"""puridiff_optim — 重组小分子蛋白纯化策略多目标优化器内核.

研究原型: "方向 2 兜底 (MOBO/qEHVI) + 方向 1 创新 (代理辅助多目标 ALA)".

本包是**独立**模块, 不依赖任何私有平台仓库 (backend/frontend) 或 CADET_DPNN
可微主线; 真实评估通过可插拔 `Problem` / `Simulator` 接口注入.

子模块:
    core       —— Problem 抽象, Pareto 支配/拥挤度, 外部档案, 通用算子
    algorithms —— ALA(单目标), MO-ALA(多目标创新点), MOPSO(基线)
    problems   —— ZDT/DTLZ 等标准多目标 benchmark 适配
    benchmark  —— 统一 runner: 多算法 x 多问题 x 多 seed, 计算 HV/IGD+
"""

from .core.problem import Problem
from .core.archive import ParetoArchive
from .core.epsilon_archive import EpsilonArchive
from .algorithms.ala import ala
from .algorithms.mo_ala import mo_ala
from .algorithms.mo_ala_plus import mo_ala_plus
from .algorithms.mopso import mopso

__all__ = ["Problem", "ParetoArchive", "EpsilonArchive",
           "ala", "mo_ala", "mo_ala_plus", "mopso"]
__version__ = "0.1.0"
