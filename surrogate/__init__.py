"""代理模型层: 每目标独立 GP, 及把代理包成可被内层搜索调用的 Problem."""

from .gp import GPSurrogate
from .surrogate_problem import SurrogateProblem

__all__ = ["GPSurrogate", "SurrogateProblem"]
