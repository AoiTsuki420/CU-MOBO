"""PyTDC 在现代 rdkit / Windows 上的兼容垫片.

PyTDC(我们用 `pip install PyTDC --no-deps` 装的 1.1.15)的 oracle 模块顶部有
`from rdkit.six import iteritems`,而 `rdkit.six` 在新版 rdkit(2024+)已删除,
直接 import 会抛 ImportError。本模块在 import tdc 之前把一个最小 `rdkit.six`
垫片注入 sys.modules,从而无需降级 rdkit(降级会破坏 molgen/VAE 主线)。

用法:
    from puridiff_optim.molecule._tdc_compat import get_oracle
    drd2 = get_oracle("DRD2")
    scores = drd2(["c1ccccc1", ...])   # list[float], 0~1 活性概率
"""

from __future__ import annotations

import sys
import types


def _install_rdkit_six_shim():
    if "rdkit.six" in sys.modules:
        return
    import rdkit  # noqa: F401  (确保父包存在)

    six_mod = types.ModuleType("rdkit.six")
    six_mod.iteritems = lambda d: iter(d.items())
    six_mod.iterkeys = lambda d: iter(d.keys())
    six_mod.itervalues = lambda d: iter(d.values())
    six_mod.string_types = (str,)
    six_mod.PY3 = True

    moves = types.ModuleType("rdkit.six.moves")
    import pickle as _pickle
    moves.cPickle = _pickle
    six_mod.moves = moves

    sys.modules["rdkit.six"] = six_mod
    sys.modules["rdkit.six.moves"] = moves
    setattr(rdkit, "six", six_mod)


def get_oracle(name: str):
    """返回 TDC Oracle(可调用: list[SMILES] -> list[float])."""
    _install_rdkit_six_shim()
    from tdc import Oracle
    return Oracle(name=name)


def get_data(name: str, **kwargs):
    """返回 TDC 数据集对象(如分子生成/单预测的 bioactivity)."""
    _install_rdkit_six_shim()
    import tdc
    return tdc, kwargs  # 占位; 具体加载在调用处按需要做
