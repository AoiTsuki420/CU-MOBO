"""桥接 molgen_baseline 的代码 (SmilesVAE / Vocab / RDKit 工具).

通过把 molgen_baseline/src 加入 sys.path 复用其已有实现, 不复制代码.
路径可用环境变量 MOLGEN_SRC 覆盖.
"""

from __future__ import annotations

import os
import sys

_DEFAULT_MOLGEN_SRC = r"E:\fullproject\笔记\boltz_run\molgen_baseline\src"


def molgen_src_path():
    return os.environ.get("MOLGEN_SRC", _DEFAULT_MOLGEN_SRC)


_p = molgen_src_path()
if _p not in sys.path:
    sys.path.insert(0, _p)

# 复用 molgen_baseline 的实现
from data_utils import Vocab  # noqa: E402
from chem_utils import canonicalize_smiles, compute_props  # noqa: E402
from latent_flow_matching import SmilesVAE  # noqa: E402

__all__ = ["Vocab", "SmilesVAE", "canonicalize_smiles", "compute_props", "molgen_src_path"]
