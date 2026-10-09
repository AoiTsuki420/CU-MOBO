"""matplotlib 中文字体设置 (Windows). 让图里的中文正常显示."""

from __future__ import annotations


def use_cjk_font():
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Microsoft JhengHei", "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
