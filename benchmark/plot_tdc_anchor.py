"""§4.6 外部锚点图(QED-only, rich vs scarce, CU-MOBO 高亮).

读 `results/tdc_anchor_qed_summary.csv`(label-rich) 与
`results/qed_scarce/tdc_anchor_qed_summary.csv`(label-scarce), 画一张两子图:
  (左) top-K(按代理μ) 真实 QED ↑   (右) 前沿 oracle-gap |μ−true| ↓
每方法两根柱(rich/scarce), CU-MOBO 高亮。输出 `results/tdc_anchor.png`。

用法:
    python -m puridiff_optim.benchmark.plot_tdc_anchor
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(__file__)
RES = os.path.abspath(os.path.join(_HERE, "..", "results"))

METHODS = ["naive", "sigma-obj", "sigma-noise", "cu-mobo", "qparego", "random", "nsga2"]
LABELS = {"naive": "naive\n(μ-greedy)", "sigma-obj": "σ-obj", "sigma-noise": "σ-noise",
          "cu-mobo": "CU-MOBO", "qparego": "qParEGO", "random": "random", "nsga2": "NSGA-II"}


def _load(path):
    df = pd.read_csv(path).set_index("method")
    return df.reindex(METHODS)


def main():
    rich = _load(os.path.join(RES, "tdc_anchor_qed_summary.csv"))
    scarce = _load(os.path.join(RES, "qed_scarce", "tdc_anchor_qed_summary.csv"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..core.plotting import use_cjk_font
    use_cjk_font()

    x = np.arange(len(METHODS))
    w = 0.38
    cu = METHODS.index("cu-mobo")

    def _colors(base):
        return [(base if m != "cu-mobo" else "#2ca02c") for m in METHODS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # ---- (left) top-K true QED ----
    ax1.bar(x - w / 2, rich["topk_mu_true"], w, yerr=rich["topk_mu_true_std"],
            capsize=2.5, color=_colors("#9ecae1"), edgecolor="#3182bd", label="label-rich (600)")
    ax1.bar(x + w / 2, scarce["topk_mu_true"], w, yerr=scarce["topk_mu_true_std"],
            capsize=2.5, color=_colors("#fdae6b"), edgecolor="#e6550d", hatch="//",
            label="label-scarce (120)")
    ax1.axvspan(cu - 0.5, cu + 0.5, color="#2ca02c", alpha=0.08)
    ax1.set_xticks(x); ax1.set_xticklabels([LABELS[m] for m in METHODS], rotation=22, ha="right", fontsize=8)
    ax1.set_ylabel("top-K(by surrogate μ) true QED ↑")
    ax1.set_ylim(0.78, 0.96)
    ax1.set_title("(a) Recommended designs' true QED")
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(axis="y", alpha=0.25)

    # ---- (right) oracle-gap ----
    ax2.bar(x - w / 2, rich["front_gap_mean"], w, color=_colors("#9ecae1"),
            edgecolor="#3182bd", label="label-rich (600)")
    ax2.bar(x + w / 2, scarce["front_gap_mean"], w, color=_colors("#fdae6b"),
            edgecolor="#e6550d", hatch="//", label="label-scarce (120)")
    ax2.axvspan(cu - 0.5, cu + 0.5, color="#2ca02c", alpha=0.08)
    ax2.set_xticks(x); ax2.set_xticklabels([LABELS[m] for m in METHODS], rotation=22, ha="right", fontsize=8)
    ax2.set_ylabel("front oracle-gap |μ − true| ↓")
    ax2.set_title("(b) Oracle-gap (lower = better calibrated)")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("External anchor on a standard TDC oracle (QED): CU-MOBO is no-regret at the lowest oracle-gap",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(RES, "tdc_anchor.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
