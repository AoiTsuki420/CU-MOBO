"""COMs/Design-Bench 式"过度优化曲线"(reward-hacking 的标准证明图)。

把 D4 的 guide_scale 扫描重画成文献标准的"真值 vs 优化压力": 随引导(优化压力)增强,
**真实活性先升后崩**, 而**代理头自评单调爬到 1.0** —— 二者的背离 = reward-hacking gap,
在去掉流先验(grad-only, 压力=∞)时最大(头信 1.0、真实 0)。复刻 COMs(Trabucco ICML2021)
Fig3「优化越狠真值越掉」的招牌可视化, 用我们自己的独立 DRD2 oracle 数据。

数据取自 `results/flow_guided_d4_s03.csv`(guide=0.3)+ `_s06.csv`(guide=0.6),
二者共享 unguided(guide=0)与 grad-only(∞)行。

用法:
    python -m puridiff_optim.benchmark.plot_overoptimization
参考: COMs arXiv:2107.06882(Fig3); Design-Bench arXiv:2202.08450; DyRAMO 10.1038/s41467-025-57582-3。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def main():
    res = os.path.join(os.path.dirname(__file__), "..", "results")
    res = os.path.abspath(res)
    s03 = pd.read_csv(os.path.join(res, "flow_guided_d4_s03.csv")).set_index("method")
    s06 = pd.read_csv(os.path.join(res, "flow_guided_d4_s06.csv")).set_index("method")

    # 优化压力轴: guide_scale 0 -> 0.3 -> 0.6 -> ∞(grad-only, 去流先验)
    rows = [
        ("0\n(纯流)", s03.loc["unguided"]),
        ("0.3", s03.loc["guided"]),
        ("0.6", s06.loc["guided"]),
        ("∞\n(grad-only\n去流)", s06.loc["grad-only"]),
    ]
    labels = [r[0] for r in rows]
    true = np.array([r[1]["mean_drd2"] for r in rows])
    head = np.array([r[1]["head_prob"] for r in rows])
    valid = np.array([r[1]["validity"] for r in rows])
    gap = head - true

    out_csv = os.path.join(res, "overoptimization.csv")
    pd.DataFrame({"pressure": ["0", "0.3", "0.6", "inf_gradonly"],
                  "true_drd2": true, "head_belief": head, "validity": valid,
                  "reward_hack_gap": gap}).to_csv(out_csv, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..core.plotting import use_cjk_font
    use_cjk_font()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(x, head, "o--", color="tab:blue", lw=2, label="代理头自评 (proxy 相信的)")
    ax.plot(x, true, "o-", color="tab:orange", lw=2.5, label="真实 DRD2 活性 (独立 oracle)")
    ax.plot(x, valid, "s:", color="tab:green", lw=1.5, alpha=0.7, label="解码有效率 (在流形上)")
    ax.fill_between(x, true, head, color="red", alpha=0.10, label="reward-hacking gap")
    # 峰值标注
    ipk = int(np.argmax(true))
    ax.annotate("真值峰", (x[ipk], true[ipk]), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9, color="tab:orange")
    ax.annotate("过度优化崩塌\n(头信1.0 真实0)", (x[-1], true[-1]), textcoords="offset points",
                xytext=(-8, 28), ha="center", fontsize=9, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=1))
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("优化压力 (引导强度 guide_scale →)")
    ax.set_ylabel("值 (均, [0,1])")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("过度优化曲线: 引导抬真实活性, 去生成先验则 reward-hacking 崩塌\n"
                 "(COMs/Design-Bench 式, 独立 DRD2 oracle)")
    ax.legend(fontsize=8, loc="center left")
    fig.tight_layout()
    png = os.path.join(res, "overoptimization.png")
    fig.savefig(png, dpi=130); plt.close(fig)
    print("真实活性(随压力):", np.round(true, 3).tolist())
    print("头自评(随压力):  ", np.round(head, 3).tolist())
    print("reward-hack gap: ", np.round(gap, 3).tolist())
    print(f"saved {out_csv}\nsaved {png}")


if __name__ == "__main__":
    main()
