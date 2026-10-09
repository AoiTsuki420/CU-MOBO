"""D1 验证: 信赖域(trust-region)硬化是否治高维 BO 退化.

在**高维合成不确定 oracle**(隐藏真值 a_true)上比较:
  cu-mobo(全局) vs cu-mobo-tr(信赖域) vs naive(无σ) vs naive-tr vs NSGA-II.
高维下全局 qNEHVI 易过度探索、退化到接近 NSGA-II;信赖域局部化应显著回升。

用法:
    python -m puridiff_optim.benchmark.run_trust_region --n-var 16 --width 0.5 --seeds 3 --budget 120 --plot
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

from ..algorithms.cu_mobo import cu_mobo
from ..problems.uncertain_oracle import UncertainOracleProblem
from .run_oracle_gap import _metrics, _topk_true, _run_nsga2

warnings.filterwarnings("ignore")

# (label, use_sigma_objective, use_sigma_noise, trust_region)
METHODS = [
    ("naive", False, False, False),
    ("cu-mobo", True, True, False),
    ("naive-tr", False, False, True),
    ("cu-mobo-tr", True, True, True),
]


def _run(method, prob, budget, n_init, batch_q, seed, k):
    _, uso, usn, tr = method
    rec = {"n": [], "true": []}

    def mon(ev, X_all, obs):
        rec["n"].append(ev)
        rec["true"].append(_topk_true(prob, X_all, obs["mu"], k))

    out = cu_mobo(prob, total_budget=budget, n_init=n_init, batch_q=batch_q, seed=seed,
                  monitor=mon, use_sigma_objective=uso, use_sigma_noise=usn, trust_region=tr)
    n = np.asarray(rec["n"], float); t = np.asarray(rec["true"], float)
    auc = float(np.trapz(t, n) / (n[-1] - n[0])) if len(n) > 1 else float("nan")
    return out, rec, auc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-var", type=int, default=16)
    ap.add_argument("--width", type=float, default=0.5, help="活性峰宽(高维需放大以保持可解)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--n-init", type=int, default=20)
    ap.add_argument("--batch-q", type=int, default=10)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)

    out_dir = os.path.abspath(args.out); os.makedirs(out_dir, exist_ok=True)
    prob = UncertainOracleProblem(n_var=args.n_var, width=args.width)

    rows, curves = [], {m[0]: [] for m in METHODS}
    for s in range(args.seeds):
        for method in METHODS:
            out, rec, auc = _run(method, prob, args.budget, args.n_init, args.batch_q, s, args.topk)
            m = _metrics(prob, out["X"], out["mu"], out["sigma"], args.topk)
            rows.append({"method": method[0], "seed": s, "auc_true": auc, **m})
            curves[method[0]].append((np.asarray(rec["n"]), np.asarray(rec["true"])))
            print(f"{method[0]:12s} seed={s} AUC={auc:.3f} topk_true={m['topk_mu_true']:.3f} "
                  f"gap={m['front_gap_mean']:.3f}")
        mm = _run_nsga2(prob, args.budget, args.pop, s, args.topk)
        rows.append({"method": "nsga2", "seed": s, "auc_true": float("nan"), **mm})
        print(f"{'nsga2':12s} seed={s} topk_true={mm['topk_mu_true']:.3f} gap={mm['front_gap_mean']:.3f}")

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(out_dir, "trust_region_raw.csv"), index=False)
    order = [m[0] for m in METHODS] + ["nsga2"]
    summary = (raw.groupby("method")
               .agg(auc_true=("auc_true", "mean"),
                    topk_mu_true=("topk_mu_true", "mean"),
                    topk_mu_true_std=("topk_mu_true", "std"),
                    front_gap_mean=("front_gap_mean", "mean"))
               .reindex(order).reset_index())
    summary.to_csv(os.path.join(out_dir, "trust_region_summary.csv"), index=False)
    print(f"\n==== 信赖域硬化 (n_var={args.n_var}, width={args.width}, {args.seeds} seed) ====")
    with pd.option_context("display.width", 150, "display.max_columns", None):
        print(summary.to_string(index=False))

    if args.plot:
        _plot(summary, curves, out_dir, args.budget, args.n_var)
        print(f"图: {os.path.join(out_dir, 'trust_region.png')}")


def _plot(summary, curves, out_dir, budget, n_var):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..core.plotting import use_cjk_font
    use_cjk_font()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    methods = summary["method"].tolist(); x = np.arange(len(methods))
    ax1.bar(x, summary["topk_mu_true"], yerr=summary["topk_mu_true_std"], color="teal")
    ax1.set_xticks(x); ax1.set_xticklabels(methods, rotation=20, ha="right")
    ax1.set_ylabel("top-K 真实活性 ↑"); ax1.set_title(f"推荐解真实活性 (n_var={n_var})")
    grid = np.linspace(0, budget, 50)
    for m in curves:
        cs = [(n, t) for n, t in curves[m] if len(n) > 1]
        if not cs:
            continue
        stk = np.vstack([np.interp(grid, n, t, left=t[0], right=t[-1]) for n, t in cs])
        lw = 2.4 if m == "cu-mobo-tr" else 1.3
        ax2.plot(grid, stk.mean(0), label=m, lw=lw)
    ax2.set_xlabel("真实评估次数"); ax2.set_ylabel("top-K 真实活性 ↑")
    ax2.set_title("收敛(信赖域 vs 全局)"); ax2.legend(fontsize=8)
    fig.suptitle("信赖域硬化治高维 BO 退化 (合成 oracle-gap)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "trust_region.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
