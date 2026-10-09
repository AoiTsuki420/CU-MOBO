"""D2 验证: 自适应可信度是否修 σ-objective 反噬(高σ最优合成基准).

在 `UncertainOptimumProblem`(真最优本身 σ 中等、安全诱饵 σ 低、陷阱 σ 极高)上比较:
  naive(max μ→陷阱) / sigma-obj(min σ→躲安全诱饵=反噬) / cu-mobo(含σ-obj→同反噬) /
  **d2-adaptive(σ 作单边自适应惩罚, 只排极高σ陷阱、留中等σ真最优)** / nsga2.
预期: d2-adaptive 触达真最优(true↑, frac_opt↑), 显著优于 sigma-obj/cu-mobo。

用法:
    python -m puridiff_optim.benchmark.run_reliability --n-var 8 --seeds 3 --budget 100 --plot
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

from ..algorithms.cu_mobo import cu_mobo
from ..problems.uncertain_optimum import UncertainOptimumProblem
from .run_oracle_gap import _topk_true, _run_nsga2

warnings.filterwarnings("ignore")

# label -> kwargs
METHODS = {
    "naive": dict(use_sigma_objective=False, use_sigma_noise=False),
    "sigma-obj": dict(use_sigma_objective=True, use_sigma_noise=False),
    "cu-mobo": dict(use_sigma_objective=True, use_sigma_noise=True),
    "d2-adaptive": dict(use_sigma_objective=False, use_sigma_noise=True, reliability_mode="adaptive"),
}


def _metrics_opt(prob, X, mu, k):
    ta = prob.true_activity(X)
    order = np.argsort(np.asarray(mu))[::-1][:k]
    return dict(topk_true=float(ta[order].mean()),
                frac_at_opt=float(np.mean(prob.at_optimum(X))),
                best_true=float(ta.max()))


def _run(label, kw, prob, budget, n_init, batch_q, seed, k):
    rec = {"n": [], "true": []}

    def mon(ev, X_all, obs):
        rec["n"].append(ev); rec["true"].append(_topk_true(prob, X_all, obs["mu"], k))

    out = cu_mobo(prob, total_budget=budget, n_init=n_init, batch_q=batch_q, seed=seed,
                  monitor=mon, **kw)
    n = np.asarray(rec["n"], float); t = np.asarray(rec["true"], float)
    auc = float(np.trapz(t, n) / (n[-1] - n[0])) if len(n) > 1 else float("nan")
    return out, auc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-var", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--n-init", type=int, default=20)
    ap.add_argument("--batch-q", type=int, default=10)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--rel-quantile", type=float, default=0.8)
    ap.add_argument("--rel-lambda", type=float, default=2.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)

    out_dir = os.path.abspath(args.out); os.makedirs(out_dir, exist_ok=True)
    prob = UncertainOptimumProblem(n_var=args.n_var)
    rows, curves = [], {m: [] for m in METHODS}
    for s in range(args.seeds):
        for label, kw in METHODS.items():
            kw = dict(kw)
            if kw.get("reliability_mode") == "adaptive":
                kw["rel_quantile"] = args.rel_quantile; kw["rel_lambda"] = args.rel_lambda
            out, auc = _run(label, kw, prob, args.budget, args.n_init, args.batch_q, s, args.topk)
            m = _metrics_opt(prob, out["X"], out["mu"], args.topk)
            rows.append({"method": label, "seed": s, "auc_true": auc, **m})
            print(f"{label:12s} seed={s} AUC={auc:.3f} topk_true={m['topk_true']:.3f} "
                  f"frac_opt={m['frac_at_opt']:.2f} best_true={m['best_true']:.3f}")
        # NSGA-II 参照(优化 [-μ, sa, -qed], 无 σ)
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem as PP
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination

        class _N(PP):
            def __init__(s_):
                super().__init__(n_var=prob.n_var, n_obj=3, xl=prob.xl, xu=prob.xu)
            def _evaluate(s_, X, out, *a, **k):
                o = prob.evaluate_oracle(X); out["F"] = np.column_stack([-o["mu"], o["sa"], -o["qed"]])
        res = minimize(_N(), NSGA2(pop_size=args.pop), get_termination("n_eval", args.budget),
                       seed=s, verbose=False)
        Xn = np.atleast_2d(res.X); on = prob.evaluate_oracle(Xn)
        m = _metrics_opt(prob, Xn, on["mu"], args.topk)
        rows.append({"method": "nsga2", "seed": s, "auc_true": float("nan"), **m})
        print(f"{'nsga2':12s} seed={s} topk_true={m['topk_true']:.3f} frac_opt={m['frac_at_opt']:.2f}")

    raw = pd.DataFrame(rows); raw.to_csv(os.path.join(out_dir, "reliability_raw.csv"), index=False)
    order = list(METHODS) + ["nsga2"]
    summary = (raw.groupby("method").agg(
        topk_true=("topk_true", "mean"), topk_true_std=("topk_true", "std"),
        frac_at_opt=("frac_at_opt", "mean"), auc_true=("auc_true", "mean"))
        .reindex(order).reset_index())
    summary.to_csv(os.path.join(out_dir, "reliability_summary.csv"), index=False)
    print(f"\n==== D2 自适应可信度 (高σ最优, n_var={args.n_var}, {args.seeds} seed) ====")
    with pd.option_context("display.width", 150, "display.max_columns", None):
        print(summary.to_string(index=False))

    if args.plot:
        _plot(summary, curves, out_dir, args.n_var)


def _plot(summary, curves, out_dir, n_var):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..core.plotting import use_cjk_font
    use_cjk_font()
    methods = summary["method"].tolist(); x = np.arange(len(methods))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.bar(x, summary["topk_true"], yerr=summary["topk_true_std"], color="darkorange")
    ax1.set_xticks(x); ax1.set_xticklabels(methods, rotation=20, ha="right")
    ax1.set_ylabel("top-K 真实活性 ↑"); ax1.set_title(f"推荐解真实活性 (n_var={n_var})")
    ax2.bar(x, summary["frac_at_opt"], color="seagreen")
    ax2.set_xticks(x); ax2.set_xticklabels(methods, rotation=20, ha="right")
    ax2.set_ylabel("触达真最优比例 ↑"); ax2.set_title("是否到真最优(避开安全诱饵/陷阱)")
    fig.suptitle("D2 自适应可信度 修 σ-objective 反噬(高σ最优基准)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "reliability.png"), dpi=130); plt.close(fig)
    print(f"图: {os.path.join(out_dir, 'reliability.png')}")


if __name__ == "__main__":
    main()
