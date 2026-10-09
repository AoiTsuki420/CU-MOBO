"""oracle-gap + PMO 样本效率验证(合成不确定 oracle).

两条证据合一:
  (A) **抗 oracle-gap**: 用隐藏 a_true 评估, CU-MOBO 推荐分子真实活性↑、gap↓、陷阱↓;
  (B) **PMO 样本效率**: "top-K(按μ)真实活性 vs 真实评估预算"曲线 + AUC, 越早达到高真实
      活性越好(PMO 口径, 因真实评估昂贵).

对照: naive / σ-目标 / σ-噪声 / cu-mobo(2×2 消融) + **qParEGO**(SOTA MOBO 基线, 忽略σ)
      + NSGA-II(无代理参照).

用法:
    python -m puridiff_optim.benchmark.run_oracle_gap --seeds 3 --budget 80 --plot
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

from ..algorithms.cu_mobo import cu_mobo
from ..algorithms.qparego import qparego
from ..problems.uncertain_oracle import UncertainOracleProblem

warnings.filterwarnings("ignore")

VARIANTS = {
    "naive": dict(use_sigma_objective=False, use_sigma_noise=False),
    "sigma-obj": dict(use_sigma_objective=True, use_sigma_noise=False),
    "sigma-noise": dict(use_sigma_objective=False, use_sigma_noise=True),
    "cu-mobo": dict(use_sigma_objective=True, use_sigma_noise=True),
}
BO_METHODS = list(VARIANTS) + ["qparego"]


def _topk_true(prob, X_all, mu_all, k):
    order = np.argsort(np.asarray(mu_all))[::-1][:k]
    return float(prob.true_activity(X_all[order]).mean())


def _cliffs_delta(a, b):
    """Cliff's δ = P(a>b) − P(a<b) ∈ [−1,1]; |δ|: .11 小 / .28 中 / .43 大 (非参效应量)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = sum(int((x > b).sum()) for x in a)
    lt = sum(int((x < b).sum()) for x in a)
    return (gt - lt) / (len(a) * len(b))


def _mag(d):
    ad = abs(d)
    return "大" if ad >= 0.43 else "中" if ad >= 0.28 else "小" if ad >= 0.11 else "微"


def _metrics(prob, X, mu, sigma, k=5, sigma_thresh=0.15):
    ta = prob.true_activity(X)
    gap = np.abs(np.asarray(mu) - ta)
    order = np.argsort(np.asarray(mu))[::-1][:k]
    return dict(front=len(X), front_true_mean=float(ta.mean()),
                front_gap_mean=float(gap.mean()), topk_mu_true=float(ta[order].mean()),
                topk_mu_sigma=float(np.asarray(sigma)[order].mean()),
                trap_frac=float(np.mean(np.asarray(sigma) > sigma_thresh)))


def _run_bo(method, prob, budget, n_init, batch_q, seed, k):
    rec = {"n": [], "true": []}

    def mon(ev, X_all, obs):
        rec["n"].append(ev)
        rec["true"].append(_topk_true(prob, X_all, obs["mu"], k))

    if method == "qparego":
        out = qparego(prob, total_budget=budget, n_init=n_init, batch_q=batch_q,
                      seed=seed, monitor=mon)
    else:
        out = cu_mobo(prob, total_budget=budget, n_init=n_init, batch_q=batch_q,
                      seed=seed, monitor=mon, **VARIANTS[method])
    n = np.asarray(rec["n"], float)
    t = np.asarray(rec["true"], float)
    auc = float(np.trapz(t, n) / (n[-1] - n[0])) if len(n) > 1 else float("nan")
    return out, rec, auc


def _run_nsga2(prob, budget, pop, seed, k):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem as PymooProb
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    class _Naive(PymooProb):
        def __init__(self):
            super().__init__(n_var=prob.n_var, n_obj=3, xl=prob.xl, xu=prob.xu)

        def _evaluate(self, X, out, *a, **k):
            o = prob.evaluate_oracle(X)
            out["F"] = np.column_stack([-o["mu"], o["sa"], -o["qed"]])

    res = minimize(_Naive(), NSGA2(pop_size=pop), get_termination("n_eval", budget),
                   seed=seed, verbose=False)
    X = np.atleast_2d(res.X)
    o = prob.evaluate_oracle(X)
    return _metrics(prob, X, o["mu"], o["sigma"], k)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget", type=int, default=80)
    ap.add_argument("--n-init", type=int, default=20)
    ap.add_argument("--batch-q", type=int, default=10)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--n-var", type=int, default=6)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    prob = UncertainOracleProblem(n_var=args.n_var)

    rows = []
    curves = {m: [] for m in BO_METHODS}
    for s in range(args.seeds):
        for method in BO_METHODS:
            out, rec, auc = _run_bo(method, prob, args.budget, args.n_init,
                                    args.batch_q, s, args.topk)
            m = _metrics(prob, out["X"], out["mu"], out["sigma"], args.topk)
            rows.append({"method": method, "seed": s, "auc_true": auc, **m})
            curves[method].append((np.asarray(rec["n"]), np.asarray(rec["true"])))
            print(f"{method:12s} seed={s} AUC_true={auc:.3f} topk_true={m['topk_mu_true']:.3f} "
                  f"gap={m['front_gap_mean']:.3f} trap={m['trap_frac']:.2f}")
        m = _run_nsga2(prob, args.budget, args.pop, s, args.topk)
        rows.append({"method": "nsga2", "seed": s, "auc_true": float("nan"), **m})
        print(f"{'nsga2':12s} seed={s} topk_true={m['topk_mu_true']:.3f} "
              f"gap={m['front_gap_mean']:.3f} trap={m['trap_frac']:.2f}")
        # 每 seed 增量存档(此 runner 慢, 中途停可保留已完成 seeds)
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "oracle_gap_raw.csv"), index=False)
        print(f"  [checkpoint: {s + 1}/{args.seeds} seeds saved]")

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(out_dir, "oracle_gap_raw.csv"), index=False)
    summary = (raw.groupby("method")
               .agg(auc_true=("auc_true", "mean"),
                    topk_mu_true=("topk_mu_true", "mean"),
                    topk_mu_true_std=("topk_mu_true", "std"),
                    front_gap_mean=("front_gap_mean", "mean"),
                    trap_frac=("trap_frac", "mean"))
               .reindex(BO_METHODS + ["nsga2"]).reset_index())
    summary.to_csv(os.path.join(out_dir, "oracle_gap_summary.csv"), index=False)

    print("\n==== oracle-gap + PMO 样本效率 (AUC/true↑ 越大越好, gap/trap↓ 越小越好) ====")
    with pd.option_context("display.width", 150, "display.max_columns", None):
        print(summary.to_string(index=False))

    # 非参效应量(Cliff's δ): CU-MOBO 相对各基线, topk_true(越大越好) 与 gap(越小越好)
    eff = []
    cu_t = raw[raw.method == "cu-mobo"]["topk_mu_true"].values
    cu_g = raw[raw.method == "cu-mobo"]["front_gap_mean"].values
    for base in ["naive", "sigma-obj", "sigma-noise", "qparego", "nsga2"]:
        bt = raw[raw.method == base]["topk_mu_true"].values
        bg = raw[raw.method == base]["front_gap_mean"].values
        dt = _cliffs_delta(cu_t, bt)        # >0: cu-mobo true 更高
        dg = _cliffs_delta(cu_g, bg)        # <0: cu-mobo gap 更低(更好)
        eff.append({"vs": base, "cliffs_delta_true": dt, "mag_true": _mag(dt),
                    "cliffs_delta_gap": dg, "mag_gap": _mag(dg)})
    edf = pd.DataFrame(eff); edf.to_csv(os.path.join(out_dir, "oracle_gap_effects.csv"), index=False)
    print(f"\n---- 效应量 Cliff's δ (CU-MOBO vs 基线, {args.seeds} seeds) ----")
    for e in eff:
        print(f"  vs {e['vs']:11s}: true δ={e['cliffs_delta_true']:+.2f}({e['mag_true']}) "
              f"| gap δ={e['cliffs_delta_gap']:+.2f}({e['mag_gap']}, 负=CU gap 更低更好)")

    if args.plot:
        _plot(summary, curves, out_dir, args.budget)
        print(f"\n图: {os.path.join(out_dir, 'oracle_gap.png')} 和 oracle_gap_convergence.png")


def _plot(summary, curves, out_dir, budget):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..core.plotting import use_cjk_font
    use_cjk_font()

    methods = summary["method"].tolist()
    x = np.arange(len(methods))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.bar(x, summary["topk_mu_true"], yerr=summary["topk_mu_true_std"], color="steelblue")
    ax1.set_xticks(x); ax1.set_xticklabels(methods, rotation=20, ha="right")
    ax1.set_ylabel("top-K(按μ) 真实活性 ↑"); ax1.set_title("推荐分子的真实活性")
    ax2.bar(x, summary["front_gap_mean"], color="indianred")
    ax2.set_xticks(x); ax2.set_xticklabels(methods, rotation=20, ha="right")
    ax2.set_ylabel("oracle-gap |μ−a_true| ↓"); ax2.set_title("前沿 oracle-gap")
    fig.suptitle("CU-MOBO 抗 oracle-gap (合成不确定 oracle)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "oracle_gap.png"), dpi=130); plt.close(fig)

    # PMO 样本效率收敛曲线
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    grid = np.linspace(0, budget, 50)
    for m in curves:
        cs = [(n, t) for n, t in curves[m] if len(n) > 1]
        if not cs:
            continue
        stacked = np.vstack([np.interp(grid, n, t, left=t[0], right=t[-1]) for n, t in cs])
        mean, std = stacked.mean(0), stacked.std(0)
        lw = 2.2 if m == "cu-mobo" else 1.3
        ax.plot(grid, mean, label=m, lw=lw)
        ax.fill_between(grid, mean - std, mean + std, alpha=0.12)
    ax.set_xlabel("真实评估次数 (oracle budget)")
    ax.set_ylabel("top-K(按μ) 真实活性 ↑")
    ax.set_title("PMO 样本效率: 真实活性 vs 预算")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "oracle_gap_convergence.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
