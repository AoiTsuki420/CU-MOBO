from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE.parent
PACKAGE_DIR = DOCS_DIR.parent
PROJECT_DIR = PACKAGE_DIR.parent
RESULTS_DIR = PACKAGE_DIR / "results"
OUT_DIR = HERE / "english"
DPI = 130

sys.path.insert(0, str(PROJECT_DIR))


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
        }
    )
    return plt


def _plot_oracle_gap_bars(out_dir: Path) -> None:
    plt = _setup_matplotlib()
    summary = pd.read_csv(RESULTS_DIR / "oracle_gap_summary.csv")
    methods = summary["method"].tolist()
    x = np.arange(len(methods))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.bar(
        x,
        summary["topk_mu_true"],
        yerr=summary["topk_mu_true_std"],
        color="steelblue",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=20, ha="right")
    ax1.set_ylabel("top-K true activity by mu")
    ax1.set_title("Recommended molecules' true activity")

    ax2.bar(x, summary["front_gap_mean"], color="indianred")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=20, ha="right")
    ax2.set_ylabel("oracle-gap |mu - a_true|")
    ax2.set_title("Pareto-front oracle-gap")

    fig.suptitle("CU-MOBO reduces oracle-gap on a synthetic uncertain oracle")
    fig.tight_layout()
    fig.savefig(out_dir / "oracle_gap.png")
    plt.close(fig)


def _plot_oracle_gap_convergence(out_dir: Path, seeds: int, budget: int) -> None:
    from puridiff_optim.benchmark.run_oracle_gap import BO_METHODS, _run_bo
    from puridiff_optim.problems.uncertain_oracle import UncertainOracleProblem

    plt = _setup_matplotlib()
    prob = UncertainOracleProblem(n_var=6)
    curves = {m: [] for m in BO_METHODS}
    for seed in range(seeds):
        for method in BO_METHODS:
            _, rec, _ = _run_bo(
                method,
                prob,
                budget=budget,
                n_init=20,
                batch_q=10,
                seed=seed,
                k=5,
            )
            curves[method].append((np.asarray(rec["n"]), np.asarray(rec["true"])))

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    grid = np.linspace(0, budget, 50)
    for method in BO_METHODS:
        samples = [(n, t) for n, t in curves[method] if len(n) > 1]
        if not samples:
            continue
        stacked = np.vstack(
            [np.interp(grid, n, t, left=t[0], right=t[-1]) for n, t in samples]
        )
        mean, std = stacked.mean(0), stacked.std(0)
        lw = 2.2 if method == "cu-mobo" else 1.3
        ax.plot(grid, mean, label=method, lw=lw)
        ax.fill_between(grid, mean - std, mean + std, alpha=0.12)

    ax.set_xlabel("True oracle evaluations")
    ax.set_ylabel("top-K true activity by mu")
    ax.set_title("PMO sample efficiency: true activity vs budget")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "oracle_gap_convergence.png")
    plt.close(fig)


def _plot_retrospective(out_dir: Path) -> None:
    plt = _setup_matplotlib()
    raw = pd.read_csv(RESULTS_DIR / "retrospective_hardened_raw.csv")
    splits = ["random", "scaffold"]
    x = np.arange(len(splits))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, metric, title in [
        (ax1, "AUC", "ROC-AUC"),
        (ax2, "BEDROC", "BEDROC(alpha=80.5)"),
    ]:
        sur = [raw[raw.split == sp][f"sur_{metric}"].mean() for sp in splits]
        nn = [raw[raw.split == sp][f"nn_{metric}"].mean() for sp in splits]
        sur_e = [raw[raw.split == sp][f"sur_{metric}"].std() for sp in splits]
        nn_e = [raw[raw.split == sp][f"nn_{metric}"].std() for sp in splits]
        ax.bar(x - width / 2, sur, width, yerr=sur_e, label="kcat surrogate", color="steelblue")
        ax.bar(x + width / 2, nn, width, yerr=nn_e, label="1-NN memory baseline", color="gray")
        ax.axhline(0.5 if metric == "AUC" else 0.0, color="k", ls=":", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Strict surrogate screening for high-kcat substrates "
        "(random vs scaffold split; vs 1-NN memory)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "retrospective_hardened.png")
    plt.close(fig)


def _calibration_curves(seed: int = 0, test_frac: float = 0.25, max_n: int = 6000):
    from puridiff_optim.benchmark.run_calibration import (
        DEFAULT_DATA,
        _load_unique_kcat,
        _metrics,
    )
    from puridiff_optim.molecule import KcatSurrogate

    rng = np.random.RandomState(seed)
    data = _load_unique_kcat(DEFAULT_DATA)
    if len(data) > max_n:
        data = data.iloc[rng.permutation(len(data))[:max_n]].reset_index(drop=True)
    idx = rng.permutation(len(data))
    n_test = int(len(data) * test_frac)
    te, tr = idx[:n_test], idx[n_test:]
    s_tr = data["canon"].values[tr]
    y_tr = data["logkcat"].values[tr]
    s_te = data["canon"].values[te]
    y_te = data["logkcat"].values[te]

    rows = {}
    curves = {}
    for tag, sigma_mode in [("tree-var", None), ("OOF-calibrated", "calibrated")]:
        surrogate = KcatSurrogate(seed=seed)
        surrogate.rf_mu.n_jobs = 1
        surrogate.rf_sigma.n_jobs = 1
        surrogate.fit(s_tr, y_tr, sigma=sigma_mode)
        mu, sigma = surrogate.predict(s_te)
        ok = ~np.isnan(mu) & ~np.isnan(sigma)
        metrics = _metrics(y_te[ok], mu[ok], np.maximum(sigma[ok], 1e-6))
        curves[tag] = metrics.pop("_curve")
        rows[tag] = metrics
    return curves, rows


def _plot_calibration(out_dir: Path) -> None:
    plt = _setup_matplotlib()
    curves, rows = _calibration_curves()
    colors = {"tree-var": "tab:gray", "OOF-calibrated": "tab:orange"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for tag, curve in curves.items():
        ax1.plot(
            curve["ps"],
            curve["obs"],
            color=colors[tag],
            lw=2,
            label=f"{tag} (miscal={rows[tag]['miscal_area']:.3f})",
        )
    ax1.set_xlabel("Nominal coverage p")
    ax1.set_ylabel("Observed coverage")
    ax1.set_title("Reliability diagram (sigma^2 as train_Yvar)")
    ax1.legend(fontsize=8)

    lim = max(
        max(curve["rmv"].max(), curve["rmse"].max()) for curve in curves.values()
    )
    ax2.plot([0, lim], [0, lim], "k--", lw=1, label="perfect")
    for tag, curve in curves.items():
        ax2.plot(
            curve["rmv"],
            curve["rmse"],
            "o-",
            color=colors[tag],
            lw=2,
            label=f"{tag} (ENCE={rows[tag]['ence']:.3f}, rho={rows[tag]['spearman']:.2f})",
        )
    ax2.set_xlabel("Predicted RMV = sqrt(mean sigma^2) per bin")
    ax2.set_ylabel("Observed RMSE per bin")
    ax2.set_title("Error calibration curve (sigma objective)")
    ax2.legend(fontsize=8)

    fig.suptitle(
        "CU-MOBO sigma calibration: OOF-calibrated sigma vs RF tree-variance sigma "
        "(held-out kcat)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "calibration.png")
    plt.close(fig)


def _plot_tdc_anchor(out_dir: Path) -> None:
    plt = _setup_matplotlib()
    methods = ["naive", "sigma-obj", "sigma-noise", "cu-mobo", "qparego", "random", "nsga2"]
    labels = {
        "naive": "naive\n(mu-greedy)",
        "sigma-obj": "sigma-obj",
        "sigma-noise": "sigma-noise",
        "cu-mobo": "CU-MOBO",
        "qparego": "qParEGO",
        "random": "random",
        "nsga2": "NSGA-II",
    }
    rich = pd.read_csv(RESULTS_DIR / "tdc_anchor_qed_summary.csv").set_index("method").reindex(methods)
    scarce = (
        pd.read_csv(RESULTS_DIR / "qed_scarce" / "tdc_anchor_qed_summary.csv")
        .set_index("method")
        .reindex(methods)
    )
    x = np.arange(len(methods))
    width = 0.38
    cu_index = methods.index("cu-mobo")

    def colors(base: str):
        return [base if method != "cu-mobo" else "#2ca02c" for method in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax1.bar(
        x - width / 2,
        rich["topk_mu_true"],
        width,
        yerr=rich["topk_mu_true_std"],
        capsize=2.5,
        color=colors("#9ecae1"),
        edgecolor="#3182bd",
        label="label-rich (600)",
    )
    ax1.bar(
        x + width / 2,
        scarce["topk_mu_true"],
        width,
        yerr=scarce["topk_mu_true_std"],
        capsize=2.5,
        color=colors("#fdae6b"),
        edgecolor="#e6550d",
        hatch="//",
        label="label-scarce (120)",
    )
    ax1.axvspan(cu_index - 0.5, cu_index + 0.5, color="#2ca02c", alpha=0.08)
    ax1.set_xticks(x)
    ax1.set_xticklabels([labels[method] for method in methods], rotation=22, ha="right", fontsize=8)
    ax1.set_ylabel("top-K(by surrogate mu) true QED")
    ax1.set_ylim(0.78, 0.96)
    ax1.set_title("(a) Recommended designs' true QED")
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(axis="y", alpha=0.25)

    ax2.bar(
        x - width / 2,
        rich["front_gap_mean"],
        width,
        color=colors("#9ecae1"),
        edgecolor="#3182bd",
        label="label-rich (600)",
    )
    ax2.bar(
        x + width / 2,
        scarce["front_gap_mean"],
        width,
        color=colors("#fdae6b"),
        edgecolor="#e6550d",
        hatch="//",
        label="label-scarce (120)",
    )
    ax2.axvspan(cu_index - 0.5, cu_index + 0.5, color="#2ca02c", alpha=0.08)
    ax2.set_xticks(x)
    ax2.set_xticklabels([labels[method] for method in methods], rotation=22, ha="right", fontsize=8)
    ax2.set_ylabel("front oracle-gap |mu - true|")
    ax2.set_title("(b) Oracle-gap (lower is better calibrated)")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "External anchor on a standard TDC oracle (QED): "
        "CU-MOBO is no-regret at the lowest oracle-gap",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "tdc_anchor.png")
    plt.close(fig)


def _plot_diversity(out_dir: Path) -> None:
    plt = _setup_matplotlib()
    raw = pd.read_csv(RESULTS_DIR / "diversity_raw.csv")
    order = ["cu-mobo", "naive", "random"]
    panels = [
        ("mean_mu", "Predicted kcat mu\n(higher is better)"),
        ("IntDiv1", "IntDiv1\n(within-set diversity)"),
        ("n_scaffold", "Unique scaffolds\n(higher is better)"),
        ("frac_novel", "Novel fraction NN<0.4\n(higher is more novel)"),
    ]
    colors = {"cu-mobo": "tab:orange", "naive": "tab:gray", "random": "lightsteelblue"}

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (column, title) in zip(axes, panels):
        mean = [raw[raw.method == method][column].mean() for method in order]
        err = [raw[raw.method == method][column].std() for method in order]
        ax.bar(order, mean, yerr=err, color=[colors[method] for method in order])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle(
        "Recommended Pareto molecule sets: CU-MOBO preserves diversity while "
        "favoring more trusted near-training regions (kcat latent space)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "diversity.png")
    plt.close(fig)


def _plot_overoptimization(out_dir: Path) -> None:
    plt = _setup_matplotlib()
    data = pd.read_csv(RESULTS_DIR / "overoptimization.csv")
    labels = ["0\n(flow only)", "0.3", "0.6", "inf\n(grad-only\nno flow)"]
    x = np.arange(len(labels))
    true = data["true_drd2"].to_numpy()
    head = data["head_belief"].to_numpy()
    validity = data["validity"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(x, head, "o--", color="tab:blue", lw=2, label="proxy head self-score")
    ax.plot(x, true, "o-", color="tab:orange", lw=2.5, label="true DRD2 activity")
    ax.plot(x, validity, "s:", color="tab:green", lw=1.5, alpha=0.7, label="decode validity")
    ax.fill_between(x, true, head, color="red", alpha=0.10, label="reward-hacking gap")

    peak = int(np.argmax(true))
    ax.annotate(
        "true peak",
        (x[peak], true[peak]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=9,
        color="tab:orange",
    )
    ax.annotate(
        "overoptimization collapse\n(head=1.0, true=0)",
        (x[-1], true[-1]),
        textcoords="offset points",
        xytext=(-8, 28),
        ha="center",
        fontsize=9,
        color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=1),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Optimization pressure (guide_scale)")
    ax.set_ylabel("Mean value in [0, 1]")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(
        "Overoptimization: true activity collapses at extreme pressure\n"
        "(COMs/Design-Bench style, independent DRD2 oracle)"
    )
    ax.legend(fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(out_dir / "overoptimization.png")
    plt.close(fig)


def main() -> None:
    global OUT_DIR, DPI

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--skip-expensive", action="store_true")
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    OUT_DIR = args.out
    DPI = args.dpi
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_oracle_gap_bars(OUT_DIR)
    if not args.skip_expensive:
        _plot_oracle_gap_convergence(OUT_DIR, seeds=args.seeds, budget=args.budget)
        _plot_calibration(OUT_DIR)
    _plot_retrospective(OUT_DIR)
    _plot_tdc_anchor(OUT_DIR)
    _plot_diversity(OUT_DIR)
    _plot_overoptimization(OUT_DIR)
    print(f"Wrote English figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
