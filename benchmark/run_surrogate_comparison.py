"""样本效率对比 (M2): 昂贵评估小预算下, 代理辅助 vs 无代理.

方法:
    plain_mo_ala —— 无代理 (同样真实评估预算直接搜)
    sa_mo_ala    —— ★ 代理辅助 MO-ALA (创新主线)
    mobo         —— qNEHVI (方向2兜底)
    nsga2        —— 无代理 GA 参照

横轴 = 真实评估次数; 纵轴 = HV. 重点看"达到给定 HV 所需真实评估数".

用法:
    python -m puridiff_optim.benchmark.run_surrogate_comparison \
        --problem zdt1 --n-var 6 --budget 120 --n-init 20 --batch-q 10 --seeds 5 --plot
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus

from ..algorithms.mo_ala import mo_ala
from ..algorithms.mobo import mobo
from ..algorithms.sa_mo_ala import sa_mo_ala
from ..problems.benchmark import PymooProblem
from .run_benchmark import HVRecorder, run_nsga2

METHODS = ["plain_mo_ala", "sa_mo_ala", "mobo", "nsga2"]


def _run_method(mth, prob, ref, budget, n_init, batch_q, pop, seed):
    rec = HVRecorder(ref)
    t0 = time.perf_counter()
    if mth == "plain_mo_ala":
        out = mo_ala(prob, n_agents=pop, max_fes=budget, seed=seed, monitor=rec)
        F, ne, hv = out["F"], rec.n_eval, rec.hv
    elif mth == "sa_mo_ala":
        out = sa_mo_ala(prob, ref, total_budget=budget, n_init=n_init,
                        batch_q=batch_q, seed=seed, monitor=rec)
        F, ne, hv = out["F"], rec.n_eval, rec.hv
    elif mth == "mobo":
        out = mobo(prob, ref, total_budget=budget, n_init=n_init,
                   batch_q=batch_q, seed=seed, monitor=rec)
        F, ne, hv = out["F"], rec.n_eval, rec.hv
    elif mth == "nsga2":
        F, ne, hv = run_nsga2(prob, ref, budget, pop=pop, seed=seed)
    else:
        raise ValueError(mth)
    return F, ne, hv, time.perf_counter() - t0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="zdt1")
    ap.add_argument("--n-var", type=int, default=6)
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--n-init", type=int, default=20)
    ap.add_argument("--batch-q", type=int, default=10)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--ref", nargs="+", type=float, default=[1.1, 1.1])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    prob = PymooProblem(args.problem, n_var=args.n_var)
    ref = np.asarray(args.ref, float)
    pf = prob.pareto_front()
    ind_hv = HV(ref_point=ref)
    ind_igd = IGDPlus(pf)

    rows = []
    curves = {m: [] for m in args.methods}
    for s in range(args.seeds):
        for mth in args.methods:
            F, ne, hv, dt = _run_method(
                mth, prob, ref, args.budget, args.n_init, args.batch_q, args.pop, s)
            fhv = float(ind_hv(F)) if F is not None and len(F) else 0.0
            figd = float(ind_igd(F)) if (pf is not None and F is not None and len(F)) else float("nan")
            rows.append({"problem": f"{args.problem}_{args.n_var}d", "method": mth,
                         "seed": s, "hv": fhv, "igd_plus": figd,
                         "n_front": 0 if F is None else len(F), "time_s": dt})
            curves[mth].append((np.asarray(ne), np.asarray(hv)))
            print(f"{mth:13s} seed={s} HV={fhv:.4f} IGD+={figd:.4f} "
                  f"|front|={0 if F is None else len(F):3d} t={dt:.1f}s")

    raw = pd.DataFrame(rows)
    tag = f"{args.problem}_{args.n_var}d_b{args.budget}"
    raw.to_csv(os.path.join(out_dir, f"surrogate_cmp_{tag}_raw.csv"), index=False)
    summary = (raw.groupby("method")
               .agg(hv_mean=("hv", "mean"), hv_std=("hv", "std"),
                    igd_mean=("igd_plus", "mean"), igd_std=("igd_plus", "std"),
                    time_mean=("time_s", "mean"))
               .reindex(args.methods).reset_index())
    summary.to_csv(os.path.join(out_dir, f"surrogate_cmp_{tag}_summary.csv"), index=False)

    print(f"\n==== {tag}: 样本效率对比 (HV 越大越好) ====")
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(summary.to_string(index=False))

    if args.plot:
        _plot(args.methods, curves, tag, out_dir, args.budget)
        print(f"图: {os.path.join(out_dir, f'surrogate_cmp_{tag}.png')}")


def _plot(methods, curves, tag, out_dir, budget):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    grid = np.linspace(0, budget, 60)
    for mth in methods:
        seeds_curves = [(n, h) for n, h in curves[mth] if len(n) > 1]
        if not seeds_curves:
            continue
        # 阶梯插值到公共真实评估网格 (HV 单调不降, 用前向填充式插值)
        stacked = []
        for n, h in seeds_curves:
            stacked.append(np.interp(grid, n, h, left=0.0, right=h[-1]))
        stacked = np.vstack(stacked)
        mean, std = stacked.mean(0), stacked.std(0)
        ax.plot(grid, mean, label=mth, lw=2 if mth == "sa_mo_ala" else 1.3)
        ax.fill_between(grid, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel("real evaluations")
    ax.set_ylabel("hypervolume")
    ax.set_title(f"{tag}: sample efficiency (surrogate vs no-surrogate)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"surrogate_cmp_{tag}.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
