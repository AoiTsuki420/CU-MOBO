"""统一 benchmark runner.

对照 3 个算法:
    mo_ala  —— 本项目创新点 (多目标 ALA)
    mopso   —— 群智能基线
    nsga2   —— pymoo, 经典 GA 强基线

在同一组问题对象上、同一 FE 预算下运行多 seed, 用 pymoo 的 HV / IGD+
统一评估, 输出原始表 + 汇总表 (+ 可选收敛曲线/前沿图).

用法:
    python -m puridiff_optim.benchmark.run_benchmark \
        --problems zdt1 zdt2 zdt3 dtlz2 --algos mo_ala mopso nsga2 \
        --seeds 5 --max-fes 10000 --pop 100 --plot
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
from ..algorithms.mo_ala_plus import mo_ala_plus
from ..algorithms.mopso import mopso
from ..problems.benchmark import get_benchmark, suite_names


# ---------------------------------------------------------------- 收敛记录器 --
class HVRecorder:
    """给 mo_ala / mopso 用的 monitor 回调: 记录 (n_eval, HV)."""

    def __init__(self, ref_point):
        self.ref = np.asarray(ref_point, float)
        self.n_eval = []
        self.hv = []
        self._ind = HV(ref_point=self.ref)

    def __call__(self, fes, archive):
        if len(archive) == 0:
            return
        self.n_eval.append(int(fes))
        self.hv.append(float(self._ind(archive.F)))


# ----------------------------------------------------------------- NSGA-II ----
def run_nsga2(prob, ref_point, max_fes, pop, seed):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.callback import Callback
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    ind_hv = HV(ref_point=np.asarray(ref_point, float))

    class _CB(Callback):
        def __init__(self):
            super().__init__()
            self.n_eval = []
            self.hv = []

        def notify(self, algorithm):
            F = algorithm.opt.get("F")
            if F is not None and len(F):
                self.n_eval.append(int(algorithm.evaluator.n_eval))
                self.hv.append(float(ind_hv(F)))

    cb = _CB()
    res = minimize(
        prob._p, NSGA2(pop_size=pop), get_termination("n_eval", max_fes),
        seed=seed, callback=cb, verbose=False,
    )
    return res.F, cb.n_eval, cb.hv


# ----------------------------------------------------------------- 单次运行 ----
def run_one(algo, prob, ref_point, max_fes, pop, seed):
    t0 = time.perf_counter()
    rec = HVRecorder(ref_point)
    if algo == "mo_ala":
        out = mo_ala(prob, n_agents=pop, max_fes=max_fes, seed=seed, monitor=rec)
        F, ne, hv = out["F"], rec.n_eval, rec.hv
    elif algo == "mo_ala_plus":
        out = mo_ala_plus(prob, n_agents=pop, max_fes=max_fes, seed=seed,
                          monitor=rec, archive_kind="epsilon")
        F, ne, hv = out["F"], rec.n_eval, rec.hv
    elif algo == "mopso":
        out = mopso(prob, n_agents=pop, max_fes=max_fes, seed=seed, monitor=rec)
        F, ne, hv = out["F"], rec.n_eval, rec.hv
    elif algo == "nsga2":
        F, ne, hv = run_nsga2(prob, ref_point, max_fes, pop, seed)
    else:
        raise ValueError(f"unknown algo {algo!r}")
    dt = time.perf_counter() - t0

    pf = prob.pareto_front()
    final_hv = float(HV(ref_point=np.asarray(ref_point, float))(F)) if F is not None and len(F) else 0.0
    final_igd = float(IGDPlus(pf)(F)) if (pf is not None and F is not None and len(F)) else float("nan")
    n_front = 0 if F is None else len(F)
    return {
        "hv": final_hv, "igd_plus": final_igd, "n_front": n_front,
        "time_s": dt, "curve_neval": ne, "curve_hv": hv, "F": F,
    }


# ----------------------------------------------------------------- 主流程 ------
def main(argv=None):
    ap = argparse.ArgumentParser(description="puridiff_optim 多目标 benchmark")
    ap.add_argument("--problems", nargs="+", default=suite_names())
    ap.add_argument("--algos", nargs="+", default=["mo_ala", "mopso", "nsga2"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-fes", type=int, default=10000)
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--tag", default="", help="输出文件名后缀, 避免覆盖")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    curves = {}  # (problem, algo) -> list of (neval, hv) per seed
    fronts = {}  # (problem, algo) -> F of seed 0
    for pname in args.problems:
        prob, ref = get_benchmark(pname)
        for algo in args.algos:
            for s in range(args.seeds):
                r = run_one(algo, prob, ref, args.max_fes, args.pop, seed=s)
                rows.append({
                    "problem": pname, "algo": algo, "seed": s,
                    "hv": r["hv"], "igd_plus": r["igd_plus"],
                    "n_front": r["n_front"], "time_s": r["time_s"],
                })
                curves.setdefault((pname, algo), []).append((r["curve_neval"], r["curve_hv"]))
                if s == 0:
                    fronts[(pname, algo)] = r["F"]
                print(f"[{pname:6s}] {algo:6s} seed={s} "
                      f"HV={r['hv']:.5f} IGD+={r['igd_plus']:.5f} "
                      f"|front|={r['n_front']} t={r['time_s']:.1f}s")

    raw = pd.DataFrame(rows)
    raw_path = os.path.join(out_dir, f"benchmark_raw{args.tag}.csv")
    raw.to_csv(raw_path, index=False)

    summary = (raw.groupby(["problem", "algo"])
               .agg(hv_mean=("hv", "mean"), hv_std=("hv", "std"),
                    igd_mean=("igd_plus", "mean"), igd_std=("igd_plus", "std"),
                    time_mean=("time_s", "mean"))
               .reset_index())
    summary_path = os.path.join(out_dir, f"benchmark_summary{args.tag}.csv")
    summary.to_csv(summary_path, index=False)

    print("\n==== SUMMARY (HV 越大越好, IGD+ 越小越好) ====")
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(summary.to_string(index=False))
    print(f"\n原始结果: {raw_path}\n汇总结果: {summary_path}")

    if args.plot:
        _make_plots(args.problems, args.algos, curves, fronts, out_dir, args.tag)
        print(f"图表已写入: {out_dir}")


def _make_plots(problems, algos, curves, fronts, out_dir, tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for pname in problems:
        # 收敛曲线 (HV vs FEs, seed 平均, 插值到公共网格)
        fig, ax = plt.subplots(figsize=(6, 4))
        for algo in algos:
            seeds_curves = curves.get((pname, algo), [])
            seeds_curves = [(np.asarray(n), np.asarray(h))
                            for n, h in seeds_curves if len(n) > 1]
            if not seeds_curves:
                continue
            lo = max(c[0][0] for c in seeds_curves)
            hi = min(c[0][-1] for c in seeds_curves)
            grid = np.linspace(lo, hi, 50)
            stacked = np.vstack([np.interp(grid, n, h) for n, h in seeds_curves])
            mean, std = stacked.mean(0), stacked.std(0)
            ax.plot(grid, mean, label=algo)
            ax.fill_between(grid, mean - std, mean + std, alpha=0.15)
        ax.set_xlabel("function evaluations")
        ax.set_ylabel("hypervolume")
        ax.set_title(f"{pname}: HV convergence")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"convergence_{pname}{tag}.png"), dpi=130)
        plt.close(fig)

        # 最终前沿 (仅 2 目标问题)
        sample = fronts.get((pname, algos[0]))
        if sample is not None and sample.shape[1] == 2:
            fig, ax = plt.subplots(figsize=(6, 4))
            prob, _ = get_benchmark(pname)
            pf = prob.pareto_front()
            if pf is not None:
                ax.plot(pf[:, 0], pf[:, 1], "k-", lw=1, alpha=0.5, label="true PF")
            for algo in algos:
                F = fronts.get((pname, algo))
                if F is not None and len(F):
                    ax.scatter(F[:, 0], F[:, 1], s=12, label=algo)
            ax.set_xlabel("f1")
            ax.set_ylabel("f2")
            ax.set_title(f"{pname}: final Pareto front (seed 0)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"front_{pname}{tag}.png"), dpi=130)
            plt.close(fig)


if __name__ == "__main__":
    main()
