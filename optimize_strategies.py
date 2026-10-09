"""策略优化入口 —— 对接原型方案 `POST /strategies/optimize` 的算法内核.

在 IEX->HIC 合成层析问题上求 purity/yield/cost 的 Pareto 前沿, 输出:
    - Pareto 策略表 CSV (每条策略的参数 + KPI), 供前端"比较结果页/策略库"使用;
    - 三个代表性推荐 (最高纯度 / 最高收率 / 最低成本);
    - purity-yield (颜色=cost) 散点图, 叠加可行参考前沿.

支持两种优化器:
    mo_ala    —— 无代理, 适合便宜评估 (大预算)
    sa_mo_ala —— 代理辅助, 适合昂贵评估 (小预算) ★ 创新主线

用法:
    python -m puridiff_optim.optimize_strategies --method sa_mo_ala --budget 160 --plot
    python -m puridiff_optim.optimize_strategies --method mo_ala --budget 8000 --plot
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

from .algorithms.mo_ala import mo_ala
from .algorithms.sa_mo_ala import sa_mo_ala
from .problems.chromatography import ChromatographyStrategyProblem

warnings.filterwarnings("ignore")

REF_POINT = np.array([0.0, 0.0, 2.0])   # HV 参考点: [-purity, -yield, cost] 的上界


def _strategies_table(problem, archive):
    rows = []
    for x in archive.X:
        d = problem.decode(x)
        rows.append({**d["params"], **d["kpis"]})
    df = pd.DataFrame(rows)
    return df.sort_values("purity", ascending=False).reset_index(drop=True)


def _recommend(problem, archive):
    F = archive.F
    picks = {
        "MAX_PURITY": int(np.argmin(F[:, 0])),
        "MAX_YIELD": int(np.argmin(F[:, 1])),
        "MIN_COST": int(np.argmin(F[:, 2])),
    }
    out = {}
    for tag, idx in picks.items():
        out[tag] = problem.decode(archive.X[idx])
    return out


def run(method="sa_mo_ala", budget=160, seed=0, hcp_min=0.65, risk_max=0.75,
        out_dir=None, plot=False):
    prob = ChromatographyStrategyProblem(hcp_min=hcp_min, risk_max=risk_max)

    if method == "mo_ala":
        out = mo_ala(prob, n_agents=80, max_fes=budget, seed=seed)
    elif method == "sa_mo_ala":
        out = sa_mo_ala(prob, REF_POINT, total_budget=budget, n_init=40, batch_q=10,
                        inner_pop=60, inner_fes=3000, seed=seed)
    else:
        raise ValueError(method)
    archive = out["archive"]

    from pymoo.indicators.hv import HV
    hv = float(HV(ref_point=REF_POINT)(archive.F)) if len(archive) else 0.0

    table = _strategies_table(prob, archive)
    recs = _recommend(prob, archive)

    out_dir = os.path.abspath(out_dir or os.path.join(os.path.dirname(__file__), "results"))
    os.makedirs(out_dir, exist_ok=True)
    tag = f"strategies_{method}_b{budget}"
    csv_path = os.path.join(out_dir, f"{tag}.csv")
    table.to_csv(csv_path, index=False)

    print(f"=== {method}  budget={budget}  |Pareto|={len(archive)}  HV={hv:.4f} ===")
    print(f"Pareto 策略表: {csv_path}  ({len(table)} 条)")
    for tag_r, d in recs.items():
        p, k = d["params"], d["kpis"]
        print(f"\n[{tag_r}]")
        print(f"  参数: pH={p['IEX_pH']:.2f}  load={p['IEX_load_mg_mL']:.1f}mg/mL  "
              f"IEX_grad={p['IEX_grad_cv']:.1f}CV  HIC_salt={p['HIC_salt_M']:.2f}M  "
              f"HIC_grad={p['HIC_grad_cv']:.1f}CV  pool={p['pool_width']:.2f}")
        print(f"  KPI : 纯度={k['purity']:.3f}  收率={k['yield']:.3f}  成本={k['cost']:.3f}  "
              f"HCP去除={k['hcp_removal']:.3f}  风险={k['risk']:.3f}")

    if plot:
        _plot(prob, archive, recs, tag, out_dir)
        print(f"\n图: {os.path.join(out_dir, tag + '.png')}")
    return {"archive": archive, "hv": hv, "table": table, "recommendations": recs}


def _plot(problem, archive, recs, tag, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .core.plotting import use_cjk_font
    use_cjk_font()

    ref = problem.reference_front(n_samples=150000)
    F = archive.F
    purity, yld, cost = -F[:, 0], -F[:, 1], F[:, 2]

    fig, ax = plt.subplots(figsize=(7, 5))
    if ref is not None:
        ax.scatter(-ref[:, 0], -ref[:, 1], s=10, c="lightgray", label="reference feasible front")
    sc = ax.scatter(purity, yld, c=cost, cmap="viridis", s=28, label="optimized Pareto strategies")
    fig.colorbar(sc, ax=ax, label="cost (time+buffer, 越低越好)")
    for tag_r, d in recs.items():
        k = d["kpis"]
        ax.scatter([k["purity"]], [k["yield"]], s=160, marker="*", edgecolor="k",
                   facecolor="red", zorder=5)
        ax.annotate(tag_r, (k["purity"], k["yield"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("purity 纯度 (越高越好)")
    ax.set_ylabel("yield 收率 (越高越好)")
    ax.set_title(f"IEX->HIC 策略 Pareto 前沿 ({tag})")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tag}.png"), dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["mo_ala", "sa_mo_ala"], default="sa_mo_ala")
    ap.add_argument("--budget", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hcp-min", type=float, default=0.65)
    ap.add_argument("--risk-max", type=float, default=0.75)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)
    run(method=args.method, budget=args.budget, seed=args.seed,
        hcp_min=args.hcp_min, risk_max=args.risk_max, plot=args.plot)


if __name__ == "__main__":
    main()
