"""分子逆向设计入口 —— 在 SmilesVAE 隐空间上跑多目标优化.

把 puridiff_optim 的优化器与 molgen_baseline 的 SMILES 生成模型耦合:
    latent z --VAE--> SMILES --(kcat 代理 + RDKit)--> (−kcat μ, σ, SA, −QED) + 约束
    --> MO-ALA+ (从真实分子隐编码 warm-start) --> 分子 Pareto 前沿

产出: Pareto 分子表 CSV(SMILES + kcat μ/σ + QED/SA/...) + 代表分子 + 活性-不确定性图.

⚠️ 本地 demo: kcat 用本地代理(KcatMoE 的 proxy)。真实闭环见 molgen PAPER_PLAN
   §3(批次送真实 KcatMoE + 独立 oracle 验证 true-hit)。

用法:
    python -m puridiff_optim.optimize_molecules --max-fes 4000 --n-agents 80 --plot
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

from .algorithms.cu_mobo import cu_mobo
from .algorithms.mo_ala_plus import mo_ala_plus
from .molecule import KcatSurrogate, MoleculeLatentProblem, VaeDecoder

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(__file__)
_ASSETS = os.path.join(_HERE, "molecule", "assets")
DEFAULT_VAE = os.path.join(_ASSETS, "vae.pt")
DEFAULT_WARMSTART = os.path.join(_ASSETS, "train_corpus.csv")
# 优先用真实 KinEAGER 400 分子(pred_kcat_log/unc_kcat_log); 回退到 molgen 的 ranked_candidates
DEFAULT_KCAT_CSV = r"E:\fullproject\project\LKM-SMA-NN\predictions_kcat.csv"


def _load_surrogate(csv_path, seed=0):
    from .mode_b.active_design import _MU_ALIASES, _SIG_ALIASES, _SMILES_ALIASES, _find_col
    df = pd.read_csv(csv_path)
    smi = _find_col(df, _SMILES_ALIASES)
    mu = _find_col(df, _MU_ALIASES)
    sig = _find_col(df, _SIG_ALIASES)
    if smi is None or mu is None:
        raise ValueError(f"kcat 表需含 smiles 和 μ 列; 见 {list(df.columns)}")
    df = df.dropna(subset=[smi, mu]).drop_duplicates(smi)
    sigma = df[sig] if sig else pd.Series([0.1] * len(df))
    ks = KcatSurrogate(seed=seed).fit(df[smi], df[mu], sigma)
    return ks, len(df)


def _archetypes(table):
    if len(table) == 0:
        return {}
    picks = {
        "MAX_KCAT": table["kcat_mu_log10"].idxmax(),
        "MIN_SIGMA": table["kcat_sigma"].idxmin(),
        "MIN_SA": table["sa"].idxmin(),
        "MAX_QED": table["qed"].idxmax(),
    }
    return {tag: table.loc[idx].to_dict() for tag, idx in picks.items()}


def run(vae_path=DEFAULT_VAE, kcat_csv=DEFAULT_KCAT_CSV, warmstart_csv=DEFAULT_WARMSTART,
        method="cu_mobo", n_agents=80, max_fes=4000, budget=160, n_init=60, batch_q=12,
        seed=0, z_bound=3.0, out_dir=None, plot=False):
    dec = VaeDecoder(vae_path)
    kcat, n_sur = _load_surrogate(kcat_csv, seed=seed)
    print(f"VAE latent_dim={dec.latent_dim} | kcat surrogate trained on {n_sur} molecules "
          f"| method={method}")

    prob = MoleculeLatentProblem(dec, kcat, z_bound=z_bound)

    # warm-start: 编码真实(分布内)分子作为初始种群/设计
    init_X = None
    if warmstart_csv and os.path.exists(warmstart_csv):
        ws = pd.read_csv(warmstart_csv)["smiles"].dropna().astype(str)
        m = max(n_agents, n_init)
        ws = ws.sample(min(len(ws), m), random_state=seed).tolist()
        init_X = dec.encode(ws)

    if method == "cu_mobo":
        out = cu_mobo(prob, total_budget=budget, n_init=n_init, batch_q=batch_q,
                      seed=seed, init_X=init_X)
        X_front = out["X"]
    elif method == "mo_ala_plus":
        out = mo_ala_plus(prob, n_agents=n_agents, max_fes=max_fes, seed=seed,
                          archive_kind="crowding", init_X=init_X)
        X_front = out["archive"].X
    else:
        raise ValueError(method)

    rows = prob.report(X_front)
    table = pd.DataFrame(rows).drop_duplicates("smiles").reset_index(drop=True)
    table = table.sort_values("kcat_mu_log10", ascending=False).reset_index(drop=True)

    out_dir = os.path.abspath(out_dir or os.path.join(_HERE, "results"))
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"molecules_pareto_{method}.csv")
    table.to_csv(csv_path, index=False)

    archs = _archetypes(table)
    print(f"\n=== 分子 Pareto ({method}): unique valid molecules={len(table)} ===")
    print(f"Pareto 分子表: {csv_path}")
    print(f"kcat_μ(log10) 范围 {table['kcat_mu_log10'].min():.2f}..{table['kcat_mu_log10'].max():.2f}  |  "
          f"σ 范围 {table['kcat_sigma'].min():.3f}..{table['kcat_sigma'].max():.3f}  |  "
          f"QED 范围 {table['qed'].min():.2f}..{table['qed'].max():.2f}")
    for tag, r in archs.items():
        print(f"\n[{tag}] {r['smiles']}")
        print(f"   kcat_μ={r['kcat_mu_log10']:.2f} σ={r['kcat_sigma']:.3f} "
              f"QED={r['qed']:.2f} SA={r['sa']:.2f} MW={r['mw']:.0f} "
              f"Lipinski={r['lipinski_pass']} PAINS={r['pains']}")

    if plot:
        _plot(table, archs, out_dir, method)
        print(f"\n图: {os.path.join(out_dir, f'molecules_pareto_{method}.png')}")
    return {"table": table, "archetypes": archs, "result": out}


def _plot(table, archs, out_dir, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .core.plotting import use_cjk_font
    use_cjk_font()

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(table["kcat_mu_log10"], table["kcat_sigma"], c=table["qed"],
                    cmap="viridis", s=30, edgecolor="none")
    fig.colorbar(sc, ax=ax, label="QED 成药性 (越高越好)")
    for atag, r in archs.items():
        ax.scatter([r["kcat_mu_log10"]], [r["kcat_sigma"]], s=170, marker="*",
                   facecolor="red", edgecolor="k", zorder=5)
        ax.annotate(atag, (r["kcat_mu_log10"], r["kcat_sigma"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("predicted kcat μ (log10, 越高越好)")
    ax.set_ylabel("predicted σ 不确定性 (越低越可信)")
    ax.set_title(f"隐空间多目标分子逆向设计 ({tag})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"molecules_pareto_{tag}.png"), dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["cu_mobo", "mo_ala_plus"], default="cu_mobo")
    ap.add_argument("--vae", default=DEFAULT_VAE)
    ap.add_argument("--kcat-csv", default=DEFAULT_KCAT_CSV)
    ap.add_argument("--warmstart-csv", default=DEFAULT_WARMSTART)
    ap.add_argument("--n-agents", type=int, default=80)
    ap.add_argument("--max-fes", type=int, default=4000)
    ap.add_argument("--budget", type=int, default=160, help="cu_mobo 真实评估预算")
    ap.add_argument("--n-init", type=int, default=60)
    ap.add_argument("--batch-q", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--z-bound", type=float, default=3.0)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)
    run(vae_path=args.vae, kcat_csv=args.kcat_csv, warmstart_csv=args.warmstart_csv,
        method=args.method, n_agents=args.n_agents, max_fes=args.max_fes,
        budget=args.budget, n_init=args.n_init, batch_q=args.batch_q,
        seed=args.seed, z_bound=args.z_bound, plot=args.plot)


if __name__ == "__main__":
    main()
