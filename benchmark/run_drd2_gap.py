"""路 B: 真实 oracle (TDC DRD2) 上的 oracle-gap 验证 —— 在 drug-like VAE 隐空间.

与合成 `run_oracle_gap` 完全同口径(2×2 σ 消融 + qParEGO + NSGA-II),但 ground
truth 换成**文献标准的真实 DRD2 打分器**(Olivecrona 2017 SVM, PMO/REINVENT 用的同一
个),优化器可见的只是**有限标注预算**下训的 RF 代理(μ + 树方差 σ)。

  裁判  = 真实 DRD2 oracle(隐藏, 只在评估时调用判新分子);
  代理  = KcatSurrogate(Morgan-FP RF, sigma=None→树方差), 在 n_label 个真实标注上训;
  搜索  = drug-like SmilesVAE 隐空间 z ∈ R^16, LHS init(各法同起点, 公平).

oracle-gap 真实来源: RF 在优化器推入的新颖隐区外推过自信。CU-MOBO 用 σ(目标 + 异方差
噪声)下采信之 → 推荐分子真实 DRD2 更高、gap 更小。**无循环论证**(代理≠裁判, 判新分子)。

用法:
    python -m puridiff_optim.benchmark.run_drd2_gap --seeds 3 --budget 120 --n-label 600 --plot
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem

# 先装 rdkit.six 垫片再 import tdc
from ..molecule._tdc_compat import get_oracle
from ..molecule import KcatSurrogate, VaeDecoder
from ..problems.drd2_oracle import DRD2OracleProblem
from .run_oracle_gap import VARIANTS, BO_METHODS, _run_bo, _run_nsga2, _metrics

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(__file__)
DEFAULT_VAE = os.path.abspath(os.path.join(_HERE, "..", "molecule", "assets", "vae_drd2", "vae.pt"))
DEFAULT_CORPUS = os.path.abspath(os.path.join(_HERE, "..", "..", "tdc_data", "drd2_corpus_in.csv"))


def _build_surrogate(corpus, oracle, n_label, seed, sampling="natural", sigma_mode="calibrated"):
    """抽 n_label 个分子, 用真实 DRD2 oracle 标注, 训 RF 代理(树方差 σ).

    sampling='natural'(默认, 真实): 随机抽, 反映 library 天然活性率(actives 稀少)→
        RF 见的 actives 少、对 active-like 新分子更不确定 → 诱发真实 oracle-gap;
    sampling='balanced': 50/50 抽(代理更准, 主要用于对照, 易饱和).
    """
    rng = np.random.default_rng(seed)
    if sampling == "balanced":
        act = corpus[corpus["is_drd2_active"] == 1]["smiles"].tolist()
        ina = corpus[corpus["is_drd2_active"] == 0]["smiles"].tolist()
        rng.shuffle(act); rng.shuffle(ina)
        half = n_label // 2
        sample = act[:half] + ina[: n_label - half]
    else:  # natural: 随机抽样, 反映真实库的稀疏 active 率
        sample = corpus["smiles"].sample(n=min(n_label, len(corpus)),
                                         random_state=seed).tolist()
    rng.shuffle(sample)
    y = np.asarray(oracle(sample), float)           # 真实 DRD2 分数 [0,1] 作训练标签
    sig = "calibrated" if sigma_mode == "calibrated" else None
    sur = KcatSurrogate(seed=seed).fit(sample, y, sigma=sig)
    return sur, float(y.mean()), int((y >= 0.5).sum())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default=DEFAULT_VAE)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--n-init", type=int, default=30)
    ap.add_argument("--batch-q", type=int, default=12)
    ap.add_argument("--n-label", type=int, default=600, help="代理的真实标注预算")
    ap.add_argument("--surrogate-sampling", default="natural", choices=["natural", "balanced"],
                    help="代理标注抽样: natural=真实稀疏率(诱发gap) / balanced=50:50(易饱和)")
    ap.add_argument("--surrogate-sigma", default="calibrated", choices=["calibrated", "tree_var"],
                    help="代理σ: calibrated=误差校准(OOF残差, 满足CU-MOBO前提) / tree_var=树方差(epistemic距离)")
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "results"))
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    corpus = pd.read_csv(args.corpus)
    oracle = get_oracle("DRD2")
    dec = VaeDecoder(args.vae)
    print(f"corpus {len(corpus)} (actives {int(corpus.is_drd2_active.sum())}) | "
          f"VAE latent_dim={dec.latent_dim} | label budget={args.n_label}")

    rows, curves = [], {m: [] for m in BO_METHODS}
    rep_rows = []
    for s in range(args.seeds):
        sur, ymean, npos = _build_surrogate(corpus, oracle, args.n_label, s,
                                            args.surrogate_sampling, args.surrogate_sigma)
        prob = DRD2OracleProblem(dec, sur, oracle)
        print(f"[seed={s}] surrogate trained: label mean DRD2={ymean:.3f}, #active(>=0.5)={npos}")
        for method in BO_METHODS:
            out, rec, auc = _run_bo(method, prob, args.budget, args.n_init,
                                    args.batch_q, s, args.topk)
            m = _metrics(prob, out["X"], out["mu"], out["sigma"], args.topk)
            rows.append({"method": method, "seed": s, "auc_true": auc, **m})
            curves[method].append((np.asarray(rec["n"]), np.asarray(rec["true"])))
            print(f"  {method:12s} seed={s} AUC_true={auc:.3f} topk_true={m['topk_mu_true']:.3f} "
                  f"gap={m['front_gap_mean']:.3f}")
            if method == "cu-mobo":
                for r in prob.decode_report(out["X"]):
                    r["seed"] = s
                    rep_rows.append(r)
        m = _run_nsga2(prob, args.budget, args.pop, s, args.topk)
        rows.append({"method": "nsga2", "seed": s, "auc_true": float("nan"), **m})
        print(f"  {'nsga2':12s} seed={s} topk_true={m['topk_mu_true']:.3f} gap={m['front_gap_mean']:.3f}")

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(out_dir, "drd2_gap_raw.csv"), index=False)
    summary = (raw.groupby("method")
               .agg(auc_true=("auc_true", "mean"),
                    topk_mu_true=("topk_mu_true", "mean"),
                    topk_mu_true_std=("topk_mu_true", "std"),
                    front_gap_mean=("front_gap_mean", "mean"),
                    front_true_mean=("front_true_mean", "mean"))
               .reindex(BO_METHODS + ["nsga2"]).reset_index())
    summary.to_csv(os.path.join(out_dir, "drd2_gap_summary.csv"), index=False)

    if rep_rows:
        rep = pd.DataFrame(rep_rows).sort_values("drd2_true", ascending=False)
        rep.drop_duplicates("smiles").head(40).to_csv(
            os.path.join(out_dir, "drd2_top_molecules.csv"), index=False)

    print("\n==== 路 B: 真实 DRD2 oracle-gap (topk_true↑ 越大越好, gap↓ 越小越好) ====")
    with pd.option_context("display.width", 150, "display.max_columns", None):
        print(summary.to_string(index=False))

    if args.plot:
        _plot(summary, curves, out_dir, args.budget)
        print(f"\n图: {os.path.join(out_dir, 'drd2_gap.png')} / drd2_gap_convergence.png")


def _plot(summary, curves, out_dir, budget):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..core.plotting import use_cjk_font
    use_cjk_font()

    methods = summary["method"].tolist()
    x = np.arange(len(methods))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.bar(x, summary["topk_mu_true"], yerr=summary["topk_mu_true_std"], color="seagreen")
    ax1.set_xticks(x); ax1.set_xticklabels(methods, rotation=20, ha="right")
    ax1.set_ylabel("top-K(按代理μ) 真实 DRD2 ↑"); ax1.set_title("推荐分子的真实 DRD2 活性")
    ax2.bar(x, summary["front_gap_mean"], color="indianred")
    ax2.set_xticks(x); ax2.set_xticklabels(methods, rotation=20, ha="right")
    ax2.set_ylabel("oracle-gap |μ−DRD2_true| ↓"); ax2.set_title("前沿 oracle-gap")
    fig.suptitle("CU-MOBO 抗 oracle-gap (真实 DRD2 oracle, drug-like VAE 隐空间)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "drd2_gap.png"), dpi=130); plt.close(fig)

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
    ax.set_ylabel("top-K(按μ) 真实 DRD2 ↑")
    ax.set_title("PMO 样本效率: 真实 DRD2 vs 预算")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "drd2_gap_convergence.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
