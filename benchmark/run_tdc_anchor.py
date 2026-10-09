"""外部锚点: 标准 TDC oracle 上的 oracle-gap / 样本效率验证(PMO 口径).

把 `run_drd2_gap` 推广到**任意 TDC oracle**, 作论文 §4.6 的**外部锚点**(external
anchor): 在文献标准、社区公认的打分器上复测 CU-MOBO, 检验 §5 的前提条件(代理须有
真实 oracle-gap 且 σ 须校准)在外部 oracle 上是否成立。

裁判  = 真实 TDC oracle(隐藏, 仅评估新分子时调用);
代理  = KcatSurrogate(Morgan-FP RF, σ="calibrated"=OOF 残差校准), 有限标注预算上训;
       —— 这关键地**同时**测了 CU-MOBO 的两半(σ-目标 + 异方差校准噪声), 非仅 proxy-σ;
搜索  = drug-like SmilesVAE 隐空间 z, 各法 LHS 同起点(公平)。

对照: naive(μ-greedy BO) / σ-obj / σ-noise / cu-mobo(2×2 消融) / qParEGO(SOTA MOBO)
      / **random**(均匀隐空间采样, PMO 下界) / NSGA-II(无代理 GP 参照)。

无循环论证: 代理 ≠ 裁判, 裁判判的是优化器推出的**新颖**分子。oracle-gap 真实来源 =
RF 在新颖隐区外推过自信。

可用 oracle(本机实测):
  QED   —— 解析(RDKit), 平滑/易, σ 校准良好但 oracle-gap 小 → 预期 CU-MOBO 无悔/中性;
  DRD2  —— Olivecrona SVM, 稀有-active/难, σ 信息弱 → 预期无代理-BO 法占优(已知, §5 前提不满足);
  (GSK3B/JNK3 的 pickled RF oracle 因 sklearn 版本不兼容无法反序列化, 故略。)

用法:
    python -m puridiff_optim.benchmark.run_tdc_anchor --oracle QED --seeds 3 --budget 120 --plot
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

from ..molecule._tdc_compat import get_oracle
from ..molecule import VaeDecoder
from ..problems.drd2_oracle import DRD2OracleProblem
from .run_oracle_gap import VARIANTS, BO_METHODS, _run_bo, _run_nsga2, _metrics, _cliffs_delta, _mag
from .run_drd2_gap import _build_surrogate

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(__file__)
DEFAULT_VAE = os.path.abspath(os.path.join(_HERE, "..", "molecule", "assets", "vae_drd2", "vae.pt"))
DEFAULT_CORPUS = os.path.abspath(os.path.join(_HERE, "..", "..", "tdc_data", "drd2_corpus_in.csv"))


def _run_random(prob, budget, seed, k):
    """PMO 下界: 均匀隐空间采样 budget 个点, 按代理 μ 取 top-k, 报真实活性 + AUC."""
    rng = np.random.default_rng(seed)
    Z = rng.uniform(prob.xl, prob.xu, size=(budget, prob.n_var))
    o = prob.evaluate_oracle(Z)
    mu = np.asarray(o["mu"], float)
    # PMO 样本效率曲线: 前 n 个采样里 top-k(按μ) 的真实活性
    rec_n, rec_t = [], []
    for n in np.linspace(max(k, budget // 10), budget, 10, dtype=int):
        order = np.argsort(mu[:n])[::-1][:k]
        rec_t.append(float(prob.true_activity(Z[:n][order]).mean()))
        rec_n.append(int(n))
    n = np.asarray(rec_n, float); t = np.asarray(rec_t, float)
    auc = float(np.trapz(t, n) / (n[-1] - n[0])) if len(n) > 1 else float("nan")
    m = _metrics(prob, Z, o["mu"], o["sigma"], k)
    return m, (np.asarray(rec_n), np.asarray(rec_t)), auc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="QED", help="TDC oracle 名(QED/DRD2/...)")
    ap.add_argument("--vae", default=DEFAULT_VAE)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--n-init", type=int, default=30)
    ap.add_argument("--batch-q", type=int, default=12)
    ap.add_argument("--n-label", type=int, default=600, help="代理的真实标注预算")
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "results"))
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    tag = args.oracle.lower()
    corpus = pd.read_csv(args.corpus)
    oracle = get_oracle(args.oracle)
    dec = VaeDecoder(args.vae)
    print(f"[{args.oracle}] corpus {len(corpus)} | VAE latent_dim={dec.latent_dim} | "
          f"label budget={args.n_label} | budget={args.budget} seeds={args.seeds}")

    methods_all = BO_METHODS + ["random", "nsga2"]
    rows, curves = [], {m: [] for m in BO_METHODS + ["random"]}
    for s in range(args.seeds):
        # natural 采样 + calibrated σ(OOF 残差): 同时测 σ-目标 与 校准异方差噪声两半
        sur, ymean, npos = _build_surrogate(corpus, oracle, args.n_label, s,
                                            "natural", "calibrated")
        prob = DRD2OracleProblem(dec, sur, oracle)
        print(f"[seed={s}] surrogate: label mean={ymean:.3f}, #(>=0.5)={npos}")
        for method in BO_METHODS:
            out, rec, auc = _run_bo(method, prob, args.budget, args.n_init,
                                    args.batch_q, s, args.topk)
            m = _metrics(prob, out["X"], out["mu"], out["sigma"], args.topk)
            rows.append({"method": method, "seed": s, "auc_true": auc, **m})
            curves[method].append((np.asarray(rec["n"]), np.asarray(rec["true"])))
            print(f"  {method:12s} seed={s} PMO-AUC={auc:.3f} topk_true={m['topk_mu_true']:.3f} "
                  f"gap={m['front_gap_mean']:.3f}")
        mr, recr, aucr = _run_random(prob, args.budget, s, args.topk)
        rows.append({"method": "random", "seed": s, "auc_true": aucr, **mr})
        curves["random"].append(recr)
        print(f"  {'random':12s} seed={s} PMO-AUC={aucr:.3f} topk_true={mr['topk_mu_true']:.3f} "
              f"gap={mr['front_gap_mean']:.3f}")
        mn = _run_nsga2(prob, args.budget, args.pop, s, args.topk)
        rows.append({"method": "nsga2", "seed": s, "auc_true": float("nan"), **mn})
        print(f"  {'nsga2':12s} seed={s} topk_true={mn['topk_mu_true']:.3f} gap={mn['front_gap_mean']:.3f}")
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"tdc_anchor_{tag}_raw.csv"), index=False)
        print(f"  [checkpoint: {s + 1}/{args.seeds} seeds saved]")

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(out_dir, f"tdc_anchor_{tag}_raw.csv"), index=False)
    summary = (raw.groupby("method")
               .agg(auc_true=("auc_true", "mean"),
                    topk_mu_true=("topk_mu_true", "mean"),
                    topk_mu_true_std=("topk_mu_true", "std"),
                    front_gap_mean=("front_gap_mean", "mean"),
                    front_true_mean=("front_true_mean", "mean"))
               .reindex(methods_all).reset_index())
    summary.to_csv(os.path.join(out_dir, f"tdc_anchor_{tag}_summary.csv"), index=False)

    print(f"\n==== 外部锚点 [{args.oracle}] (PMO-AUC/topk_true↑ 越大越好, gap↓ 越小越好) ====")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(summary.to_string(index=False))

    # 效应量: CU-MOBO vs naive/random(topk_true 越大越好, gap 越小越好)
    eff = []
    cu_t = raw[raw.method == "cu-mobo"]["topk_mu_true"].values
    cu_g = raw[raw.method == "cu-mobo"]["front_gap_mean"].values
    for base in ["naive", "sigma-obj", "sigma-noise", "qparego", "random", "nsga2"]:
        bt = raw[raw.method == base]["topk_mu_true"].values
        bg = raw[raw.method == base]["front_gap_mean"].values
        dt = _cliffs_delta(cu_t, bt); dg = _cliffs_delta(cu_g, bg)
        eff.append({"vs": base, "cliffs_delta_true": dt, "mag_true": _mag(dt),
                    "cliffs_delta_gap": dg, "mag_gap": _mag(dg)})
    edf = pd.DataFrame(eff); edf.to_csv(os.path.join(out_dir, f"tdc_anchor_{tag}_effects.csv"), index=False)
    print(f"\n---- 效应量 Cliff's δ (CU-MOBO vs 基线, {args.seeds} seeds) ----")
    for e in eff:
        print(f"  vs {e['vs']:11s}: true δ={e['cliffs_delta_true']:+.2f}({e['mag_true']}) "
              f"| gap δ={e['cliffs_delta_gap']:+.2f}({e['mag_gap']}, 负=CU gap 更低更好)")

    if args.plot:
        _plot(summary, curves, out_dir, args.budget, args.oracle, tag)
        print(f"\n图: tdc_anchor_{tag}.png / tdc_anchor_{tag}_convergence.png")


def _plot(summary, curves, out_dir, budget, oracle_name, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..core.plotting import use_cjk_font
    use_cjk_font()

    methods = summary["method"].tolist()
    x = np.arange(len(methods))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.bar(x, summary["topk_mu_true"], yerr=summary["topk_mu_true_std"], color="teal")
    ax1.set_xticks(x); ax1.set_xticklabels(methods, rotation=20, ha="right")
    ax1.set_ylabel(f"top-K(按代理μ) 真实 {oracle_name} ↑"); ax1.set_title(f"推荐分子的真实 {oracle_name}")
    ax2.bar(x, summary["front_gap_mean"], color="indianred")
    ax2.set_xticks(x); ax2.set_xticklabels(methods, rotation=20, ha="right")
    ax2.set_ylabel("oracle-gap |μ−true| ↓"); ax2.set_title("前沿 oracle-gap")
    fig.suptitle(f"外部锚点: CU-MOBO 在标准 TDC {oracle_name} oracle (drug-like VAE 隐空间)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"tdc_anchor_{tag}.png"), dpi=130); plt.close(fig)

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
        ax.fill_between(grid, mean - std, mean + std, alpha=0.10)
    ax.set_xlabel("真实评估次数 (oracle budget)")
    ax.set_ylabel(f"top-K(按μ) 真实 {oracle_name} ↑")
    ax.set_title(f"PMO 样本效率: 真实 {oracle_name} vs 预算")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"tdc_anchor_{tag}_convergence.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
