"""路 A: 真实 kcat 标签的回顾性"底物空间富集"验证(留出版, 不靠预测器当裁判).

裁判 = **真实实验 kcat 标签**. 流程:
  1. 51k 真实数据 -> 每个唯一规范 SMILES 取 max log10(kcat);
  2. 留出 HIGH(高 kcat top 分位 抽 n_test)与 LOW(低分位)测试集, surrogate 训练**不含**它们(控泄漏);
  3. 在 train 上训真实-标签 kcat surrogate(σ 取 RF 树间方差);
  4. CU-MOBO 在 VAE 隐空间优化 surrogate -> 生成分子;随机 VAE 采样作基线;
  5. 指标(Tanimoto/Morgan):recovery@τ = 留出 HIGH 中"有生成分子相似度≥τ"的比例;
     enrichment = recovery_cu / recovery_random; 特异性 = recover HIGH vs LOW.

诚实 caveat: VAE 语料可能与留出分子重叠(VAE 能随机生成它们), 但 surrogate 留出严格,
且 enrichment 对比的是"CU 引导 vs 随机", 隔离的是**引导/泛化**效应; 真实活性终需湿验证.

用法:
    python -m puridiff_optim.benchmark.run_retrospective_kcat --budget 120 --n-test 60 --tau 0.4 --plot
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity

from ..algorithms.cu_mobo import cu_mobo
from ..molecule import KcatSurrogate, MoleculeLatentProblem, VaeDecoder

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(__file__)
DEFAULT_VAE = os.path.abspath(os.path.join(_HERE, "..", "molecule", "assets", "vae16", "vae.pt"))
DEFAULT_DATA = r"E:\fullproject\project\gptsrc\data\unified_train_mixed_phys.csv"
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _fp(s):
    m = Chem.MolFromSmiles(str(s))
    return _GEN.GetFingerprint(m) if m else None


def _max_sim_to_ref(gen_smiles, ref_smiles):
    """每个 ref 分子对 gen 集合的最大 Tanimoto."""
    gfps = [fp for fp in (_fp(s) for s in gen_smiles) if fp is not None]
    out = []
    for r in ref_smiles:
        rf = _fp(r)
        if rf is None or not gfps:
            out.append(0.0)
        else:
            out.append(max(BulkTanimotoSimilarity(rf, gfps)))
    return np.array(out)


def _load_unique_kcat(data_csv):
    df = pd.read_csv(data_csv, usecols=["Smiles", "kcat(s^-1)"])
    df["kc"] = pd.to_numeric(df["kcat(s^-1)"], errors="coerce")
    df = df[df["kc"] > 0].copy()
    df["canon"] = df["Smiles"].astype(str).map(lambda s: (Chem.MolToSmiles(Chem.MolFromSmiles(s))
                                                          if Chem.MolFromSmiles(s) else None))
    df = df.dropna(subset=["canon"])
    df["logkcat"] = np.log10(df["kc"])
    g = df.groupby("canon")["logkcat"].max().reset_index()
    return g  # canon, logkcat


def _run_one(g, dec, args, seed):
    """单 seed 回顾性富集; 返回 (res dict, 末次生成物用于画图)."""
    rng = np.random.default_rng(seed)
    hi_thr = g["logkcat"].quantile(0.75)
    lo_thr = g["logkcat"].quantile(0.25)
    high = g[g["logkcat"] >= hi_thr]["canon"].tolist()
    low = g[g["logkcat"] <= lo_thr]["canon"].tolist()
    rng.shuffle(high); rng.shuffle(low)
    test_high = high[: args.n_test]
    test_low = low[: args.n_test]
    holdout = set(test_high) | set(test_low)
    train = g[~g["canon"].isin(holdout)]
    # 可选: 去近邻控制 — 从 surrogate 训练集剔除对留出 HIGH 的 Tc≥阈值近邻(治 analogue 记忆)
    nn_remove = getattr(args, "nn_remove", 0.0)
    if nn_remove and nn_remove > 0:
        th_fps = [f for f in (_fp(s) for s in test_high) if f is not None]
        keep = []
        for c in train["canon"]:
            f = _fp(c)
            keep.append(bool(f is not None and th_fps and max(BulkTanimotoSimilarity(f, th_fps)) < nn_remove))
        n_before = len(train); train = train[keep]
        print(f"[seed={seed}] nn-remove Tc>={nn_remove}: train {n_before} -> {len(train)} "
              f"(剔除留出 HIGH 的近邻)")
    print(f"[seed={seed}] unique: {len(g)} | high(>=p75): {len(high)} | "
          f"held-out HIGH: {len(test_high)}, LOW: {len(test_low)} | train: {len(train)}")

    # 真实标签 surrogate(留出严格)
    sur = KcatSurrogate(seed=seed).fit(train["canon"], train["logkcat"], sigma=None)
    prob = MoleculeLatentProblem(dec, sur, z_bound=3.0)

    # warm-start: 用 train 里的高 kcat 分子(不含留出)
    train_high = train[train["logkcat"] >= hi_thr]["canon"].tolist()
    rng.shuffle(train_high)
    init_X = dec.encode(train_high[: args.n_init]) if train_high else None

    cu = cu_mobo(prob, total_budget=args.budget, n_init=args.n_init, batch_q=args.batch_q,
                 seed=seed, init_X=init_X)
    gen_cu = [s for s in dec.decode(cu["X_all"]) if s]
    # 随机 VAE 基线(同等数量)
    gen_rand = [s for s in dec.decode(rng.standard_normal((len(cu["X_all"]), dec.latent_dim))) if s]
    print(f"[seed={seed}] generated: CU-MOBO {len(gen_cu)} valid | random {len(gen_rand)} valid")

    def recovery(gen, ref):
        sim = _max_sim_to_ref(gen, ref)
        return float(np.mean(sim >= args.tau)), float(sim.mean())

    rec_cu_hi, sim_cu_hi = recovery(gen_cu, test_high)
    rec_rd_hi, sim_rd_hi = recovery(gen_rand, test_high)
    rec_cu_lo, sim_cu_lo = recovery(gen_cu, test_low)
    enrich = rec_cu_hi / rec_rd_hi if rec_rd_hi > 0 else float("nan")

    res = {
        "seed": seed, "held_out_high": len(test_high), "tau": args.tau,
        "recovery_HIGH_cu": rec_cu_hi, "recovery_HIGH_random": rec_rd_hi,
        "enrichment_vs_random": enrich,
        "recovery_LOW_cu": rec_cu_lo,
        "specificity_high_over_low": (rec_cu_hi / rec_cu_lo) if rec_cu_lo > 0 else float("nan"),
        "mean_maxsim_HIGH_cu": sim_cu_hi, "mean_maxsim_HIGH_random": sim_rd_hi,
    }
    print(f"[seed={seed}] recovery@{args.tau} CU {rec_cu_hi:.3f} vs rand {rec_rd_hi:.3f} "
          f"→ enrich {enrich:.2f}× | spec {res['specificity_high_over_low']:.2f}×")
    return res, (gen_cu, gen_rand, test_high, test_low)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default=DEFAULT_VAE)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--n-init", type=int, default=60)
    ap.add_argument("--batch-q", type=int, default=12)
    ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--tau", type=float, default=0.4)
    ap.add_argument("--nn-remove", type=float, default=0.0,
                    help="去近邻控制: 从 surrogate 训练剔除对留出 HIGH 的 Tc≥此值近邻(0=关)")
    ap.add_argument("--seed", type=int, default=0, help="起始 seed")
    ap.add_argument("--seeds", type=int, default=1, help="seed 数量 (从 --seed 起), >1 出误差棒")
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "results"))
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)

    g = _load_unique_kcat(args.data)  # seed 无关, 只加载一次
    dec = VaeDecoder(args.vae)

    rows = []
    last_gen = None
    for s in range(args.seed, args.seed + args.seeds):
        res, gen = _run_one(g, dec, args, s)
        rows.append(res)
        last_gen = gen

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(out_dir, "retrospective_kcat_raw.csv"), index=False)

    import json
    metric_cols = ["recovery_HIGH_cu", "recovery_HIGH_random", "enrichment_vs_random",
                   "recovery_LOW_cu", "specificity_high_over_low",
                   "mean_maxsim_HIGH_cu", "mean_maxsim_HIGH_random"]
    summary = {"n_seeds": args.seeds, "tau": args.tau}
    for c in metric_cols:
        summary[f"{c}_mean"] = float(raw[c].mean())
        summary[f"{c}_std"] = float(raw[c].std(ddof=0))
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, "retrospective_kcat_summary.csv"), index=False)

    print(f"\n==== 路 A: 真实 kcat 回顾性富集 ({args.seeds} seed) ====")
    if args.seeds > 1:
        print(f"recovery_HIGH_cu     = {summary['recovery_HIGH_cu_mean']:.3f} ± {summary['recovery_HIGH_cu_std']:.3f}")
        print(f"recovery_HIGH_random = {summary['recovery_HIGH_random_mean']:.3f} ± {summary['recovery_HIGH_random_std']:.3f}")
        print(f"enrichment_vs_random = {summary['enrichment_vs_random_mean']:.3f} ± {summary['enrichment_vs_random_std']:.3f}")
        print(f"specificity (hi/lo)  = {summary['specificity_high_over_low_mean']:.3f} ± {summary['specificity_high_over_low_std']:.3f}")
    else:
        print(json.dumps(rows[0], indent=2, ensure_ascii=False))

    if args.plot and last_gen is not None:
        _plot(*last_gen, out_dir, args.tau)
        print(f"图: {os.path.join(out_dir, 'retrospective_kcat.png')}")


def _plot(gen_cu, gen_rand, test_high, test_low, out_dir, tau):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..core.plotting import use_cjk_font
    use_cjk_font()

    s_cu_hi = _max_sim_to_ref(gen_cu, test_high)
    s_rd_hi = _max_sim_to_ref(gen_rand, test_high)
    s_cu_lo = _max_sim_to_ref(gen_cu, test_low)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = np.linspace(0, 1, 25)
    ax1.hist(s_cu_hi, bins=bins, alpha=0.6, label="CU-MOBO → 留出高kcat", color="steelblue")
    ax1.hist(s_rd_hi, bins=bins, alpha=0.6, label="随机 → 留出高kcat", color="gray")
    ax1.axvline(tau, color="k", ls="--", lw=1)
    ax1.set_xlabel("对留出真实高kcat底物的最大 Tanimoto")
    ax1.set_ylabel("留出分子数"); ax1.set_title("富集: CU-MOBO vs 随机"); ax1.legend(fontsize=8)
    labels = ["CU→高kcat", "随机→高kcat", "CU→低kcat"]
    recs = [np.mean(s_cu_hi >= tau), np.mean(s_rd_hi >= tau), np.mean(s_cu_lo >= tau)]
    ax2.bar(labels, recs, color=["steelblue", "gray", "lightsteelblue"])
    ax2.set_ylabel(f"recovery@{tau} ↑"); ax2.set_title("留出真实底物的回收率")
    fig.suptitle("路 A: 真实 kcat 标签回顾性富集(留出版)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "retrospective_kcat.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
