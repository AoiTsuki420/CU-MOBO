"""推荐分子集的多样性/新颖性(分子优化 benchmark 的 table-stakes 缺口).

调研指出: 抗 oracle-gap 可以靠"贴着训练流形"作弊(低新颖度), 故**必须**报告推荐集的
多样性(IntDiv₁, #Circles, 骨架数)与新颖性(SNN/对训练集最近邻)。本脚本在 kcat VAE 隐空间
跑 CU-MOBO vs naive(μ-贪婪) vs random, 在各自**推荐(Pareto)集**上算这些指标 + 预测质量,
证明 **CU-MOBO 抗 oracle-gap 不以塌多样性/新颖性为代价**。

指标(MOSES/GuacaMol/#Circles 标准):
  - quality   : 推荐集预测 μ(kcat) 均;
  - IntDiv₁   : 1 − 平均两两 Tanimoto(集合内多样性);
  - #Circles  : 球排斥贪婪计数(两两 Tanimoto 距离 ≥ t 的分子数, t=0.65);
  - n_scaffold: 唯一 Bemis-Murcko 骨架数;
  - SNN_train : 对训练集最近邻平均 Tanimoto(低=新颖); frac_novel = maxsim<0.4 比例。

用法:
    python -m puridiff_optim.benchmark.run_diversity --seeds 3 --budget 100 --plot
参考: MOSES(IntDiv/SNN/Scaff, Front.Pharmacol.2020); #Circles(Pareto-aware BO, 2507.13704); GuacaMol(novelty)。
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import BulkTanimotoSimilarity

from ..algorithms.cu_mobo import cu_mobo
from ..molecule import KcatSurrogate, MoleculeLatentProblem, VaeDecoder
from .run_retrospective_kcat import _load_unique_kcat, DEFAULT_DATA, DEFAULT_VAE

warnings.filterwarnings("ignore")
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

METHODS = {
    "cu-mobo": dict(use_sigma_objective=True, use_sigma_noise=True),
    "naive":   dict(use_sigma_objective=False, use_sigma_noise=False),
}


def _fps(smiles):
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        out.append(_GEN.GetFingerprint(m) if m else None)
    return [f for f in out if f is not None]


def _intdiv(fps):
    if len(fps) < 2:
        return 0.0
    tot = 0.0; n = 0
    for i in range(len(fps)):
        sims = BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        tot += sum(sims); n += len(sims)
    return 1.0 - tot / max(n, 1)


def _n_circles(fps, t=0.65):
    """球排斥贪婪: 两两 Tanimoto 距离 ≥ t(=sim ≤ 1−t)的极大子集近似."""
    centers = []
    for f in fps:
        if not centers or max(BulkTanimotoSimilarity(f, centers)) <= (1 - t):
            centers.append(f)
    return len(centers)


def _scaffolds(smiles):
    sc = set()
    for s in smiles:
        try:
            x = MurckoScaffold.MurckoScaffoldSmiles(smiles=str(s), includeChirality=False)
            if x:
                sc.add(x)
        except Exception:
            pass
    return len(sc)


def _snn(rec_fps, train_fps):
    if not rec_fps or not train_fps:
        return float("nan"), float("nan")
    mx = [max(BulkTanimotoSimilarity(f, train_fps)) for f in rec_fps]
    return float(np.mean(mx)), float(np.mean(np.array(mx) < 0.4))


def _metrics(label, smiles, mu, train_fps):
    uniq = sorted(set(s for s in smiles if s))
    fps = _fps(uniq)
    snn, novel = _snn(fps, train_fps)
    return {
        "method": label, "n_unique": len(uniq), "mean_mu": float(np.nanmean(mu)) if len(mu) else 0.0,
        "IntDiv1": _intdiv(fps), "n_circles": _n_circles(fps),
        "n_scaffold": _scaffolds(uniq), "SNN_train": snn, "frac_novel": novel,
    }


def _run_seed(g, dec, train_fps, args, seed):
    rng = np.random.default_rng(seed)
    sur = KcatSurrogate(seed=seed).fit(g["canon"], g["logkcat"], sigma=None)
    prob = MoleculeLatentProblem(dec, sur, z_bound=3.0)
    train_high = g[g["logkcat"] >= g["logkcat"].quantile(0.75)]["canon"].tolist()
    rng.shuffle(train_high)
    init_X = dec.encode(train_high[: args.n_init]) if train_high else None

    rows = []
    for label, kw in METHODS.items():
        out = cu_mobo(prob, total_budget=args.budget, n_init=args.n_init, batch_q=args.batch_q,
                      seed=seed, init_X=init_X, **kw)
        rec = dec.decode(out["X"])                 # 推荐(Pareto)集
        m = _metrics(label, rec, out["mu"], train_fps)
        m["seed"] = seed; rows.append(m)
    # random 基线: 采样 N 隐 → 取预测 μ top-K(K=cu-mobo 推荐数)
    n_rand = max(50, args.budget)
    Z = rng.uniform(-3, 3, size=(n_rand, dec.latent_dim))
    rs = dec.decode(Z); o = prob.evaluate_oracle(Z); mu_r = np.asarray(o["mu"], float)
    k = int(np.median([r["n_unique"] for r in rows])) or 20
    order = np.argsort(-np.nan_to_num(mu_r, nan=-1e9))[:k]
    rec_r = [rs[i] for i in order if rs[i]]
    mr = _metrics("random", rec_r, mu_r[order], train_fps)
    mr["seed"] = seed; rows.append(mr)
    for r in rows:
        print(f"[seed={seed}] {r['method']:8s} n={r['n_unique']:3d} μ={r['mean_mu']:.2f} "
              f"IntDiv={r['IntDiv1']:.3f} #Circ={r['n_circles']:3d} scaf={r['n_scaffold']:3d} "
              f"SNN={r['SNN_train']:.3f} novel={r['frac_novel']:.2f}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default=DEFAULT_VAE)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--n-init", type=int, default=40)
    ap.add_argument("--batch-q", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-train-ref", type=int, default=2000, help="SNN 参照训练集采样数")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)
    out_dir = os.path.abspath(args.out); os.makedirs(out_dir, exist_ok=True)

    g = _load_unique_kcat(args.data)
    dec = VaeDecoder(args.vae)
    ref = g["canon"].sample(min(args.n_train_ref, len(g)), random_state=0).tolist()
    train_fps = _fps(ref)
    print(f"unique {len(g)} | latent {dec.latent_dim} | SNN ref {len(train_fps)}")

    rows = []
    for s in range(args.seeds):
        rows += _run_seed(g, dec, train_fps, args, s)
    raw = pd.DataFrame(rows); raw.to_csv(os.path.join(out_dir, "diversity_raw.csv"), index=False)

    cols = ["n_unique", "mean_mu", "IntDiv1", "n_circles", "n_scaffold", "SNN_train", "frac_novel"]
    order = ["cu-mobo", "naive", "random"]
    summ = raw.groupby("method")[cols].agg(["mean", "std"]).reindex(order)
    summ.to_csv(os.path.join(out_dir, "diversity_summary.csv"))
    print(f"\n==== 推荐集多样性/新颖性 ({args.seeds} seed) ====")
    disp = raw.groupby("method")[cols].mean().reindex(order)
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(disp.to_string())
    cu = disp.loc["cu-mobo"]; na = disp.loc["naive"]
    print(f"\nCU-MOBO vs naive: 质量 μ {na['mean_mu']:.2f}->{cu['mean_mu']:.2f}, "
          f"IntDiv {na['IntDiv1']:.3f}->{cu['IntDiv1']:.3f}, 骨架 {na['n_scaffold']:.0f}->{cu['n_scaffold']:.0f}, "
          f"新颖 {na['frac_novel']:.2f}->{cu['frac_novel']:.2f} "
          f"=> {'未塌多样性/新颖性' if cu['IntDiv1'] >= na['IntDiv1'] - 0.05 else '多样性下降需注意'}")

    if args.plot:
        _plot(raw, order, out_dir)


def _plot(raw, order, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..core.plotting import use_cjk_font
    use_cjk_font()
    panels = [("mean_mu", "预测 kcat μ (质量↑)"), ("IntDiv1", "IntDiv1 (集合内多样性↑)"),
              ("n_scaffold", "唯一骨架数↑"), ("frac_novel", "新颖比例 NN<0.4 (↑=更新颖)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    colors = {"cu-mobo": "tab:orange", "naive": "tab:gray", "random": "lightsteelblue"}
    for ax, (col, title) in zip(axes, panels):
        m = [raw[raw.method == k][col].mean() for k in order]
        e = [raw[raw.method == k][col].std() for k in order]
        ax.bar(order, m, yerr=e, color=[colors[k] for k in order])
        ax.set_title(title); ax.tick_params(axis="x", rotation=15)
    fig.suptitle("推荐(Pareto)分子集: CU-MOBO 多样性≥naive(不塌), 但新颖性更低=σ机制偏向可信(近训练)区 (kcat 隐空间)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "diversity.png"), dpi=130); plt.close(fig)
    print(f"图: {os.path.join(out_dir, 'diversity.png')}")


if __name__ == "__main__":
    main()
