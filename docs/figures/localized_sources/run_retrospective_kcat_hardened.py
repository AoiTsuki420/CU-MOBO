"""§4.3 严格化(回应富集调研的三处批评): 标准 VS 指标 + 骨架划分 + 记忆基线 + bootstrap CI.

原 §4.3 用 recovery@Tanimoto0.4 + recovery_CU/recovery_random, 被指: (a) 0.4 太松;
(b) 随机划分有 analogue-bias(留出近邻仍在训练); (c) 基线弱。本脚本补一条**严格的检索式
(surrogate-screening)富集**——直接测"留出严格的真实-标签 kcat surrogate 能否把真高-kcat 底物
排在前面", 用领域标准指标, 不依赖生成、跑得快:

  - 指标: **EF@1%/5%/10% + BEDROC(α=80.5) + ROC-AUC**(`rdkit.ML.Scoring`);
  - 划分: **随机 vs Bemis-Murcko 骨架划分**(骨架划分=训练/测试骨架不相交, 治 analogue-bias);
  - 基线: **1-NN Tanimoto 记忆基线**(按"到训练高-kcat 的最大相似度"排序)——若纯记忆即可, surrogate 不应明显胜它;
  - 统计: **bootstrap 95% CI**(对留出 active 重抽样)。

池: actives = 留出 HIGH(≥p_hi), decoys = 留出随机背景(<p_lo); 二者均不入 surrogate 训练(控泄漏)。
诚实: 这测的是 **surrogate 的泛化/可排序性**(CU-MOBO 优化的那个目标本身可信), 与原 §4.3 的**生成**回收互补。

用法:
    python -m puridiff_optim.benchmark.run_retrospective_kcat_hardened --seeds 5 --plot
参考: Truchon&Bayly JCIM2007(BEDROC); Ash&Hughes-Oliver JCheminf2022(CI); Tossou 2406.00873(scaffold split); LIT-PCBA 2507.21404(1-NN 记忆基线)。
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
from rdkit.ML.Scoring import Scoring

from ..molecule import KcatSurrogate
from .run_retrospective_kcat import _load_unique_kcat, DEFAULT_DATA

warnings.filterwarnings("ignore")
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _fp(s):
    m = Chem.MolFromSmiles(str(s))
    return _GEN.GetFingerprint(m) if m else None


def _scaffold(s):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=str(s), includeChirality=False)
    except Exception:
        return ""


def _vs_metrics(scores, labels):
    """scores↑ 越像 active; labels 1=active. 返回 EF1/5/10, BEDROC(80.5), AUC."""
    order = np.argsort(-np.asarray(scores))
    ranked = [[int(labels[i])] for i in order]      # 已按分降序; activity 在 col 0
    ef = Scoring.CalcEnrichment(ranked, 0, [0.01, 0.05, 0.10])
    bedroc = Scoring.CalcBEDROC(ranked, 0, 80.5)
    auc = Scoring.CalcAUC(ranked, 0)
    return dict(EF1=float(ef[0]), EF5=float(ef[1]), EF10=float(ef[2]),
                BEDROC=float(bedroc), AUC=float(auc))


def _holdout_random(g, hi_thr, lo_thr, n_act, n_dec, rng):
    high = g[g["logkcat"] >= hi_thr]["canon"].tolist()
    bg = g[g["logkcat"] < lo_thr]["canon"].tolist()
    rng.shuffle(high); rng.shuffle(bg)
    act = high[:n_act]; dec = bg[:n_dec]
    hold = set(act) | set(dec)
    train = g[~g["canon"].isin(hold)]
    return act, dec, train


def _holdout_scaffold(g, hi_thr, lo_thr, n_act, n_dec, rng):
    """按 Bemis-Murcko 骨架整组留出 → 训练/测试骨架不相交."""
    gg = g.copy()
    gg["scaf"] = gg["canon"].map(_scaffold)
    scafs = [s for s in gg["scaf"].unique() if s]
    rng.shuffle(scafs)
    act, dec, held_scaf = [], [], set()
    for sc in scafs:
        if len(act) >= n_act and len(dec) >= n_dec:
            break
        members = gg[gg["scaf"] == sc]
        held_scaf.add(sc)
        for c, lk in zip(members["canon"], members["logkcat"]):
            if lk >= hi_thr and len(act) < n_act:
                act.append(c)
            elif lk < lo_thr and len(dec) < n_dec:
                dec.append(c)
    train = gg[~gg["scaf"].isin(held_scaf)]
    return act, dec, train[["canon", "logkcat"]]


def _nn_to_trainhigh(pool, train_high):
    """1-NN 记忆基线: 每个池分子到训练集高-kcat 的最大 Tanimoto."""
    tfps = [fp for fp in (_fp(s) for s in train_high) if fp is not None]
    out = []
    for s in pool:
        f = _fp(s)
        out.append(max(BulkTanimotoSimilarity(f, tfps)) if (f is not None and tfps) else 0.0)
    return np.array(out)


def _run_one(g, split, n_act, n_dec, seed):
    rng = np.random.default_rng(seed)
    hi_thr = g["logkcat"].quantile(0.75); lo_thr = g["logkcat"].quantile(0.50)
    ho = _holdout_scaffold if split == "scaffold" else _holdout_random
    act, dec, train = ho(g, hi_thr, lo_thr, n_act, n_dec, rng)
    pool = list(act) + list(dec)
    labels = np.array([1] * len(act) + [0] * len(dec))
    sur = KcatSurrogate(seed=seed).fit(train["canon"], train["logkcat"], sigma=None)
    mu, _ = sur.predict(pool)
    ok = ~np.isnan(mu)
    pool_ok = [p for p, k in zip(pool, ok) if k]; lab_ok = labels[ok]
    m_sur = _vs_metrics(mu[ok], lab_ok)
    # 1-NN 记忆基线
    train_high = train[train["logkcat"] >= hi_thr]["canon"].tolist()
    nn = _nn_to_trainhigh(pool_ok, train_high)
    m_nn = _vs_metrics(nn, lab_ok)
    row = {"split": split, "seed": seed, "n_act": int(lab_ok.sum()), "n_pool": int(len(lab_ok))}
    for k, v in m_sur.items():
        row[f"sur_{k}"] = v
    for k, v in m_nn.items():
        row[f"nn_{k}"] = v
    print(f"[{split:8s} seed={seed}] act={int(lab_ok.sum())}/{len(lab_ok)} | "
          f"surrogate EF1={m_sur['EF1']:.2f} EF5={m_sur['EF5']:.2f} BEDROC={m_sur['BEDROC']:.3f} "
          f"AUC={m_sur['AUC']:.3f} || 1-NN AUC={m_nn['AUC']:.3f} BEDROC={m_nn['BEDROC']:.3f}")
    return row


def _bootstrap_ci(vals, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, float); vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    bs = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--n-act", type=int, default=60)
    ap.add_argument("--n-dec", type=int, default=540, help="背景 decoy 数(→ active 占比~10%)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--splits", nargs="+", default=["random", "scaffold"])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)
    out_dir = os.path.abspath(args.out); os.makedirs(out_dir, exist_ok=True)

    g = _load_unique_kcat(args.data)
    print(f"unique {len(g)} | actives≥p75, decoys<p50 | pool≈{args.n_act}+{args.n_dec}")
    rows = []
    for split in args.splits:
        for s in range(args.seeds):
            rows.append(_run_one(g, split, args.n_act, args.n_dec, s))
    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(out_dir, "retrospective_hardened_raw.csv"), index=False)

    # 汇总 + bootstrap CI(对 per-seed 值)
    summ = []
    for split in args.splits:
        sub = raw[raw.split == split]
        rec = {"split": split, "seeds": len(sub)}
        for col in ["sur_EF1", "sur_EF5", "sur_BEDROC", "sur_AUC", "nn_AUC", "nn_BEDROC"]:
            lo, hi = _bootstrap_ci(sub[col].values)
            rec[f"{col}_mean"] = float(sub[col].mean()); rec[f"{col}_lo"] = lo; rec[f"{col}_hi"] = hi
        summ.append(rec)
    sdf = pd.DataFrame(summ)
    sdf.to_csv(os.path.join(out_dir, "retrospective_hardened_summary.csv"), index=False)
    print("\n==== §4.3 严格化: surrogate 检索式富集 (EF/BEDROC/AUC, 95% bootstrap CI) ====")
    for r in summ:
        print(f"[{r['split']:8s}] surrogate: AUC {r['sur_AUC_mean']:.3f}[{r['sur_AUC_lo']:.3f},{r['sur_AUC_hi']:.3f}] "
              f"BEDROC {r['sur_BEDROC_mean']:.3f}[{r['sur_BEDROC_lo']:.3f},{r['sur_BEDROC_hi']:.3f}] "
              f"EF1 {r['sur_EF1_mean']:.2f} EF5 {r['sur_EF5_mean']:.2f}  ||  "
              f"1-NN: AUC {r['nn_AUC_mean']:.3f} BEDROC {r['nn_BEDROC_mean']:.3f}")

    if args.plot:
        _plot(raw, args.splits, out_dir)


def _plot(raw, splits, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..core.plotting import use_cjk_font
    use_cjk_font()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(splits)); w = 0.35
    for ax, met, title in [(ax1, "AUC", "ROC-AUC"), (ax2, "BEDROC", "BEDROC(α=80.5)")]:
        sur = [raw[raw.split == sp][f"sur_{met}"].mean() for sp in splits]
        nn = [raw[raw.split == sp][f"nn_{met}"].mean() for sp in splits]
        sur_e = [raw[raw.split == sp][f"sur_{met}"].std() for sp in splits]
        nn_e = [raw[raw.split == sp][f"nn_{met}"].std() for sp in splits]
        ax.bar(x - w / 2, sur, w, yerr=sur_e, label="kcat surrogate", color="steelblue")
        ax.bar(x + w / 2, nn, w, yerr=nn_e, label="1-NN 记忆基线", color="gray")
        ax.axhline(0.5 if met == "AUC" else 0.0, color="k", ls=":", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(splits); ax.set_title(title); ax.legend(fontsize=8)
    fig.suptitle("§4.3 严格化: surrogate 检索真高-kcat 底物 (随机 vs 骨架划分; vs 1-NN 记忆)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "retrospective_hardened.png"), dpi=130)
    plt.close(fig)
    print(f"图: {os.path.join(out_dir, 'retrospective_hardened.png')}")


if __name__ == "__main__":
    main()
