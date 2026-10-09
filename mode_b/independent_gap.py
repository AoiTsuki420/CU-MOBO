"""独立 oracle 交叉验证 —— 打破"自评循环"(molgen PAPER_PLAN §3B/§3C/§3D).

输入两份对**同一批分子**的预测:
  --kcatmoe     KcatMoE 预测 (smiles, pred_kcat_log10=μ, sigma_kcat_log10=σ)
  --independent 独立 oracle 预测 (smiles, <活性列>)  —— 如 UniKP/DLKcat/CataPro

把独立 oracle 当"真值代理", 度量:
  1. oracle-gap = |μ − y_indep| (KcatMoE 预测 vs 独立信号之差);
  2. 朴素 top-K(按 μ) vs 不确定性感知 top-K(按 μ−λσ) 的**独立真实活性**对比
     —— σ 门控是否真的挑出"独立 oracle 也认可"的分子;
  3. σ 校准性: Spearman(σ, oracle-gap) > 0 说明 σ 能标记 KcatMoE 出错处;
  4. σ 门控对被选集 oracle-gap 的削减.

这是 true-hit / 反循环的关键证据: 用**不依赖 KcatMoE 自身**的信号判定成败.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..molecule._molgen import canonicalize_smiles

_IND_ALIASES = ["pred_kcat_log10", "kcat_log10", "y_pred", "pred", "kcat", "activity", "log10_kcat"]


def _canon_key(df):
    df = df.copy()
    df["_k"] = df["smiles"].astype(str).map(lambda s: canonicalize_smiles(s) or s)
    return df


def analyze(kcatmoe_csv, independent_csv, top_k=10, lam=1.0, ind_col=None):
    a = _canon_key(pd.read_csv(kcatmoe_csv))
    b = _canon_key(pd.read_csv(independent_csv))
    if ind_col is None:
        low = {c.lower(): c for c in b.columns}
        ind_col = next((low[x] for x in _IND_ALIASES if x in low), None)
        if ind_col is None:
            raise ValueError(f"独立 oracle 活性列未找到; 见 {list(b.columns)}")
    a = a.rename(columns={"pred_kcat_log10": "mu", "sigma_kcat_log10": "sigma"})
    m = a.merge(b[["_k", ind_col]].rename(columns={ind_col: "y_ind"}), on="_k", how="inner")
    m = m.dropna(subset=["mu", "sigma", "y_ind"]).reset_index(drop=True)
    if len(m) < 3:
        raise ValueError(f"两份预测仅 {len(m)} 个共同分子, 不足以分析")

    m["mu"] = pd.to_numeric(m["mu"], errors="coerce")
    m["sigma"] = pd.to_numeric(m["sigma"], errors="coerce")
    m["y_ind"] = pd.to_numeric(m["y_ind"], errors="coerce")
    m["gap"] = (m["mu"] - m["y_ind"]).abs()
    m["acq"] = m["mu"] - lam * m["sigma"]

    k = min(top_k, len(m))
    naive = m.sort_values("mu", ascending=False).head(k)
    ua = m.sort_values("acq", ascending=False).head(k)
    rho, p = spearmanr(m["sigma"], m["gap"])

    res = {
        "n_common": int(len(m)),
        "top_k": int(k),
        "lambda": lam,
        "naive_topk_true_indep": float(naive["y_ind"].mean()),
        "ua_topk_true_indep": float(ua["y_ind"].mean()),
        "naive_topk_gap": float(naive["gap"].mean()),
        "ua_topk_gap": float(ua["gap"].mean()),
        "naive_topk_sigma": float(naive["sigma"].mean()),
        "ua_topk_sigma": float(ua["sigma"].mean()),
        "sigma_gap_spearman": float(rho),
        "sigma_gap_spearman_p": float(p),
    }
    return res, m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kcatmoe", required=True)
    ap.add_argument("--independent", required=True)
    ap.add_argument("--ind-col", default=None)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    res, m = analyze(args.kcatmoe, args.independent, args.top_k, args.lam, args.ind_col)
    import json
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n[解读] 独立 oracle 下: σ-感知 top-{res['top_k']} 真实活性 "
          f"{res['ua_topk_true_indep']:.3f} vs 朴素 {res['naive_topk_true_indep']:.3f}; "
          f"oracle-gap {res['ua_topk_gap']:.3f} vs {res['naive_topk_gap']:.3f}; "
          f"σ–误差 Spearman={res['sigma_gap_spearman']:.3f}"
          f"({'校准' if res['sigma_gap_spearman'] > 0 else '未校准'}).")
    if args.out:
        m.to_csv(args.out, index=False)
        print(f"明细写入 {args.out}")


if __name__ == "__main__":
    main()
