"""验证 CU-MOBO 的命门前提: σ 是否**真校准**(σ 追踪真实误差)。

方法名是 "Calibrated-Uncertainty", 但旧稿只**断言**校准、从未**测量**。本脚本在**真实 kcat 标签**
上, 对同一 RF 代理的两种 σ 做标准回归校准评估, 直接回答审稿人"σ 凭什么算校准":
  - tree-var σ : RF 树间方差(epistemic 距离, 旧默认);
  - OOF σ      : out-of-fold 残差训练的误差预测头(`sigma="calibrated"`)。
μ 两者相同 → |err|=|y−μ| 相同, 只比"哪种 σ 更会排序/匹配同一批误差"。

两类校准都报(本文都依赖):
  - **排序校准 Spearman(σ,|err|)**(scale-free)= 撑 **σ-as-Pareto-objective**(常数 σ 无用);
  - **离散度/平均校准 ENCE + reliability diagram**(全局尺度对齐后看形状)= 撑 **σ²-as-train_Yvar**。
辅助: NLL、sharpness、Cv。图 = reliability diagram + 误差校准曲线(RMSE_j vs RMV_j)。

用法:
    python -m puridiff_optim.benchmark.run_calibration --test-frac 0.25 --seed 0 --plot
参考: Kuleshov ICML2018; Levi ENCE(Sensors2022); Hirschfeld JCIM2020; Scalia JCIM2020; Pernot 2305.11905。
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from ..molecule import KcatSurrogate
from .run_retrospective_kcat import _load_unique_kcat, DEFAULT_DATA

warnings.filterwarnings("ignore")


def _reliability(y, mu, sigma, n=50):
    """居中区间覆盖: 名义 p vs 观测; 返回 (p_grid, obs, miscalibration_area)."""
    z = np.abs((y - mu) / np.maximum(sigma, 1e-9))
    ps = np.linspace(0.0, 1.0, n)
    k = norm.ppf(0.5 + ps / 2.0)            # 居中区间半宽(σ 单位), p=1 -> inf
    obs = np.array([np.mean(z <= kk) for kk in k])
    area = float(np.trapz(np.abs(obs - ps), ps))
    return ps, obs, area


def _ence(y, mu, sigma, n_bins=10):
    """按 σ 等数分箱, 误差校准曲线 RMSE_j vs RMV_j + ENCE."""
    order = np.argsort(sigma)
    e2 = (y - mu) ** 2
    bins = np.array_split(order, n_bins)
    rmv = np.array([np.sqrt(np.mean(sigma[b] ** 2)) for b in bins])
    rmse = np.array([np.sqrt(np.mean(e2[b])) for b in bins])
    ence = float(np.mean(np.abs(rmv - rmse) / np.maximum(rmv, 1e-9)))
    return rmv, rmse, ence


def _metrics(y, mu, sigma_raw):
    err = np.abs(y - mu)
    # 全局尺度对齐(单标量): 让总 RMV 匹配总 RMSE → 隔离"形状/离散度校准"与绝对尺度
    rmse_tot = np.sqrt(np.mean((y - mu) ** 2))
    rmv_tot = np.sqrt(np.mean(sigma_raw ** 2))
    scale = rmse_tot / max(rmv_tot, 1e-9)
    sig = sigma_raw * scale
    ps, obs, area = _reliability(y, mu, sig)
    rmv, rmse, ence = _ence(y, mu, sig)
    nll = float(np.mean(0.5 * np.log(2 * np.pi * sig ** 2) + (y - mu) ** 2 / (2 * sig ** 2)))
    rho, p = spearmanr(sigma_raw, err)      # 排序校准用原始 σ(scale-free)
    return dict(
        spearman=float(rho), spearman_p=float(p),
        ence=ence, miscal_area=area, nll=nll,
        sharpness=float(np.sqrt(np.mean(sigma_raw ** 2))),
        cv=float(np.std(sigma_raw) / max(np.mean(sigma_raw), 1e-9)),
        scale=float(scale),
        _curve=dict(ps=ps, obs=obs, rmv=rmv, rmse=rmse),
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--max-n", type=int, default=6000, help="子采样上限(加速)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)
    out_dir = os.path.abspath(args.out); os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    g = _load_unique_kcat(args.data)        # canon, logkcat
    if len(g) > args.max_n:
        g = g.iloc[rng.permutation(len(g))[:args.max_n]].reset_index(drop=True)
    idx = rng.permutation(len(g))
    n_test = int(len(g) * args.test_frac)
    te, tr = idx[:n_test], idx[n_test:]
    s_tr = g["canon"].values[tr]; y_tr = g["logkcat"].values[tr]
    s_te = g["canon"].values[te]; y_te = g["logkcat"].values[te]
    print(f"unique {len(g)} | train {len(tr)} | test {len(te)} | log10(kcat) range "
          f"[{g['logkcat'].min():.2f},{g['logkcat'].max():.2f}]")

    rows = {}
    curves = {}
    for tag, sig_mode in [("tree-var", None), ("OOF-calibrated", "calibrated")]:
        sur = KcatSurrogate(seed=args.seed).fit(s_tr, y_tr, sigma=sig_mode)
        mu, sigma = sur.predict(s_te)
        ok = ~np.isnan(mu) & ~np.isnan(sigma)
        m = _metrics(y_te[ok], mu[ok], np.maximum(sigma[ok], 1e-6))
        curves[tag] = m.pop("_curve")
        rows[tag] = m
        print(f"[{tag:14s}] Spearman(σ,|err|)={m['spearman']:.3f}(p={m['spearman_p']:.1e}) "
              f"ENCE={m['ence']:.3f} miscal_area={m['miscal_area']:.3f} NLL={m['nll']:.3f} "
              f"sharp={m['sharpness']:.3f} Cv={m['cv']:.2f}")

    df = pd.DataFrame(rows).T.reset_index().rename(columns={"index": "sigma_mode"})
    df.to_csv(os.path.join(out_dir, "calibration_summary.csv"), index=False)
    print("\n==== σ 校准评估(真实 kcat, held-out) ====")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(df.drop(columns=["spearman_p", "scale"]).to_string(index=False))
    better = "OOF-calibrated" if rows["OOF-calibrated"]["spearman"] > rows["tree-var"]["spearman"] else "tree-var"
    print(f"\n排序校准(σ-as-objective 命门): {better} 更会排序误差 "
          f"(Spearman {rows['tree-var']['spearman']:.3f} -> {rows['OOF-calibrated']['spearman']:.3f})。")

    if args.plot:
        _plot(curves, rows, out_dir)


def _plot(curves, rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..core.plotting import use_cjk_font
    use_cjk_font()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    colors = {"tree-var": "tab:gray", "OOF-calibrated": "tab:orange"}
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="完美校准")
    for tag, c in curves.items():
        ax1.plot(c["ps"], c["obs"], color=colors[tag], lw=2,
                 label=f"{tag} (miscal={rows[tag]['miscal_area']:.3f})")
    ax1.set_xlabel("名义覆盖度 p"); ax1.set_ylabel("观测覆盖度")
    ax1.set_title("Reliability diagram (σ²-as-train_Yvar 命门)"); ax1.legend(fontsize=8)
    lim = 0
    for tag, c in curves.items():
        lim = max(lim, c["rmv"].max(), c["rmse"].max())
    ax2.plot([0, lim], [0, lim], "k--", lw=1, label="完美")
    for tag, c in curves.items():
        ax2.plot(c["rmv"], c["rmse"], "o-", color=colors[tag], lw=2,
                 label=f"{tag} (ENCE={rows[tag]['ence']:.3f}, ρ={rows[tag]['spearman']:.2f})")
    ax2.set_xlabel("预测 RMV = sqrt(mean σ²) (分箱)"); ax2.set_ylabel("实测 RMSE (分箱)")
    ax2.set_title("误差校准曲线 (σ-as-objective: ρ=Spearman)"); ax2.legend(fontsize=8)
    fig.suptitle("CU-MOBO σ 校准验证: OOF-校准 σ vs RF 树方差 σ (真实 kcat held-out)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "calibration.png"), dpi=130); plt.close(fig)
    print(f"图: {os.path.join(out_dir, 'calibration.png')}")


if __name__ == "__main__":
    main()
