"""CU-MOBO —— 校准不确定性多目标贝叶斯优化.

面向昂贵 + 不确定 oracle 的约束多目标逆向设计. 三个原理性机制:
  1. σ 作一等目标: Pareto 优化 [μ↑, −σ↑, qed↑, −sa↑] (无 λ);
  2. 校准 σ 作异方差观测噪声: 活性 GP 用 train_Yvar=σ², 去噪 + 下采信不可靠观测;
  3. qNEHVI 批量风险感知采集, 样本高效.

flags 支持 2x2 消融:
  use_sigma_objective : σ 是否进 Pareto 目标
  use_sigma_noise     : σ² 是否作活性 GP 观测噪声
四种组合: naive(F,F) / σ-目标(T,F) / σ-噪声(F,T) / CU-MOBO 全(T,T).

problem 需提供: xl, xu, n_var, evaluate_oracle(X)->{mu,sigma,qed,sa}.
预算单位 = 真实 oracle 评估次数.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import SumMarginalLogLikelihood

from ..core.dominance import non_dominated_mask
from ..core.sampling import latin_hypercube

_DTYPE = torch.double


def _dominates_max(a, b):
    """maximize 空间: a 是否支配 b (a 各维 >= b 且至少一维 >)."""
    return bool(np.all(a >= b) and np.any(a > b))


def _improves_front(Y_old, y_new):
    """y_new 是否未被 Y_old 任一点支配(=触达/拓展前沿, 记一次 TR success)."""
    return not any(_dominates_max(yo, y_new) for yo in Y_old)


def _tr_bounds(center, L, xl, xu):
    """以 center 为心、边长 L·(xu−xl) 的信赖域盒(裁剪到全局界)."""
    span = (xu - xl)
    lo = np.clip(center - 0.5 * L * span, xl, xu)
    hi = np.clip(center + 0.5 * L * span, xl, xu)
    return torch.tensor(np.vstack([lo, hi]), dtype=_DTYPE)


def _local_idx(X, box, center, xl, xu, min_pts):
    """TuRBO-faithful 局部 GP 的训练子集: 信赖域盒内的点(不足则取离心最近 min_pts 个)."""
    lo, hi = box[0].cpu().numpy(), box[1].cpu().numpy()
    inside = np.where(np.all((X >= lo) & (X <= hi), axis=1))[0]
    if len(inside) >= min_pts:
        return inside
    span = np.where((xu - xl) > 0, xu - xl, 1.0)
    d = np.linalg.norm((X - center) / span, axis=1)
    return np.argsort(d)[:min_pts]


def _objective_matrix(mu, sigma, qed, sa, use_sigma_objective):
    """maximize 空间的目标矩阵 (n, m) 与列名."""
    cols = [mu, -sa, qed]
    names = ["mu", "neg_sa", "qed"]
    if use_sigma_objective:
        cols.append(-sigma)
        names.append("neg_sigma")
    return np.column_stack(cols), names


def _fit_model(Xt, Y, names, sigma, bounds, dim, use_sigma_noise):
    models = []
    for j, nm in enumerate(names):
        yj = torch.as_tensor(Y[:, j: j + 1], dtype=_DTYPE)
        kw = dict(input_transform=Normalize(d=dim, bounds=bounds),
                  outcome_transform=Standardize(m=1))
        if nm == "mu" and use_sigma_noise:
            yvar = torch.as_tensor((np.asarray(sigma, float) ** 2).reshape(-1, 1) + 1e-6,
                                   dtype=_DTYPE)
            gp = SingleTaskGP(Xt, yj, train_Yvar=yvar, **kw)
        else:
            gp = SingleTaskGP(Xt, yj, **kw)
        models.append(gp)
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def cu_mobo_propose(X, mu, sigma, qed, sa, xl, xu, q=10,
                    use_sigma_objective=True, use_sigma_noise=True,
                    mc_samples=24, num_restarts=3, raw_samples=32, seed=None):
    """单步采集: 在已有真实数据 (X, μ, σ, qed, sa) 上拟合 GP, 返回 q 个建议点 X (q, dim).

    供 Mode B 异步人在环闭环调用(每轮 propose 一批送真实 oracle).
    """
    if seed is not None:
        torch.manual_seed(seed)
    X = np.atleast_2d(np.asarray(X, float))
    dim = X.shape[1]
    bounds = torch.tensor(np.vstack([xl, xu]), dtype=_DTYPE)
    Xt = torch.as_tensor(X, dtype=_DTYPE)
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    finite = np.isfinite(sigma)
    if not finite.all():  # 用中位数填补缺失 σ, 避免 train_Yvar=NaN
        sigma = np.where(finite, sigma, np.median(sigma[finite]) if finite.any() else 0.1)
    Y, names = _objective_matrix(mu, sigma, np.asarray(qed, float), np.asarray(sa, float),
                                 use_sigma_objective)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _fit_model(Xt, Y, names, sigma, bounds, dim, use_sigma_noise)
        span = Y.max(0) - Y.min(0)
        ref = (Y.min(0) - 0.05 * np.where(span > 0, span, 1.0)).tolist()
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        acqf = qNoisyExpectedHypervolumeImprovement(
            model=model, ref_point=ref, X_baseline=Xt, sampler=sampler, prune_baseline=True)
        cand, _ = optimize_acqf(acqf, bounds=bounds, q=int(q),
                                num_restarts=num_restarts, raw_samples=raw_samples,
                                sequential=True)
    return cand.detach().cpu().numpy()


def _reliability_mu(mu, sigma, mode, lam, q):
    """D2 自适应可信度: 把 σ 从'最小化目标'换成**单边自适应惩罚** baked 进活性目标。
    μ_eff = μ − λ·relu(σ − τ), τ = σ 的 q 分位(自适应)。只惩罚**超阈(极不可信)**候选
    (= 排除 oracle-gap 陷阱), 不惩罚中等 σ(= 保留新颖/稀有真最优)→ 治 σ-objective 反噬。
    返回 (μ_eff, τ)。"""
    sigma = np.asarray(sigma, float)
    tau = float(np.quantile(sigma, q)) if len(sigma) else 0.0
    mu_eff = np.asarray(mu, float) - lam * np.maximum(0.0, sigma - tau)
    return mu_eff, tau


def _obj_rel(mu, sig, qed, sa, use_sigma_objective, reliability_mode, lam, q):
    """按 reliability_mode 出目标矩阵: 'adaptive' 用单边自适应 σ 惩罚(去 σ-目标),
    其余维持原 _objective_matrix(默认逐字一致)。"""
    if reliability_mode == "adaptive":
        mu_eff, _ = _reliability_mu(mu, sig, "adaptive", lam, q)
        return _objective_matrix(mu_eff, sig, qed, sa, False)
    return _objective_matrix(mu, sig, qed, sa, use_sigma_objective)


def cu_mobo(problem, total_budget=120, n_init=20, batch_q=8,
            use_sigma_objective=True, use_sigma_noise=True,
            mc_samples=24, num_restarts=3, raw_samples=32,
            seed=None, monitor=None, verbose=False, init_X=None,
            trust_region=False, tr_length0=0.8, tr_min=0.5 ** 7, tr_max=1.6,
            tr_succ_tol=3, tr_fail_tol=None,
            reliability_mode="none", rel_lambda=1.0, rel_quantile=0.8):
    """返回 dict: X(前沿解), mu/sigma/qed/sa(前沿观测), X_all, obs_all, n_real_evals.

    init_X: 可选 (m, n_var) 初始设计 (warm-start, 如分子隐空间从真实分子编码);
            不足 n_init 时用 LHS 补齐.
    trust_region: True 时启用 MORBO-lite 信赖域(局部 qNEHVI, 治高维退化, 与 σ-noise
            协同)。以最高 μ 的已评估点为心, 成功(新点拓展前沿)扩张/失败收缩, 坍缩重启。
            默认 False → 与论文路径逐字一致。
    """
    if seed is not None:
        torch.manual_seed(seed)
    xl, xu, dim = problem.xl, problem.xu, problem.n_var
    bounds = torch.tensor(np.vstack([xl, xu]), dtype=_DTYPE)
    rng = np.random.default_rng(seed)
    tr_L, tr_sc, tr_fc = float(tr_length0), 0, 0
    tr_fail_tol = int(tr_fail_tol) if tr_fail_tol is not None else max(4, int(np.ceil(dim / batch_q)))

    if init_X is not None:
        init_X = np.atleast_2d(np.asarray(init_X, float))
        init_X = np.clip(init_X, xl, xu)
        if len(init_X) < n_init:
            extra = latin_hypercube(n_init - len(init_X), xl, xu, seed=seed)
            X_all = np.vstack([init_X, extra])
        else:
            X_all = init_X[:n_init]
    else:
        X_all = latin_hypercube(n_init, xl, xu, seed=seed)
    o = problem.evaluate_oracle(X_all)
    mu_all, sig_all = np.asarray(o["mu"], float), np.asarray(o["sigma"], float)
    qed_all, sa_all = np.asarray(o["qed"], float), np.asarray(o["sa"], float)
    real_evals = n_init
    if monitor is not None:
        monitor(real_evals, X_all, dict(mu=mu_all, sigma=sig_all, qed=qed_all, sa=sa_all))

    while real_evals < total_budget:
        Y_full, names = _obj_rel(mu_all, sig_all, qed_all, sa_all, use_sigma_objective,
                                 reliability_mode, rel_lambda, rel_quantile)
        if trust_region:                           # 信赖域: 局部 GP + 局部采集盒
            # 关键1: oracle-gap 下陷阱处 μ 虚高, 以 argmax(μ) 为心会锁进陷阱;
            #        用 σ-折扣 (μ−σ) 选心 → 偏向真实最优(低 σ), 与 CU-MOBO 主张一致。
            score = mu_all - sig_all if (use_sigma_objective or use_sigma_noise) else mu_all
            center = X_all[int(np.argmax(score))]
            acq_bounds = _tr_bounds(center, tr_L, xl, xu)
            # 关键2: TuRBO 用**局部 GP**(只拟合信赖域内点), 这才是它高维不退化的根源,
            #        全局 GP + 仅限盒采集 = 半吊子, 故这里取局部子集拟合。
            loc = _local_idx(X_all, acq_bounds, center, xl, xu, min_pts=max(2 * dim, 8))
        else:
            acq_bounds = bounds
            loc = np.arange(len(X_all))
        Xsub, Ysub, sigsub = X_all[loc], Y_full[loc], sig_all[loc]
        Xt = torch.as_tensor(Xsub, dtype=_DTYPE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _fit_model(Xt, Ysub, names, sigsub, bounds, dim, use_sigma_noise)
            span = Ysub.max(0) - Ysub.min(0)
            ref = (Ysub.min(0) - 0.05 * np.where(span > 0, span, 1.0)).tolist()
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
            acqf = qNoisyExpectedHypervolumeImprovement(
                model=model, ref_point=ref, X_baseline=Xt,
                sampler=sampler, prune_baseline=True)
            q = int(min(batch_q, total_budget - real_evals))
            cand, _ = optimize_acqf(acqf, bounds=acq_bounds, q=q,
                                    num_restarts=num_restarts, raw_samples=raw_samples,
                                    sequential=True)
        Xq = cand.detach().cpu().numpy()
        oq = problem.evaluate_oracle(Xq)
        if trust_region:                           # 成功(新点改进 σ-折扣最优)→ 扩张; 失败→ 收缩; 坍缩→ 重启
            new_score = float((np.asarray(oq["mu"], float) - np.asarray(oq["sigma"], float)).max())
            best_score = float((mu_all - sig_all).max())   # 追加前的当前最优(与居中同口径)
            if new_score > best_score + 1e-9:
                tr_sc += 1; tr_fc = 0
            else:
                tr_fc += 1; tr_sc = 0
            if tr_sc >= tr_succ_tol:
                tr_L = min(tr_L * 2.0, tr_max); tr_sc = 0
            elif tr_fc >= tr_fail_tol:
                tr_L /= 2.0; tr_fc = 0
            if tr_L < tr_min:                      # 信赖域坍缩 → 重启(新心 + 复位边长)
                tr_L, tr_sc, tr_fc = float(tr_length0), 0, 0
        X_all = np.vstack([X_all, Xq])
        mu_all = np.concatenate([mu_all, np.asarray(oq["mu"], float)])
        sig_all = np.concatenate([sig_all, np.asarray(oq["sigma"], float)])
        qed_all = np.concatenate([qed_all, np.asarray(oq["qed"], float)])
        sa_all = np.concatenate([sa_all, np.asarray(oq["sa"], float)])
        real_evals += len(Xq)
        if monitor is not None:
            monitor(real_evals, X_all, dict(mu=mu_all, sigma=sig_all, qed=qed_all, sa=sa_all))
        if verbose:
            print(f"[CU-MOBO] real_evals={real_evals}"
                  + (f" tr_L={tr_L:.4f}" if trust_region else ""))

    # 最终前沿: 在观测目标空间取非支配(reliability_mode 一致)
    Y, _ = _obj_rel(mu_all, sig_all, qed_all, sa_all, use_sigma_objective,
                    reliability_mode, rel_lambda, rel_quantile)
    mask = non_dominated_mask(-Y)   # non_dominated_mask 假设最小化, 故取 −Y
    return {
        "X": X_all[mask],
        "mu": mu_all[mask], "sigma": sig_all[mask], "qed": qed_all[mask], "sa": sa_all[mask],
        "X_all": X_all,
        "obs_all": dict(mu=mu_all, sigma=sig_all, qed=qed_all, sa=sa_all),
        "n_real_evals": real_evals,
    }
