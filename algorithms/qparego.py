"""qParEGO —— 标准多目标贝叶斯优化基线(随机标量化, ParEGO 的批量 BoTorch 版).

每轮抽一个随机权重 -> 增广 Chebyshev 标量化目标 -> 单输出 GP -> qLogNEI 批量采集.
**忽略校准 σ**(只用 μ/qed/sa 作目标), 作为"不用不确定性的 SOTA MOBO"对照.
接口同 cu_mobo: problem.evaluate_oracle(X)->{mu,sigma,qed,sa}; 预算=真实评估次数.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

from ..core.dominance import non_dominated_mask
from ..core.sampling import latin_hypercube

_DTYPE = torch.double


def _objectives(mu, sa, qed):
    # maximize 空间: [mu, -sa, qed] (不含 σ)
    return np.column_stack([mu, -sa, qed])


def qparego(problem, total_budget=120, n_init=20, batch_q=8, mc_samples=24,
            num_restarts=3, raw_samples=32, rho=0.05, seed=None, monitor=None):
    if seed is not None:
        torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xl, xu, dim = problem.xl, problem.xu, problem.n_var
    bounds = torch.tensor(np.vstack([xl, xu]), dtype=_DTYPE)

    X_all = latin_hypercube(n_init, xl, xu, seed=seed)
    o = problem.evaluate_oracle(X_all)
    mu_all, sig_all = np.asarray(o["mu"], float), np.asarray(o["sigma"], float)
    qed_all, sa_all = np.asarray(o["qed"], float), np.asarray(o["sa"], float)
    real_evals = n_init
    if monitor is not None:
        monitor(real_evals, X_all, dict(mu=mu_all, sigma=sig_all, qed=qed_all, sa=sa_all))

    while real_evals < total_budget:
        Y = _objectives(mu_all, sa_all, qed_all)
        # 归一化到 [0,1] + 随机权重增广 Chebyshev 标量化
        lo, hi = Y.min(0), Y.max(0)
        Yn = (Y - lo) / np.where(hi - lo > 0, hi - lo, 1.0)
        w = rng.random(Y.shape[1])
        w = w / w.sum()
        wY = w * Yn
        scal = wY.min(1) + rho * wY.sum(1)   # 越大越好

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xt = torch.as_tensor(X_all, dtype=_DTYPE)
            yt = torch.as_tensor(scal.reshape(-1, 1), dtype=_DTYPE)
            gp = SingleTaskGP(Xt, yt, input_transform=Normalize(d=dim, bounds=bounds),
                              outcome_transform=Standardize(m=1))
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
            acqf = qLogNoisyExpectedImprovement(model=gp, X_baseline=Xt, sampler=sampler,
                                                prune_baseline=True)
            q = int(min(batch_q, total_budget - real_evals))
            cand, _ = optimize_acqf(acqf, bounds=bounds, q=q, num_restarts=num_restarts,
                                    raw_samples=raw_samples, sequential=True)
        Xq = cand.detach().cpu().numpy()
        oq = problem.evaluate_oracle(Xq)
        X_all = np.vstack([X_all, Xq])
        mu_all = np.concatenate([mu_all, np.asarray(oq["mu"], float)])
        sig_all = np.concatenate([sig_all, np.asarray(oq["sigma"], float)])
        qed_all = np.concatenate([qed_all, np.asarray(oq["qed"], float)])
        sa_all = np.concatenate([sa_all, np.asarray(oq["sa"], float)])
        real_evals += len(Xq)
        if monitor is not None:
            monitor(real_evals, X_all, dict(mu=mu_all, sigma=sig_all, qed=qed_all, sa=sa_all))

    Y = _objectives(mu_all, sa_all, qed_all)
    mask = non_dominated_mask(-Y)
    return {"X": X_all[mask], "mu": mu_all[mask], "sigma": sig_all[mask],
            "qed": qed_all[mask], "sa": sa_all[mask], "X_all": X_all,
            "obs_all": dict(mu=mu_all, sigma=sig_all, qed=qed_all, sa=sa_all),
            "n_real_evals": real_evals}
