"""MOBO —— 多目标贝叶斯优化 (qNEHVI), 方向2兜底 / 样本效率对照基线.

最小化问题转成 botorch 的最大化: 对 Y = -F 建模, 参考点取 -ref_point.
每轮拟合 ModelListGP, 用 qNoisyExpectedHypervolumeImprovement 顺序贪婪挑 q 点.
预算单位 = 真实评估次数.
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

from ..core.archive import ParetoArchive
from ..core.dominance import constraint_violation
from ..core.sampling import latin_hypercube

_DTYPE = torch.double


def mobo(problem, ref_point, total_budget=120, n_init=20, batch_q=10,
         mc_samples=64, num_restarts=5, raw_samples=64,
         archive_size=200, seed=None, monitor=None, verbose=False):
    """返回 dict: X, F (真实前沿), archive, n_real_evals."""
    if seed is not None:
        torch.manual_seed(seed)
    nobj = problem.n_obj
    dim = problem.n_var
    bounds = torch.tensor(np.vstack([problem.xl, problem.xu]), dtype=_DTYPE)
    ref_max = (-np.asarray(ref_point, float)).tolist()

    X = latin_hypercube(n_init, problem.xl, problem.xu, seed=seed)
    F, G = problem.evaluate(X)
    real_evals = n_init

    Xt = torch.as_tensor(X, dtype=_DTYPE)
    Yt = torch.as_tensor(-F, dtype=_DTYPE)   # 最大化 -F

    archive = ParetoArchive(max_size=archive_size)
    archive.add(X, F, constraint_violation(G))
    if monitor is not None:
        monitor(real_evals, archive)

    while real_evals < total_budget:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models = [
                SingleTaskGP(
                    Xt, Yt[:, m: m + 1],
                    input_transform=Normalize(d=dim, bounds=bounds),
                    outcome_transform=Standardize(m=1),
                )
                for m in range(nobj)
            ]
            model = ModelListGP(*models)
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
            acqf = qNoisyExpectedHypervolumeImprovement(
                model=model, ref_point=ref_max, X_baseline=Xt,
                sampler=sampler, prune_baseline=True,
            )
            q = int(min(batch_q, total_budget - real_evals))
            cand, _ = optimize_acqf(
                acqf, bounds=bounds, q=q,
                num_restarts=num_restarts, raw_samples=raw_samples,
                sequential=True,
            )
        Xq = cand.detach().cpu().numpy()
        Fq, Gq = problem.evaluate(Xq)
        real_evals += len(Xq)

        Xt = torch.cat([Xt, torch.as_tensor(Xq, dtype=_DTYPE)])
        Yt = torch.cat([Yt, torch.as_tensor(-Fq, dtype=_DTYPE)])
        archive.add(Xq, Fq, constraint_violation(Gq))
        if monitor is not None:
            monitor(real_evals, archive)
        if verbose:
            print(f"[MOBO] real_evals={real_evals}  |archive|={len(archive)}")

    return {"X": archive.X, "F": archive.F, "archive": archive,
            "n_real_evals": real_evals}
