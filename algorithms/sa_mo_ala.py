"""SA-MO-ALA —— 代理辅助多目标 ALA (本项目创新主线).

闭环:
    1. 拉丁超立方初采样, 真实评估 -> 档案;
    2. 用全部真实数据拟合 GP 代理 (每目标一个 + 每约束一个);
    3. 在**便宜的代理**上跑内层 MO-ALA (带预测约束, 具备可行性意识);
    4. infill: 在预测可行候选里用乐观超体积增益挑 q 点; 若尚无真实可行点,
       则优先送"预测违反量最小"的点去真实评估以定位可行域;
    5. 仅这 q 个点做**真实评估**, 回填档案与训练集;
    6. 重复直到真实评估预算耗尽.

预算单位 = 真实评估次数 (昂贵), 内层在代理上的评估不计入.
"""

from __future__ import annotations

import numpy as np

from ..core.archive import ParetoArchive
from ..core.dominance import constraint_violation
from ..core.hv_infill import greedy_hvi_batch
from ..core.sampling import latin_hypercube
from ..surrogate.gp import GPSurrogate
from ..surrogate.surrogate_problem import SurrogateProblem
from .mo_ala import mo_ala


def sa_mo_ala(problem, ref_point, total_budget=200, n_init=24, batch_q=8,
              inner_pop=50, inner_fes=3000, inner_kappa=0.0, infill_kappa=1.5,
              archive_size=200, seed=None, monitor=None, verbose=False):
    """返回 dict: X, F (真实前沿), archive, n_real_evals."""
    rng = np.random.default_rng(seed)
    has_constr = problem.n_constr > 0

    # 1. 初采样 + 真实评估
    X = latin_hypercube(n_init, problem.xl, problem.xu, seed=seed)
    F, G = problem.evaluate(X)
    CV = constraint_violation(G)
    if CV is None:
        CV = np.zeros(len(X))
    real_evals = n_init

    archive = ParetoArchive(max_size=archive_size)
    archive.add(X, F, CV)
    X_train, F_train = X.copy(), F.copy()
    G_train = G.copy() if has_constr else None
    if monitor is not None:
        monitor(real_evals, archive)

    while real_evals < total_budget:
        # 2. 拟合 GP 代理 (目标 + 约束)
        sur = GPSurrogate(problem.xl, problem.xu).fit(X_train, F_train, G_train)

        # 3. 内层 MO-ALA 在代理上搜索 (带预测约束)
        sp = SurrogateProblem(problem, sur, kappa=inner_kappa, use_constraints=True)
        inner = mo_ala(sp, n_agents=inner_pop, max_fes=inner_fes,
                       archive_size=archive_size, seed=int(rng.integers(1_000_000_000)))
        cand_X = inner["X"]
        if cand_X is None or len(cand_X) == 0:
            cand_X = latin_hypercube(max(batch_q * 4, 40), problem.xl, problem.xu,
                                     seed=int(rng.integers(1_000_000_000)))

        # 4. 可行性意识 infill
        cand_mean, cand_std = sur.predict(cand_X)
        cand_c = sur.predict_constraints(cand_X) if has_constr else None
        if cand_c is not None:
            pred_viol = np.maximum(cand_c, 0.0).sum(axis=1)
        else:
            pred_viol = np.zeros(len(cand_X))
        q = int(min(batch_q, total_budget - real_evals))

        if len(archive) == 0:
            # 尚无真实可行点: 优先定位可行域 (送预测违反量最小者)
            idx = list(np.argsort(pred_viol)[:q])
        else:
            pool = np.where(pred_viol <= 0.0)[0]
            if len(pool) < q:  # 可行候选不足, 用违反量小的补齐
                order = np.argsort(pred_viol)
                seen = set(pool.tolist())
                extra = [int(i) for i in order if int(i) not in seen]
                pool = np.concatenate([pool, np.array(extra[: q - len(pool)], dtype=int)])
            sub = greedy_hvi_batch(archive.F, cand_mean[pool], cand_std[pool],
                                   ref_point, q, kappa=infill_kappa)
            idx = list(np.asarray(pool)[sub]) if sub else list(pool[:q])

        Xq = np.atleast_2d(cand_X[idx])

        # 5. 真实评估 q 点
        Fq, Gq = problem.evaluate(Xq)
        CVq = constraint_violation(Gq)
        if CVq is None:
            CVq = np.zeros(len(Xq))
        real_evals += len(Xq)

        X_train = np.vstack([X_train, Xq])
        F_train = np.vstack([F_train, Fq])
        if has_constr:
            G_train = np.vstack([G_train, Gq])
        archive.add(Xq, Fq, CVq)
        if monitor is not None:
            monitor(real_evals, archive)
        if verbose:
            print(f"[SA-MO-ALA] real_evals={real_evals}  |archive|={len(archive)}")

    return {"X": archive.X, "F": archive.F, "archive": archive,
            "n_real_evals": real_evals}
