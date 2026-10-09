"""MOPSO —— 多目标粒子群优化 (基线).

Coello Coello 等 (2004) 风格: 外部档案 + 拥挤度选 leader + 速度更新 + 湍流变异.
与 MO-ALA 同为 "档案 + leader" 群智能范式, 作为公平对照基线.
预算单位: 函数评估次数 FEs.
"""

from __future__ import annotations

import numpy as np

from ..core.archive import ParetoArchive
from ..core.dominance import constraint_violation, dominates


def mopso(problem, n_agents=100, max_fes=10000, archive_size=200,
          w=0.4, c1=2.0, c2=2.0, mutation_rate=0.1,
          seed=None, monitor=None, verbose=False):
    """多目标 PSO. 返回 dict: X, F, archive, n_fes."""
    rng = np.random.default_rng(seed)
    dim, xl, xu = problem.n_var, problem.xl, problem.xu
    span = xu - xl

    X = xl + rng.random((n_agents, dim)) * span
    V = np.zeros((n_agents, dim))
    F, G = problem.evaluate(X)
    CV = constraint_violation(G)
    if CV is None:
        CV = np.zeros(n_agents)
    fes = n_agents

    pbest_X = X.copy()
    pbest_F = F.copy()
    pbest_CV = CV.copy()

    archive = ParetoArchive(max_size=archive_size)
    archive.add(X, F, CV)
    if monitor is not None:
        monitor(fes, archive)

    vmax = 0.5 * span

    while fes < max_fes:
        for i in range(n_agents):
            leader = archive.leader(rng)
            if leader is None:
                leader = pbest_X[i]
            r1 = rng.random(dim)
            r2 = rng.random(dim)
            V[i] = w * V[i] + c1 * r1 * (pbest_X[i] - X[i]) + c2 * r2 * (leader - X[i])
            V[i] = np.clip(V[i], -vmax, vmax)
            X[i] = X[i] + V[i]
            # 湍流变异 (随 FEs 衰减)
            rate = mutation_rate * (1.0 - fes / max_fes)
            if rng.random() < rate:
                d = rng.integers(dim)
                X[i, d] = xl[d] + rng.random() * span[d]
            # 边界: 裁剪并反弹速度
            below = X[i] < xl
            above = X[i] > xu
            X[i] = np.clip(X[i], xl, xu)
            V[i][below | above] *= -1.0

        Fnew, Gnew = problem.evaluate(X)
        CVnew = constraint_violation(Gnew)
        if CVnew is None:
            CVnew = np.zeros(n_agents)
        fes += n_agents
        F, CV = Fnew, CVnew

        # 更新 pbest
        for i in range(n_agents):
            if dominates(F[i], pbest_F[i], CV[i], pbest_CV[i]):
                pbest_X[i], pbest_F[i], pbest_CV[i] = X[i].copy(), F[i].copy(), CV[i]
            elif (not dominates(pbest_F[i], F[i], pbest_CV[i], CV[i])) and rng.random() < 0.5:
                pbest_X[i], pbest_F[i], pbest_CV[i] = X[i].copy(), F[i].copy(), CV[i]

        archive.add(X, F, CV)
        if monitor is not None:
            monitor(fes, archive)
        if verbose and (fes // n_agents) % 20 == 0:
            print(f"[MOPSO] FEs={fes}  |archive|={len(archive)}")

    return {"X": archive.X, "F": archive.F, "archive": archive, "n_fes": fes}
