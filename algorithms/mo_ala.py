"""MO-ALA —— 多目标 Artificial Lemming Algorithm (本项目创新点).

据现有文献, ALA 尚无多目标版本. 本实现把单目标 ALA 扩展为多目标:

    1. 全局最优 Position  ->  每个个体从**外部 Pareto 档案**按拥挤度挑选 leader;
    2. 标量 `f1 < f2` 贪婪比较  ->  Deb 约束支配比较;
    3. 新增有界外部档案 (非支配 + 拥挤度截断) 作为输出前沿;
    4. 完整保留 ALA 的能量切换 theta/E、F 方向标志、四行为、Brownian、Lévy.

与 MOPSO 同属 "档案 + leader 引导" 的群智能多目标范式, 便于公平对照.
预算单位仍是函数评估次数 FEs.
"""

from __future__ import annotations

import numpy as np

from ..core.archive import ParetoArchive
from ..core.dominance import constraint_violation, dominates
from ..core.operators import levy_flight


def mo_ala(problem, n_agents=100, max_fes=10000, archive_size=200,
           seed=None, monitor=None, verbose=False):
    """多目标 ALA.

    monitor: 可选回调 monitor(fes, archive), 每代结束调用一次 (用于记录 HV 曲线).
    返回 dict: X, F (前沿), archive, n_fes.
    """
    rng = np.random.default_rng(seed)
    dim, xl, xu = problem.n_var, problem.xl, problem.xu

    X = xl + rng.random((n_agents, dim)) * (xu - xl)
    F, G = problem.evaluate(X)
    CV = constraint_violation(G)
    if CV is None:
        CV = np.zeros(n_agents)
    fes = n_agents

    archive = ParetoArchive(max_size=archive_size)
    archive.add(X, F, CV)
    if monitor is not None:
        monitor(fes, archive)

    vec_flag = np.array([1.0, -1.0])

    while fes < max_fes:
        RB = rng.standard_normal((n_agents, dim))
        Fflag = vec_flag[rng.integers(0, 2)]
        theta = 2.0 * np.arctan(1.0 - fes / max_fes)
        Xnew = X.copy()

        for i in range(n_agents):
            # leader 取代单目标里的全局 Position
            leader = archive.leader(rng)
            if leader is None:
                leader = X[rng.integers(n_agents)]

            E = 2.0 * np.log(1.0 / rng.random()) * theta
            if E > 1.0:                                     # === 探索 ===
                if rng.random() < 0.3:                      # 长距迁徙 (3.2)
                    r1 = 2.0 * rng.random(dim) - 1.0
                    a = rng.integers(n_agents)
                    Xnew[i] = leader + Fflag * RB[i] * (
                        r1 * (leader - X[i]) + (1.0 - r1) * (X[i] - X[a])
                    )
                else:                                       # 挖洞 (3.6)
                    r2 = rng.random() * (1.0 + np.sin(0.5 * fes))
                    b = rng.integers(n_agents)
                    Xnew[i] = X[i] + Fflag * r2 * (leader - X[b])
            else:                                           # === 开发 ===
                if rng.random() < 0.5:                      # 觅食 (3.8)
                    radius = np.sqrt(np.sum((leader - X[i]) ** 2))
                    r3 = rng.random()
                    spiral = radius * (np.sin(2 * np.pi * r3) + np.cos(2 * np.pi * r3))
                    Xnew[i] = leader + Fflag * X[i] * spiral * rng.random()
                else:                                       # 躲避天敌 (3.11)
                    Gc = 2.0 * np.sign(rng.random() - 0.5) * (1.0 - fes / max_fes)
                    Xnew[i] = leader + Fflag * Gc * levy_flight(dim, rng) * (leader - X[i])

        Xnew = np.clip(Xnew, xl, xu)
        Fnew, Gnew = problem.evaluate(Xnew)
        CVnew = constraint_violation(Gnew)
        if CVnew is None:
            CVnew = np.zeros(n_agents)
        fes += n_agents

        # 贪婪接受: 新支配旧->替换; 旧支配新->保留; 互不支配->以 0.5 概率替换
        for i in range(n_agents):
            if dominates(Fnew[i], F[i], CVnew[i], CV[i]):
                X[i], F[i], CV[i] = Xnew[i], Fnew[i], CVnew[i]
            elif dominates(F[i], Fnew[i], CV[i], CVnew[i]):
                pass
            elif rng.random() < 0.5:
                X[i], F[i], CV[i] = Xnew[i], Fnew[i], CVnew[i]

        archive.add(Xnew, Fnew, CVnew)
        if monitor is not None:
            monitor(fes, archive)
        if verbose and (fes // n_agents) % 20 == 0:
            print(f"[MO-ALA] FEs={fes}  |archive|={len(archive)}")

    return {"X": archive.X, "F": archive.F, "archive": archive, "n_fes": fes}
