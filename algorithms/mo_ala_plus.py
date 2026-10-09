"""MO-ALA+ —— 增强版多目标 ALA (M1.x 改进型贡献).

针对 base MO-ALA 在 M1 暴露的两个弱点:
    - ZDT2 偶发**早熟坍缩** (种群多样性丧失);
    - DTLZ2 (3 目标) 前沿分布/收敛偏弱.

加入两个增强策略 (保留 ALA 能量切换/四行为/Brownian/Lévy 不变):
    1. 拉丁超立方初始化 + 种群多样性监测;
    2. 停滞/坍缩触发**反向学习(OBL)再注入重启**, 恢复种群多样性;
    3. 可选 **ε-支配档案** (改善多目标分布).

base 版 (mo_ala) 仍作为对照基线保持忠实不变.
"""

from __future__ import annotations

import numpy as np

from ..core.archive import ParetoArchive
from ..core.dominance import constraint_violation, dominates
from ..core.epsilon_archive import EpsilonArchive
from ..core.operators import levy_flight
from ..core.sampling import latin_hypercube


def _pop_diversity(X, xl, xu):
    """归一化种群多样性 = 各维位置标准差 / 取值范围, 再取均值."""
    return float(np.mean(np.std(X, axis=0) / (xu - xl + 1e-12)))


def mo_ala_plus(problem, n_agents=100, max_fes=10000, archive_size=200,
                archive_kind="crowding", div_tol=0.03, restart_frac=0.5,
                restart_cooldown=5, seed=None, monitor=None, verbose=False,
                init_X=None):
    """返回 dict: X, F, archive, n_fes, n_restarts.

    init_X: 可选 (m, n_var) 初始种群 (warm-start, 如分子隐空间从真实分子编码而来);
            不足 n_agents 时用 LHS 补齐.
    """
    rng = np.random.default_rng(seed)
    dim, xl, xu = problem.n_var, problem.xl, problem.xu

    # 1. 初始化 (LHS, 或 warm-start init_X)
    if init_X is not None:
        init_X = np.atleast_2d(np.asarray(init_X, float))
        X = np.empty((n_agents, dim))
        m = min(len(init_X), n_agents)
        X[:m] = np.clip(init_X[:m], xl, xu)
        if m < n_agents:
            X[m:] = latin_hypercube(n_agents - m, xl, xu, seed=seed)
    else:
        X = latin_hypercube(n_agents, xl, xu, seed=seed)
    F, G = problem.evaluate(X)
    CV = constraint_violation(G)
    if CV is None:
        CV = np.zeros(n_agents)
    fes = n_agents

    if archive_kind == "epsilon":
        archive = EpsilonArchive(target_size=archive_size)
    else:
        archive = ParetoArchive(max_size=archive_size)
    archive.add(X, F, CV)
    if monitor is not None:
        monitor(fes, archive)

    vec_flag = np.array([1.0, -1.0])
    last_restart = 0
    gen = 0
    n_restarts = 0

    while fes < max_fes:
        RB = rng.standard_normal((n_agents, dim))
        Fflag = vec_flag[rng.integers(0, 2)]
        theta = 2.0 * np.arctan(1.0 - fes / max_fes)
        Xnew = X.copy()

        for i in range(n_agents):
            leader = archive.leader(rng)
            if leader is None:
                leader = X[rng.integers(n_agents)]
            E = 2.0 * np.log(1.0 / rng.random()) * theta
            if E > 1.0:
                if rng.random() < 0.3:
                    r1 = 2.0 * rng.random(dim) - 1.0
                    a = rng.integers(n_agents)
                    Xnew[i] = leader + Fflag * RB[i] * (
                        r1 * (leader - X[i]) + (1.0 - r1) * (X[i] - X[a]))
                else:
                    r2 = rng.random() * (1.0 + np.sin(0.5 * fes))
                    b = rng.integers(n_agents)
                    Xnew[i] = X[i] + Fflag * r2 * (leader - X[b])
            else:
                if rng.random() < 0.5:
                    radius = np.sqrt(np.sum((leader - X[i]) ** 2))
                    r3 = rng.random()
                    spiral = radius * (np.sin(2 * np.pi * r3) + np.cos(2 * np.pi * r3))
                    Xnew[i] = leader + Fflag * X[i] * spiral * rng.random()
                else:
                    Gc = 2.0 * np.sign(rng.random() - 0.5) * (1.0 - fes / max_fes)
                    Xnew[i] = leader + Fflag * Gc * levy_flight(dim, rng) * (leader - X[i])

        Xnew = np.clip(Xnew, xl, xu)
        Fnew, Gnew = problem.evaluate(Xnew)
        CVnew = constraint_violation(Gnew)
        if CVnew is None:
            CVnew = np.zeros(n_agents)
        fes += n_agents

        for i in range(n_agents):
            if dominates(Fnew[i], F[i], CVnew[i], CV[i]):
                X[i], F[i], CV[i] = Xnew[i], Fnew[i], CVnew[i]
            elif dominates(F[i], Fnew[i], CV[i], CVnew[i]):
                pass
            elif rng.random() < 0.5:
                X[i], F[i], CV[i] = Xnew[i], Fnew[i], CVnew[i]

        archive.add(Xnew, Fnew, CVnew)
        gen += 1

        # --- 多样性监测 + OBL 再注入重启 ---
        diversity = _pop_diversity(X, xl, xu)
        collapsed = diversity < div_tol or len(archive) < max(3, int(0.05 * n_agents))
        if collapsed and (gen - last_restart) >= restart_cooldown and fes < max_fes:
            k = max(1, int(restart_frac * n_agents))
            n_lhs = k // 2
            n_obl = k - n_lhs
            inj = [latin_hypercube(n_lhs, xl, xu, seed=int(rng.integers(1_000_000_000)))]
            # 反向学习: 围绕档案 leader 取反向点 + 噪声
            obl = []
            for _ in range(n_obl):
                lead = archive.leader(rng)
                if lead is None:
                    lead = X[rng.integers(n_agents)]
                x_opp = xl + xu - lead + 0.05 * (xu - xl) * rng.standard_normal(dim)
                obl.append(np.clip(x_opp, xl, xu))
            inj.append(np.array(obl))
            Xinj = np.vstack(inj)
            Finj, Ginj = problem.evaluate(Xinj)
            CVinj = constraint_violation(Ginj)
            if CVinj is None:
                CVinj = np.zeros(len(Xinj))
            fes += len(Xinj)
            # 替换种群中随机 k 个个体
            repl = rng.choice(n_agents, size=len(Xinj), replace=False)
            X[repl], F[repl], CV[repl] = Xinj, Finj, CVinj
            archive.add(Xinj, Finj, CVinj)
            last_restart = gen
            n_restarts += 1
            if verbose:
                print(f"[MO-ALA+] restart@gen{gen} fes={fes} div={diversity:.4f} "
                      f"|archive|={len(archive)}")

        if monitor is not None:
            monitor(fes, archive)

    return {"X": archive.X, "F": archive.F, "archive": archive,
            "n_fes": fes, "n_restarts": n_restarts}
