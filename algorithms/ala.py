"""Artificial Lemming Algorithm (ALA) —— 单目标版, 忠实复现原论文/官方 MATLAB.

参考: Xiao, Cui, Abu Khurma, Castillo,
      "Artificial lemming algorithm: a novel bionic meta-heuristic technique
       for solving real-world engineering optimization problems",
      Artificial Intelligence Review, 2025. (官方 ALA.m, 2024-05-16)

四种行为 (公式编号沿用 CALA 复现稿 nhm-20-05-063):
    探索: 长距迁徙 (3.2) / 挖洞 (3.6)
    开发: 觅食 (3.8) / 躲避天敌 (3.11)
能量切换: theta = 2*atan(1 - FEs/MaxFEs); E = 2*ln(1/rand)*theta;
          E>1 走探索, 否则走开发.
预算单位: 函数评估次数 FEs (而非迭代数) —— 与"昂贵评估"代理框架天然对齐.
"""

from __future__ import annotations

import numpy as np

from ..core.operators import levy_flight


def ala(problem, n_agents=30, max_fes=10000, seed=None, verbose=False):
    """单目标 ALA. 返回 dict: x(最优解), f(最优值), curve(收敛曲线), n_fes."""
    rng = np.random.default_rng(seed)
    dim, xl, xu = problem.n_var, problem.xl, problem.xu

    X = xl + rng.random((n_agents, dim)) * (xu - xl)
    f = problem.evaluate(X)[0].ravel()  # (N,) —— 单目标
    fes = n_agents
    best_idx = int(np.argmin(f))
    pos = X[best_idx].copy()
    score = float(f[best_idx])
    curve = []
    vec_flag = np.array([1.0, -1.0])

    while fes < max_fes:
        RB = rng.standard_normal((n_agents, dim))           # Brownian motion
        F = vec_flag[rng.integers(0, 2)]                    # ±1 方向标志 (每轮一次)
        theta = 2.0 * np.arctan(1.0 - fes / max_fes)        # 时变参数
        Xnew = X.copy()

        for i in range(n_agents):
            E = 2.0 * np.log(1.0 / rng.random()) * theta
            if E > 1.0:                                     # === 探索 ===
                if rng.random() < 0.3:                      # 长距迁徙 (3.2)
                    r1 = 2.0 * rng.random(dim) - 1.0
                    a = rng.integers(n_agents)
                    Xnew[i] = pos + F * RB[i] * (
                        r1 * (pos - X[i]) + (1.0 - r1) * (X[i] - X[a])
                    )
                else:                                       # 挖洞 (3.6)
                    r2 = rng.random() * (1.0 + np.sin(0.5 * fes))
                    b = rng.integers(n_agents)
                    Xnew[i] = X[i] + F * r2 * (pos - X[b])
            else:                                           # === 开发 ===
                if rng.random() < 0.5:                      # 觅食 (3.8)
                    radius = np.sqrt(np.sum((pos - X[i]) ** 2))
                    r3 = rng.random()
                    spiral = radius * (np.sin(2 * np.pi * r3) + np.cos(2 * np.pi * r3))
                    Xnew[i] = pos + F * X[i] * spiral * rng.random()
                else:                                       # 躲避天敌 (3.11)
                    G = 2.0 * np.sign(rng.random() - 0.5) * (1.0 - fes / max_fes)
                    Xnew[i] = pos + F * G * levy_flight(dim, rng) * (pos - X[i])

        Xnew = np.clip(Xnew, xl, xu)
        fnew = problem.evaluate(Xnew)[0].ravel()
        for i in range(n_agents):
            fes += 1
            if fnew[i] < f[i]:                              # 贪婪接受
                X[i] = Xnew[i]
                f[i] = fnew[i]
            if f[i] < score:
                pos = X[i].copy()
                score = float(f[i])
            if fes >= max_fes:
                break

        curve.append(score)
        if verbose and len(curve) % 20 == 0:
            print(f"[ALA] FEs={fes}  best={score:.6g}")

    return {"x": pos, "f": score, "curve": np.asarray(curve), "n_fes": fes}
