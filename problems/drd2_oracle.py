"""真实 oracle 版 oracle-gap 基准 (DRD2).

把合成 `UncertainOracleProblem` 升级成**真实、文献标准**的目标:
  - 隐藏 ground truth = TDC DRD2 oracle(Olivecrona 2017 的 SVM, PMO/REINVENT 标准打分器);
  - 优化器可见的"不确定 oracle" = 在**有限标注预算**下训的 Morgan-FP 随机森林代理
    (μ = RF 均值预测, σ = 树间方差 = epistemic 不确定性);
  - 决策变量 = drug-like SmilesVAE 隐空间 z ∈ R^d, 解码成 SMILES.

oracle-gap 来源(真实、非人造): RF 在优化器推入的**新颖**隐空间区域外推过自信 →
μ 虚高但真实 DRD2 低; σ(树方差)在这些区域升高 → CU-MOBO 用 σ(目标 + 异方差噪声)
下采信之, 抗 oracle-exploitation. **无循环论证**: 代理 ≠ 裁判, 裁判判的是新分子.

接口与 `UncertainOracleProblem` 一致 (evaluate_oracle / true_activity), 故复用
`run_oracle_gap` 的对照机制(2×2 消融 + qParEGO + NSGA-II).
"""

from __future__ import annotations

import numpy as np

from ..molecule._molgen import compute_props


class DRD2OracleProblem:
    def __init__(self, decoder, surrogate, drd2_oracle, z_bound=3.0, min_heavy=6,
                 enforce_druglike=True):
        self.decoder = decoder
        self.surrogate = surrogate          # KcatSurrogate, 训练在真实 DRD2 分数上
        self.oracle = drd2_oracle           # callable: list[SMILES] -> list[float] in [0,1]
        self.n_var = int(decoder.latent_dim)
        self.xl = np.full(self.n_var, -float(z_bound))
        self.xu = np.full(self.n_var, float(z_bound))
        self.min_heavy = int(min_heavy)
        self.enforce_druglike = bool(enforce_druglike)
        self.obj_names = ["mu", "sigma", "qed", "sa"]
        self._true_cache = {}               # SMILES -> 真实 DRD2 (oracle 调用有成本, 缓存)

    # -- 优化器可见: 有限预算 RF 代理 (带 epistemic σ) -------------------------
    def evaluate_oracle(self, Z):
        smiles = self.decoder.decode(Z)
        N = len(smiles)
        mu = np.full(N, -1.0)               # 无效/非成药的惩罚目标
        sigma = np.full(N, 1.0)
        qed = np.zeros(N)
        sa = np.full(N, 10.0)
        valid = [i for i, s in enumerate(smiles) if s]
        if valid:
            vs = [smiles[i] for i in valid]
            m, sg = self.surrogate.predict(vs)
            for k, i in enumerate(valid):
                props = compute_props(smiles[i])
                if props is None or np.isnan(m[k]):
                    continue
                if self.enforce_druglike and (props.heavy_atoms < self.min_heavy
                                              or not props.lipinski_pass or props.pains):
                    continue
                mu[i] = float(m[k])
                sigma[i] = float(sg[k])
                qed[i] = float(props.qed)
                sa[i] = float(props.simple_sa)
        return {"mu": mu, "sigma": sigma, "qed": qed, "sa": sa}

    # -- 隐藏 ground truth: 真实 DRD2 oracle (仅评估用) -----------------------
    def true_activity(self, Z):
        smiles = self.decoder.decode(Z)
        out = np.zeros(len(smiles))         # 无效 SMILES -> 0 活性
        todo, idx = [], []
        for i, s in enumerate(smiles):
            if not s:
                continue
            if s in self._true_cache:
                out[i] = self._true_cache[s]
            else:
                todo.append(s); idx.append(i)
        if todo:
            scores = self.oracle(todo)
            scores = [float(x) for x in (scores if isinstance(scores, (list, tuple, np.ndarray)) else [scores])]
            for s, sc, i in zip(todo, scores, idx):
                self._true_cache[s] = sc
                out[i] = sc
        return out

    def decode_report(self, Z):
        """把隐空间点解码成可读分子记录 (smiles + 代理 μ/σ + 真实 DRD2 + 成药属性)."""
        smiles = self.decoder.decode(Z)
        valid = [i for i, s in enumerate(smiles) if s]
        if not valid:
            return []
        vs = [smiles[i] for i in valid]
        m, sg = self.surrogate.predict(vs)
        true = self.true_activity(Z)
        rows = []
        for k, i in enumerate(valid):
            props = compute_props(smiles[i])
            if props is None or np.isnan(m[k]):
                continue
            rows.append({
                "smiles": smiles[i],
                "drd2_surrogate_mu": float(m[k]),
                "surrogate_sigma": float(sg[k]),
                "drd2_true": float(true[i]),
                "qed": float(props.qed),
                "sa": float(props.simple_sa),
                "mw": float(props.mw),
                "lipinski_pass": bool(props.lipinski_pass),
                "pains": bool(props.pains),
            })
        return rows
