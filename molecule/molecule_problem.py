"""MoleculeLatentProblem —— 在 SmilesVAE 隐空间上的多目标分子逆向设计问题.

决策变量: latent z ∈ R^d  (d = VAE latent_dim)
解码:     z -> VAE.generate -> SMILES
目标(最小化): [ −kcat_μ, σ, simple_sa, −QED ]
约束 g<=0:    [ 无效(1/0), 非Lipinski(1/0), PAINS(1/0) ]

无效 SMILES 给一组很差但有限的目标 + 触发"无效"约束, 由约束支配把搜索推向
有效、可信(低σ)、高活性、可合成、成药的分子区域.
"""

from __future__ import annotations

import numpy as np

from ..core.problem import Problem
from ._molgen import compute_props


class MoleculeLatentProblem(Problem):
    def __init__(self, decoder, kcat, z_bound=3.0, min_heavy=6):
        d = decoder.latent_dim
        super().__init__(n_var=d, n_obj=4, xl=-z_bound, xu=z_bound,
                         n_constr=4, name="mol_latent")
        self.decoder = decoder
        self.kcat = kcat
        self.min_heavy = int(min_heavy)
        self.obj_names = ["neg_kcat_mu", "sigma", "sa", "neg_qed"]
        self.constr_names = ["invalid", "not_lipinski", "pains", "too_small"]

    def _evaluate(self, Z):
        smiles = self.decoder.decode(Z)
        N = len(smiles)
        # 无效分子的兜底目标 (差但有限) + 无效约束
        F = np.column_stack([np.ones(N), np.full(N, 5.0), np.full(N, 10.0), np.zeros(N)])
        G = np.column_stack([np.ones(N), np.zeros(N), np.zeros(N), np.zeros(N)])
        valid = [i for i, s in enumerate(smiles) if s]
        if valid:
            vs = [smiles[i] for i in valid]
            mu, sigma = self.kcat.predict(vs)
            for k, i in enumerate(valid):
                props = compute_props(smiles[i])
                if props is None or np.isnan(mu[k]):
                    continue
                F[i] = [-float(mu[k]), float(sigma[k]), float(props.simple_sa), -float(props.qed)]
                G[i] = [0.0,
                        0.0 if props.lipinski_pass else 1.0,
                        0.0 if not props.pains else 1.0,
                        float(max(0, self.min_heavy - props.heavy_atoms))]
        return F, G

    def evaluate_oracle(self, X):
        """CU-MOBO oracle 接口: 返回 {mu, sigma, qed, sa}.

        mu=kcat 活性(↑), sigma=kcat 不确定性(↓), qed(↑), sa(↓).
        无效 SMILES 给惩罚值(低活性/高不确定/低成药/高合成成本), 让 BO 学会避开.
        """
        smiles = self.decoder.decode(X)
        N = len(smiles)
        mu = np.full(N, -1.0)
        sigma = np.full(N, 1.0)
        qed = np.zeros(N)
        sa = np.full(N, 10.0)
        valid = [i for i, s in enumerate(smiles) if s]
        if valid:
            vs = [smiles[i] for i in valid]
            m, sg = self.kcat.predict(vs)
            for k, i in enumerate(valid):
                props = compute_props(smiles[i])
                if props is None or np.isnan(m[k]):
                    continue
                # 医药化学硬过滤: 太小 / 非 Lipinski / PAINS -> 留惩罚值, 让 BO 避开
                if (props.heavy_atoms < self.min_heavy or not props.lipinski_pass
                        or props.pains):
                    continue
                mu[i] = float(m[k])
                sigma[i] = float(sg[k])
                qed[i] = float(props.qed)
                sa[i] = float(props.simple_sa)
        return {"mu": mu, "sigma": sigma, "qed": qed, "sa": sa}

    def report(self, X):
        """把一组 latent 解码成可读分子记录 (去重前). 用于导出 Pareto 策略表."""
        smiles = self.decoder.decode(X)
        rows = []
        valid = [i for i, s in enumerate(smiles) if s]
        if not valid:
            return rows
        vs = [smiles[i] for i in valid]
        mu, sigma = self.kcat.predict(vs)
        for k, i in enumerate(valid):
            props = compute_props(smiles[i])
            if props is None or np.isnan(mu[k]):
                continue
            rows.append({
                "smiles": smiles[i],
                "kcat_mu_log10": float(mu[k]),
                "kcat_sigma": float(sigma[k]),
                "qed": float(props.qed),
                "sa": float(props.simple_sa),
                "mw": float(props.mw),
                "logp": float(props.logp),
                "lipinski_pass": bool(props.lipinski_pass),
                "pains": bool(props.pains),
            })
        return rows
