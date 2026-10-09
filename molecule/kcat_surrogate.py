"""本地 kcat 代理 (demo proxy of KcatMoE).

从 molgen_baseline 已有的预测表 (smiles, pred_kcat_log10=μ, sigma_kcat_log10=σ)
训练 Morgan 指纹 -> (μ, σ) 的随机森林. 用作隐空间优化的**在环**活性 + 不确定性目标.

⚠️ 这是真实 KcatMoE oracle 的**本地代理**, 仅供本地全流程 demo. 真实研究闭环
(Mode B) 必须把候选批次送回真实 KcatMoE, 且 true-hit 用独立 oracle (见 molgen
项目 PAPER_PLAN §3). 优化 μ 又拿同源信号评估存在循环论证, 见 README 说明.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.ensemble import RandomForestRegressor


class KcatSurrogate:
    def __init__(self, n_bits=2048, radius=2, n_estimators=300, seed=0):
        self.n_bits = n_bits
        self._gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        self.rf_mu = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
        self.rf_sigma = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
        self.fitted = False

    def _fp(self, smiles):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        fp = self._gen.GetFingerprint(mol)
        arr = np.zeros((self.n_bits,), dtype=np.float32)
        ConvertToNumpyArray(fp, arr)
        return arr

    def fit(self, smiles, mu, sigma=None):
        """σ 三种模式:
          sigma=<array>     -> 单独 RF 建模给定的 per-molecule σ;
          sigma=None        -> σ 取 RF 树间方差(epistemic 距离), 只有标签、无不确定性时用;
          sigma="calibrated"-> **误差校准 σ**: 用 out-of-fold 残差 |y−ŷ| 训 σ 模型, 使 σ
                               追踪**预测误差**(而非单纯离训练数据的距离)。这满足 CU-MOBO
                               "σ 须真校准"的前提: 自信且正确处 σ 低(含拟合好的稀有 active),
                               预测错处 σ 高(假阳性 trap) → σ-目标/噪声不会误避真 active."""
        calibrated = isinstance(sigma, str) and sigma == "calibrated"
        X, ymu, ysig = [], [], []
        sigma = None if calibrated else (np.asarray(sigma, float) if sigma is not None else None)
        if sigma is not None:
            sig_fill = np.nanmean(sigma) if np.any(~np.isnan(sigma)) else 0.1
        for i, (s, m) in enumerate(zip(smiles, np.asarray(mu, float))):
            if np.isnan(m):
                continue
            fp = self._fp(s)
            if fp is None:
                continue
            X.append(fp)
            ymu.append(float(m))
            if sigma is not None:
                sg = sigma[i]
                ysig.append(float(sg) if not np.isnan(sg) else float(sig_fill))
        if not X:
            raise ValueError("KcatSurrogate.fit: no valid (smiles, mu) training rows")
        X = np.asarray(X); ymu = np.asarray(ymu)
        self.rf_mu.fit(X, ymu)
        if calibrated:
            # out-of-fold 残差 -> σ 模型 (σ 追踪预测误差, 非 epistemic 距离)
            from sklearn.base import clone
            from sklearn.model_selection import cross_val_predict
            cv = int(min(5, max(2, len(X) // 50)))
            oof = cross_val_predict(clone(self.rf_mu), X, ymu, cv=cv)
            resid = np.abs(oof - ymu)
            self.rf_sigma.fit(X, resid)
            self.sigma_mode = "model"
        elif sigma is not None:
            self.rf_sigma.fit(X, np.asarray(ysig))
            self.sigma_mode = "model"
        else:
            self.sigma_mode = "tree_var"
        self.fitted = True
        self.n_train = len(X)
        return self

    def predict(self, smiles):
        """-> (mu, sigma), 无效 SMILES 处为 NaN."""
        n = len(smiles)
        mu = np.full(n, np.nan)
        sigma = np.full(n, np.nan)
        idx, X = [], []
        for i, s in enumerate(smiles):
            fp = self._fp(s)
            if fp is not None:
                idx.append(i)
                X.append(fp)
        if X:
            X = np.asarray(X)
            mu[idx] = self.rf_mu.predict(X)
            if getattr(self, "sigma_mode", "model") == "model":
                sigma[idx] = self.rf_sigma.predict(X)
            else:  # 树间方差作 epistemic σ
                preds = np.stack([est.predict(X) for est in self.rf_mu.estimators_])
                sigma[idx] = preds.std(axis=0)
        return mu, sigma
