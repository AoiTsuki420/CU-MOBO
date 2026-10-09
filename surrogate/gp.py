"""GP 代理: 每个目标一个独立 SingleTaskGP (BoTorch), 输入归一化 + 输出标准化.

predict(X) 返回后验均值 mean 和标准差 std, 形状均为 (n, n_obj).
统一按"最小化"使用: 直接对原始目标值建模, 由消费方决定方向.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import SumMarginalLogLikelihood

_DTYPE = torch.double


class GPSurrogate:
    def __init__(self, xl, xu):
        self.xl = np.asarray(xl, float)
        self.xu = np.asarray(xu, float)
        self.dim = len(self.xl)
        self._bounds = torch.tensor(
            np.vstack([self.xl, self.xu]), dtype=_DTYPE
        )
        self.model = None        # 目标 GP (ModelListGP)
        self.cmodel = None       # 约束 GP (ModelListGP), 无约束时为 None
        self.n_obj = None
        self.n_constr = 0

    def _fit_modellist(self, Xt, Y):
        models = []
        for m in range(Y.shape[1]):
            gp = SingleTaskGP(
                Xt, torch.as_tensor(Y[:, m: m + 1], dtype=_DTYPE),
                input_transform=Normalize(d=self.dim, bounds=self._bounds),
                outcome_transform=Standardize(m=1),
            )
            models.append(gp)
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(mll)
        return model

    def fit(self, X, F, G=None):
        X = np.atleast_2d(np.asarray(X, float))
        F = np.atleast_2d(np.asarray(F, float))
        self.n_obj = F.shape[1]
        Xt = torch.as_tensor(X, dtype=_DTYPE)
        self.model = self._fit_modellist(Xt, F)
        if G is not None:
            G = np.atleast_2d(np.asarray(G, float))
            if G.shape[1] > 0:
                self.n_constr = G.shape[1]
                self.cmodel = self._fit_modellist(Xt, G)
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        Xt = torch.as_tensor(X, dtype=_DTYPE)
        self.model.eval()
        with torch.no_grad():
            post = self.model.posterior(Xt)
            mean = post.mean.cpu().numpy().reshape(len(X), self.n_obj)
            var = post.variance.clamp_min(1e-12).cpu().numpy().reshape(len(X), self.n_obj)
        return mean, np.sqrt(var)

    def predict_constraints(self, X):
        """约束后验均值 (N, n_constr). 无约束模型时返回 None."""
        if self.cmodel is None:
            return None
        X = np.atleast_2d(np.asarray(X, float))
        Xt = torch.as_tensor(X, dtype=_DTYPE)
        self.cmodel.eval()
        with torch.no_grad():
            post = self.cmodel.posterior(Xt)
            cmean = post.mean.cpu().numpy().reshape(len(X), self.n_constr)
        return cmean
