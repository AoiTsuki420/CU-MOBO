"""Mode B 状态机 + propose/ingest 驱动.

状态 = 所有曾被真实 oracle 评估过的分子(latent z + SMILES + 真实 μ,σ + qed,sa)
       以及当前轮 proposed 但还没拿到真实预测的分子.
CU-MOBO 在已评估集上拟合 GP, 每轮 propose 一批送真实 oracle, ingest 回填.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..algorithms.cu_mobo import cu_mobo_propose
from ..molecule._molgen import compute_props

_COLS = ["candidate_id", "round", "status", "smiles", "z",
         "mu", "sigma", "qed", "sa", "mw", "logp"]


class ActiveState:
    def __init__(self, df=None):
        self.df = df.copy() if df is not None else pd.DataFrame(columns=_COLS)

    @classmethod
    def load(cls, path):
        return cls(pd.read_csv(path))

    def save(self, path):
        self.df.to_csv(path, index=False)

    def evaluated(self):
        return self.df[self.df["status"] == "evaluated"]

    def proposed(self):
        return self.df[self.df["status"] == "proposed"]

    def training_arrays(self):
        ev = self.evaluated()
        Z = np.array([json.loads(z) for z in ev["z"]], dtype=float)
        return (Z, ev["mu"].astype(float).values, ev["sigma"].astype(float).values,
                ev["qed"].astype(float).values, ev["sa"].astype(float).values)

    def next_round(self):
        if len(self.df) == 0:
            return 1
        return int(pd.to_numeric(self.df["round"], errors="coerce").max()) + 1

    def known_smiles(self):
        return set(self.df["smiles"].astype(str))


# 真实 oracle 列名兼容: KcatMoE(pred_kcat_log10/sigma_kcat_log10) 与
# KinEAGER(pred_kcat_log/unc_kcat_log) 都能识别. log 列优先于未取对数的 pred_kcat.
_MU_ALIASES = ["pred_kcat_log10", "pred_kcat_log", "y_pred_kcat_log10", "kcat_pred_log10",
               "mu_kcat", "mu", "pred_log10_kcat"]
_SIG_ALIASES = ["sigma_kcat_log10", "unc_kcat_log", "kcat_sigma_log10", "sigma_kcat",
                "sigma", "uncertainty", "unc_kcat", "pred_sigma"]
_SMILES_ALIASES = ["smiles", "canonical_smiles", "smi"]


def _find_col(df, aliases):
    low = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in low:
            return low[a.lower()]
    return None


# ----------------------------------------------------------------- seed ------
def seed_from_kcatmoe(predictions_csv, decoder):
    """用真实 oracle 预测表 seed 初始状态(自动识别 μ/σ 列, 全部标记 evaluated)."""
    df = pd.read_csv(predictions_csv)
    smi_col = _find_col(df, _SMILES_ALIASES)
    mu_col = _find_col(df, _MU_ALIASES)
    sig_col = _find_col(df, _SIG_ALIASES)
    if smi_col is None or mu_col is None:
        raise ValueError(f"预测表需含 smiles 和 μ 列(如 pred_kcat_log); 见 {list(df.columns)}")
    df = df.dropna(subset=[smi_col, mu_col]).drop_duplicates(smi_col).reset_index(drop=True)
    Z = decoder.encode(df[smi_col].astype(str).tolist())
    rows = []
    for i, row in df.iterrows():
        s = str(row[smi_col])
        props = compute_props(s)
        if props is None:
            continue
        rows.append({
            "candidate_id": f"seed_{i:04d}", "round": 0, "status": "evaluated",
            "smiles": s, "z": json.dumps([float(v) for v in Z[i]]),
            "mu": float(row[mu_col]),
            "sigma": float(row[sig_col]) if sig_col else np.nan,
            "qed": float(props.qed), "sa": float(props.simple_sa),
            "mw": float(props.mw), "logp": float(props.logp),
        })
    return ActiveState(pd.DataFrame(rows, columns=_COLS))


# -------------------------------------------------------------- propose ------
def propose(state, decoder, q=10, z_bound=3.0, min_heavy=6, seed=0):
    """CU-MOBO 单步采集 -> 解码 -> 过滤 -> 返回 q 条 proposed 分子(DataFrame)."""
    Z, mu, sigma, qed, sa = state.training_arrays()
    dim = Z.shape[1]
    xl, xu = -z_bound * np.ones(dim), z_bound * np.ones(dim)
    # 多提一些以补足有效/合规过滤
    Xq = cu_mobo_propose(Z, mu, sigma, qed, sa, xl, xu, q=int(q * 2) + 4, seed=seed)
    smiles = decoder.decode(Xq)

    rnd = state.next_round()
    seen = state.known_smiles()
    rows = []
    for z, s in zip(Xq, smiles):
        if not s or s in seen:
            continue
        props = compute_props(s)
        if props is None or props.heavy_atoms < min_heavy or not props.lipinski_pass or props.pains:
            continue
        seen.add(s)
        rows.append({
            "candidate_id": f"r{rnd}_{len(rows):03d}", "round": rnd, "status": "proposed",
            "smiles": s, "z": json.dumps([float(v) for v in z]),
            "mu": "", "sigma": "", "qed": float(props.qed), "sa": float(props.simple_sa),
            "mw": float(props.mw), "logp": float(props.logp),
        })
        if len(rows) >= q:
            break
    return pd.DataFrame(rows, columns=_COLS)


# --------------------------------------------------------------- ingest ------
def ingest(state, predictions_csv):
    """读回真实 KcatMoE 预测, 填 μ,σ 并标记 evaluated.

    优先按 candidate_id 匹配; 不中则按**规范化 SMILES** 匹配
    (因为 make_kcatmoe_inputs 会把 candidate_id 重排成 gen_NNN).
    """
    from ..molecule._molgen import canonicalize_smiles

    pred = pd.read_csv(predictions_csv)
    mu_col = _find_col(pred, _MU_ALIASES)
    sig_col = _find_col(pred, _SIG_ALIASES)
    if mu_col is None:
        raise ValueError(f"预测表缺 μ 列; 见 {list(pred.columns)}")
    mu_vals = pd.to_numeric(pred[mu_col], errors="coerce")
    sig_vals = pd.to_numeric(pred[sig_col], errors="coerce") if sig_col else None

    mu_by_id, sig_by_id, mu_by_smi, sig_by_smi = {}, {}, {}, {}
    for j in range(len(pred)):
        m = mu_vals.iloc[j]
        s = sig_vals.iloc[j] if sig_vals is not None else np.nan
        if "candidate_id" in pred.columns:
            mu_by_id[str(pred["candidate_id"].iloc[j])] = m
            sig_by_id[str(pred["candidate_id"].iloc[j])] = s
        if "smiles" in pred.columns:
            cs = canonicalize_smiles(str(pred["smiles"].iloc[j]))
            if cs:
                mu_by_smi[cs] = m
                sig_by_smi[cs] = s

    df = state.df.copy()
    n = 0
    for idx, row in df[df["status"] == "proposed"].iterrows():
        cid, m, s = str(row["candidate_id"]), None, np.nan
        if cid in mu_by_id and not pd.isna(mu_by_id[cid]):
            m, s = mu_by_id[cid], sig_by_id.get(cid, np.nan)
        else:
            cs = canonicalize_smiles(str(row["smiles"]))
            if cs in mu_by_smi and not pd.isna(mu_by_smi[cs]):
                m, s = mu_by_smi[cs], sig_by_smi.get(cs, np.nan)
        if m is not None:
            df.at[idx, "mu"] = float(m)
            df.at[idx, "sigma"] = float(s) if not pd.isna(s) else np.nan
            df.at[idx, "status"] = "evaluated"
            n += 1
    return ActiveState(df), n
