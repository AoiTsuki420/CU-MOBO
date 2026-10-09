"""分子逆向设计集成: 在 molgen_baseline 的 SmilesVAE 隐空间上做多目标优化.

把 puridiff_optim 的 MO-ALA / SA-MO-ALA 与 SMILES 生成模型耦合:
    latent z --VAE--> SMILES --(kcat 代理 + RDKit)--> 多目标 + 约束 --> Pareto 分子
"""

from .vae_decoder import VaeDecoder
from .kcat_surrogate import KcatSurrogate
from .molecule_problem import MoleculeLatentProblem

__all__ = ["VaeDecoder", "KcatSurrogate", "MoleculeLatentProblem"]
