"""Mode B —— 接真实(远程 AutoDL)KcatMoE/Boltz oracle 的异步人在环主动逆向设计闭环.

把 molgen_baseline 的"一次性生成 → 标量 rank_score 选 top-N"升级成:
    CU-MOBO 每轮按 qNEHVI 采集**最有信息量的一批**候选 → 解码 SMILES → 导出 AutoDL
    输入 → [远程跑真实 KcatMoE/Boltz] → 回填 (μ,σ) → 下一轮.

样本高效 + 校准不确定性感知 + Pareto(非 μ−λσ 标量).
"""

from .active_design import ActiveState, propose, ingest, seed_from_kcatmoe

__all__ = ["ActiveState", "propose", "ingest", "seed_from_kcatmoe"]
