"""Mode B CLI —— 真实 oracle 异步人在环主动逆向设计.

子命令:
  init     用已有真实 KcatMoE 预测表 seed 初始状态
  propose  CU-MOBO 采集一批候选 -> 解码 SMILES -> 导出 AutoDL/Boltz 输入(供远程跑)
  ingest   读回远程跑完的真实预测 -> 回填 -> 准备下一轮

典型循环:
  python -m puridiff_optim.mode_b.run_mode_b init     --state runs_modeb/state.csv
  python -m puridiff_optim.mode_b.run_mode_b propose  --state runs_modeb/state.csv --q 12 --out-dir runs_modeb/round1
  # 把 runs_modeb/round1/*_kcatmoe_autodl.csv 上传 AutoDL 跑 KcatMoE, 下载预测
  python -m puridiff_optim.mode_b.run_mode_b ingest   --state runs_modeb/state.csv --predictions <下载的预测.csv>
  python -m puridiff_optim.mode_b.run_mode_b propose  --state runs_modeb/state.csv --q 12 --out-dir runs_modeb/round2
  ...
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import warnings

import pandas as pd

from ..molecule import VaeDecoder
from ..molecule._molgen import molgen_src_path
from .active_design import ActiveState, ingest, propose, seed_from_kcatmoe

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(__file__)
_ASSETS = os.path.join(_HERE, "..", "molecule", "assets")
DEFAULT_VAE = os.path.abspath(os.path.join(_ASSETS, "vae16", "vae.pt"))
DEFAULT_PREDICTIONS = r"E:\fullproject\project\LKM-SMA-NN\predictions_kcat.csv"  # 400 真实 KinEAGER 分子
DEFAULT_TARGET = r"E:\fullproject\笔记\boltz_run\molgen_baseline\data\targets\kcatmoe_real_target.yaml"


def cmd_init(args):
    dec = VaeDecoder(args.vae)
    state = seed_from_kcatmoe(args.predictions, dec)
    os.makedirs(os.path.dirname(os.path.abspath(args.state)), exist_ok=True)
    state.save(args.state)
    print(f"[init] seeded {len(state.evaluated())} 个真实评估分子 -> {args.state}")


def cmd_propose(args):
    dec = VaeDecoder(args.vae)
    state = ActiveState.load(args.state)
    batch = propose(state, dec, q=args.q, seed=args.seed)
    if len(batch) == 0:
        print("[propose] 无有效候选(都无效/不合规),换 seed 重试")
        return
    rnd = int(batch["round"].iloc[0])
    state.df = pd.concat([state.df, batch], ignore_index=True)
    state.save(args.state)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    mol_csv = os.path.join(out_dir, f"round{rnd}_molecules.csv")
    batch[["candidate_id", "smiles", "qed", "mw", "logp"]].to_csv(mol_csv, index=False)

    autodl_csv = os.path.join(out_dir, f"round{rnd}_kcatmoe_autodl.csv")
    script = os.path.join(molgen_src_path(), "make_kcatmoe_inputs.py")
    try:
        subprocess.run([sys.executable, script, "--molecules", mol_csv,
                        "--protein-yaml", args.target_yaml, "--output", autodl_csv,
                        "--top-n", str(len(batch)), "--schema", "autodl"],
                       check=True, capture_output=True, text=True)
        print(f"[propose] AutoDL KcatMoE 输入就绪: {autodl_csv}")
    except Exception as e:
        print(f"[propose] make_kcatmoe_inputs 失败({e}); 批次 SMILES 在 {mol_csv}")

    print(f"[propose] 第 {rnd} 轮提议 {len(batch)} 个候选 (CU-MOBO qNEHVI 采集):")
    print(batch[["candidate_id", "smiles", "qed", "sa"]].to_string(index=False))
    print(f"\n→ 把 {os.path.basename(autodl_csv)} 上传 AutoDL 跑 KcatMoE(+可选 Boltz),"
          f"下载预测后:\n   python -m puridiff_optim.mode_b.run_mode_b ingest "
          f"--state {args.state} --predictions <预测.csv>")


def cmd_ingest(args):
    state = ActiveState.load(args.state)
    state, n = ingest(state, args.predictions)
    state.save(args.state)
    print(f"[ingest] 回填 {n} 条真实预测; 状态现有 {len(state.evaluated())} 个已评估分子")
    print("→ 继续下一轮: propose")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Mode B 真实 oracle 主动逆向设计")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init")
    pi.add_argument("--state", required=True)
    pi.add_argument("--vae", default=DEFAULT_VAE)
    pi.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    pi.set_defaults(func=cmd_init)

    pp = sub.add_parser("propose")
    pp.add_argument("--state", required=True)
    pp.add_argument("--out-dir", required=True)
    pp.add_argument("--q", type=int, default=12)
    pp.add_argument("--seed", type=int, default=0)
    pp.add_argument("--vae", default=DEFAULT_VAE)
    pp.add_argument("--target-yaml", default=DEFAULT_TARGET)
    pp.set_defaults(func=cmd_propose)

    pg = sub.add_parser("ingest")
    pg.add_argument("--state", required=True)
    pg.add_argument("--predictions", required=True)
    pg.set_defaults(func=cmd_ingest)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
