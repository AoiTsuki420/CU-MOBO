"""D4: 流匹配 + 梯度引导采样(LaMBO-2 / 分类器引导思路)的最小实例.

承接 D3: 用 D3 的属性感知 VAE(隐空间已按活性组织)+ 其属性头 g(z)=logit(active|z).
在该隐空间上训练一个**整流流(rectified flow)速度场** v(z,t)(复用 molgen VelocityField),
然后做**引导采样**: 沿 ODE 积分时叠加 g 的梯度
    z <- z + dt·( v(z,t) + s·∇_z g(z) )
即"用生成式先验把样本拉回数据流形(高有效率/多样), 同时用属性梯度往高活性爬"。

诚实三方对照(本项目一贯主题: off-manifold 会被 reward-hack):
  unguided  : 纯流采样(s=0)      → 在流形上, 但无活性偏好(基线活性);
  guided    : 流 + 属性梯度引导     → 仍在流形上, 且活性↑(引导有效);
  grad-only : 去掉流、只在 z 上对 g 做梯度上升 → g 预测极高, 但**脱离流形** →
              解码有效率/真实 DRD2 活性塌掉(= 隐空间里的 reward-hacking, 呼应 oracle-gap)。

独立评测信号 = **真实 DRD2 oracle**(非引导用的属性头, 避免自证), 外加有效率/唯一数.

用法:
    python -m puridiff_optim.molecule.flow_guided_d4 \
        --d3-ckpt puridiff_optim/molecule/assets/vae_drd2_d3/vae.pt \
        --corpus tdc_data/drd2_corpus_hard.csv --fm-epochs 60 \
        --n-sample 96 --steps 100 --guide-scale 0.6
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import _molgen  # noqa: F401  触发把 molgen src 加入 sys.path
from latent_flow_matching import VelocityField  # noqa: E402  (molgen)
from ._molgen import canonicalize_smiles  # noqa: E402
from .train_latent_aware import PropHead  # noqa: E402
from .vae_decoder import VaeDecoder  # noqa: E402


def _load_d3(ckpt, device="cpu"):
    dec = VaeDecoder(ckpt, device=device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    if "prop_head_state" not in ck:
        raise SystemExit("ckpt 无 prop_head_state — 请用 train_latent_aware.py(D3)产出的 VAE")
    head = PropHead(dec.latent_dim)
    head.load_state_dict(ck["prop_head_state"]); head.eval()
    return dec, head


def _train_flow(field, Z, epochs, batch, lr, seed):
    """整流流: z0~N(0,I), z1=数据隐, target=z1-z0, MSE(v, target)."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(field.parameters(), lr=lr, weight_decay=1e-3)
    Zt = torch.as_tensor(Z, dtype=torch.float32)
    n = Zt.size(0)
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n); tot = 0.0; nb = 0
        for i in range(0, n, batch):
            z1 = Zt[perm[i:i + batch]]
            cond = torch.zeros(z1.size(0), dtype=torch.long)
            z0 = torch.randn_like(z1)
            t = torch.rand(z1.size(0), 1)
            zt = (1 - t) * z0 + t * z1
            v = field(zt, t, cond)
            loss = F.mse_loss(v, z1 - z0)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        if ep % 10 == 0 or ep == 1:
            print(f"[d4-fm] ep {ep:03d} mse={tot/nb:.4f}")


def _flow_sample(field, head, n, d, steps, guide_scale, seed):
    """引导 Euler 采样: z += dt·(v(z,t) + s·∇_z g(z)). guide_scale=0 即纯流采样."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, d, generator=g)
    cond = torch.zeros(n, dtype=torch.long)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n, 1), i * dt)
        with torch.no_grad():
            v = field(z, t, cond)
        if guide_scale != 0.0:
            zc = z.detach().requires_grad_(True)
            grad = torch.autograd.grad(head(zc).sum(), zc)[0]
            v = v + guide_scale * grad
        z = (z + dt * v).detach()
    return z.numpy()


def _grad_only(head, n, d, steps, lr, seed):
    """去掉流, 只在 z 上对属性头做梯度上升(故意 off-manifold 对照)."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, d, generator=g)
    for _ in range(steps):
        zc = z.detach().requires_grad_(True)
        grad = torch.autograd.grad(head(zc).sum(), zc)[0]
        z = (z + lr * grad).detach()
    return z.numpy()


def _eval_batch(label, Z, dec, head, oracle):
    smis = dec.decode(Z)                         # canonical SMILES 或 None
    valid = [s for s in smis if s]
    uniq = sorted(set(valid))
    with torch.no_grad():
        ph = torch.sigmoid(head(torch.as_tensor(Z, dtype=torch.float32))).numpy()
    drd2 = oracle(uniq) if uniq else []
    drd2 = np.asarray(drd2, float)
    return {
        "method": label,
        "validity": len(valid) / len(smis),
        "unique": len(uniq),
        "head_prob": float(ph.mean()),            # 引导头自评(应随引导上升)
        "mean_drd2": float(drd2.mean()) if len(drd2) else 0.0,   # 独立真值
        "frac_active": float(np.mean(drd2 > 0.5)) if len(drd2) else 0.0,
        "max_drd2": float(drd2.max()) if len(drd2) else 0.0,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--d3-ckpt", default="puridiff_optim/molecule/assets/vae_drd2_d3/vae.pt")
    ap.add_argument("--corpus", default="tdc_data/drd2_corpus_hard.csv")
    ap.add_argument("--fm-hidden", type=int, default=128)
    ap.add_argument("--fm-layers", type=int, default=4)
    ap.add_argument("--fm-epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-sample", type=int, default=96)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--guide-scale", type=float, default=0.6)
    ap.add_argument("--ga-lr", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-oracle", action="store_true", help="跳过真实 DRD2(仅有效率/唯一数)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)
    out_dir = os.path.abspath(args.out); os.makedirs(out_dir, exist_ok=True)

    dec, head = _load_d3(args.d3_ckpt)
    d = dec.latent_dim
    df = pd.read_csv(args.corpus)
    smiles = [c for c in (canonicalize_smiles(str(s)) for s in df["smiles"]) if c]
    Z = dec.encode(smiles)                       # 语料隐(后验均值)
    print(f"D4: 语料 {len(smiles)} | latent {d} | 训练整流流 {args.fm_epochs} ep")

    field = VelocityField(d, 1, args.fm_hidden, args.fm_layers)
    fm_path = os.path.join(os.path.dirname(os.path.abspath(args.d3_ckpt)), "fm.pt")
    if os.path.exists(fm_path):
        field.load_state_dict(torch.load(fm_path, map_location="cpu")["model_state"])
        print(f"复用已训练流 {fm_path}")
    else:
        _train_flow(field, Z, args.fm_epochs, args.batch, args.lr, args.seed)
        torch.save({"model_state": field.state_dict()}, fm_path)
        print(f"saved {fm_path}")
    field.eval()

    if args.no_oracle:
        oracle = lambda xs: [0.0] * len(xs)  # noqa: E731
    else:
        from ._tdc_compat import get_oracle
        oracle = get_oracle("DRD2")

    n, d_ = args.n_sample, d
    batches = {
        "unguided": _flow_sample(field, head, n, d_, args.steps, 0.0, args.seed),
        "guided":   _flow_sample(field, head, n, d_, args.steps, args.guide_scale, args.seed),
        "grad-only": _grad_only(head, n, d_, args.steps, args.ga_lr, args.seed),
    }
    rows = [_eval_batch(lbl, Z_, dec, head, oracle) for lbl, Z_ in batches.items()]
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "flow_guided_d4.csv"), index=False)
    print(f"\n==== D4 流引导采样 (n={n}, steps={args.steps}, guide={args.guide_scale}) ====")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(res.to_string(index=False))
    g = res[res.method == "guided"].iloc[0]; u = res[res.method == "unguided"].iloc[0]
    ga = res[res.method == "grad-only"].iloc[0]
    print(f"\n引导 vs 纯流: mean_DRD2 {u.mean_drd2:.3f} -> {g.mean_drd2:.3f} "
          f"({'[OK up]' if g.mean_drd2 > u.mean_drd2 else '[!]'}), 有效率 {u.validity:.2f}->{g.validity:.2f}")
    print(f"off-manifold 对照(grad-only): head自评 {ga.head_prob:.3f}(高) 但 真实DRD2 {ga.mean_drd2:.3f}/"
          f"有效率 {ga.validity:.2f} -> {'[OK] 印证脱流形被 reward-hack' if ga.validity < g.validity else '?'}")

    if args.plot:
        _plot(res, out_dir, args.guide_scale)


def _plot(res, out_dir, guide_scale):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..core.plotting import use_cjk_font
    use_cjk_font()
    m = res["method"].tolist(); x = np.arange(len(m))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.bar(x - 0.2, res["mean_drd2"], 0.4, label="真实 DRD2(均)", color="darkorange")
    ax1.bar(x + 0.2, res["head_prob"], 0.4, label="属性头自评", color="steelblue")
    ax1.set_xticks(x); ax1.set_xticklabels(m); ax1.legend(); ax1.set_title("活性: 真值 vs 引导头自评")
    ax2.bar(x, res["validity"], color="seagreen")
    ax2.set_xticks(x); ax2.set_xticklabels(m); ax2.set_ylabel("解码有效率 ↑")
    ax2.set_title("是否在数据流形上")
    fig.suptitle(f"D4 流匹配+梯度引导采样 (guide={guide_scale})")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "flow_guided_d4.png"), dpi=130); plt.close(fig)
    print(f"图: {os.path.join(out_dir, 'flow_guided_d4.png')}")


if __name__ == "__main__":
    main()
