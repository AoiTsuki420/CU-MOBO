"""D3: 属性感知(property-aware)隐空间训练 —— LOL-BO/CoBO/PG-LBO 思路的最小实例.

痛点(我们实测): 纯重构 VAE 的隐空间与"活性"无关 → 隐空间坍缩/饱和、GP 学不动。
解法: 给 VAE 隐空间加一个**属性预测头** z→活性, 联合训练(recon + β·KL + word-dropout
  + w_prop·属性损失)。属性监督把隐空间**按活性组织**, 让 BO/GP 在其上更顺。

廉价验证(不跑整轮 BO): **z→活性 的线性探针 AUC**。property-aware 隐空间 AUC 应 >
  vanilla(纯重构)隐空间 → 隐几何与目标相关 = 对 BO 更友好(CoBO 的核心主张)。

复用 molgen SmilesVAE(默认架构, VaeDecoder 可直接 load), 属性头单独存。

用法:
    python -m puridiff_optim.molecule.train_latent_aware \
        --corpus tdc_data/drd2_corpus_hard.csv --prop-col is_drd2_active \
        --out puridiff_optim/molecule/assets/vae_drd2_d3/vae.pt \
        --vanilla puridiff_optim/molecule/assets/vae_drd2_hard/vae.pt --epochs 40 --w-prop 5.0
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ._molgen import SmilesVAE, Vocab, canonicalize_smiles
from data_utils import PAD, BOS, UNK  # noqa: E402


class PropHead(nn.Module):
    def __init__(self, latent_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


def _word_dropout(x_in, p):
    if p <= 0:
        return x_in
    drop = (torch.rand_like(x_in, dtype=torch.float) < p) & (x_in != PAD) & (x_in != BOS)
    out = x_in.clone(); out[drop] = UNK
    return out


def _probe_auc(vae, vocab, max_len, smiles, y, seed=0):
    """z(后验均值)→属性 的线性探针 AUC(留出), 衡量隐空间与属性的相关性."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    ids = torch.tensor([vocab.encode(s, max_len) for s in smiles], dtype=torch.long)
    with torch.no_grad():
        mu, _ = vae.encode(ids)
    Z = mu.cpu().numpy()
    Ztr, Zte, ytr, yte = train_test_split(Z, y, test_size=0.3, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=500).fit(Ztr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Zte)[:, 1]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--prop-col", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vanilla", default=None, help="对照(纯重构)VAE ckpt, 比较探针 AUC")
    ap.add_argument("--max-len", type=int, default=100)
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--free-bits", type=float, default=0.2)
    ap.add_argument("--word-dropout", type=float, default=0.4)
    ap.add_argument("--w-prop", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    df = pd.read_csv(args.corpus)
    keep = [(canonicalize_smiles(str(s)), float(p)) for s, p in zip(df["smiles"], df[args.prop_col])]
    keep = [(s, p) for s, p in keep if s]
    smiles = [s for s, _ in keep]; y = np.array([p for _, p in keep], float)
    vocab = Vocab.build(smiles)
    X = torch.tensor([vocab.encode(s, args.max_len) for s in smiles], dtype=torch.long)
    yt = torch.tensor(y, dtype=torch.float)
    loader = DataLoader(TensorDataset(X, yt), batch_size=args.batch_size, shuffle=True)
    print(f"corpus {len(smiles)} | prop '{args.prop_col}' pos-rate {y.mean():.2f} | "
          f"vocab {len(vocab.id_to_token)} | w_prop {args.w_prop}")

    vae = SmilesVAE(len(vocab.id_to_token), args.max_len, args.latent_dim, args.d_model)
    head = PropHead(args.latent_dim)
    opt = torch.optim.AdamW(list(vae.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=1e-2)
    fb = float(args.free_bits)
    for epoch in range(1, args.epochs + 1):
        vae.train(); head.train(); tot = rl = kll = pl = 0.0; nb = 0
        beta = args.beta * min(1.0, epoch / max(1, args.epochs // 3))
        for x, yb in loader:
            mu, logvar = vae.encode(x)
            std = torch.exp(0.5 * logvar); z = mu + std * torch.randn_like(std)
            logits = vae.decode_logits(z, _word_dropout(x[:, :-1], args.word_dropout))
            recon = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), ignore_index=PAD)
            kl = torch.clamp(-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()), min=fb).sum(1).mean()
            prop = F.binary_cross_entropy_with_logits(head(mu), yb)   # 属性监督(用后验均值)
            loss = recon + beta * kl + args.w_prop * prop
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(list(vae.parameters()) + list(head.parameters()), 1.0); opt.step()
            tot += float(loss.detach()); rl += float(recon.detach()); kll += float(kl.detach())
            pl += float(prop.detach()); nb += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"[d3-vae] ep {epoch:03d} loss={tot/nb:.3f} recon={rl/nb:.3f} kl={kll/nb:.3f} prop={pl/nb:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"model_state": vae.state_dict(),
                "vocab": {"token_to_id": vocab.token_to_id, "id_to_token": vocab.id_to_token},
                "prop_head_state": head.state_dict(), "condition_to_id": {"uncond": 0},
                "config": {"max_len": args.max_len, "latent_dim": args.latent_dim, "d_model": args.d_model}},
               args.out)
    print(f"saved {args.out}")

    # 解码器健康自检(关键: 强属性监督会塌解码器 → D4 采样无效)
    try:
        from .train_drug_vae import _selfcheck
        _selfcheck(args.out, smiles)
    except Exception as e:
        print(f"[selfcheck] skipped: {e}")

    # 验证: z→属性 线性探针 AUC(property-aware vs vanilla)
    vae.eval()
    auc_aware = _probe_auc(vae, vocab, args.max_len, smiles, y, args.seed)
    print(f"[D3 probe] property-aware 隐空间 z→{args.prop_col} 线性探针 AUC = {auc_aware:.3f}")
    if args.vanilla and os.path.exists(args.vanilla):
        from .vae_decoder import VaeDecoder
        vdec = VaeDecoder(args.vanilla)
        auc_van = _probe_auc(vdec.vae, vdec.vocab, vdec.max_len, smiles, y, args.seed)
        print(f"[D3 probe] vanilla(纯重构) 隐空间 AUC = {auc_van:.3f}  ->  "
              f"{'[OK] property-aware 更强' if auc_aware > auc_van else '[!] 未提升'} (delta={auc_aware-auc_van:+.3f})")


if __name__ == "__main__":
    main()
