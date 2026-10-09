"""自带抗后验坍缩(posterior collapse)的 drug-like SmilesVAE 训练器.

molgen 的 `latent_flow_matching.py --mode train_vae` 在强自回归 GRU 解码器下会**编码器
坍缩**: 不同分子全编码到原点附近(mu std≈0.1), 解码器无视 z → 隐空间不可控, 无法做隐
空间优化。本训练器复用 molgen `SmilesVAE`(**保持默认架构**以便 `VaeDecoder` 直接加载),
但加三件标准抗坍缩措施:
  1. **word dropout**(Bowman 2016): teacher-forcing 输入随机替换成 [UNK], 强迫解码器靠 z;
  2. **free bits**: 每维 KL 至少 λ nats, 不被压到 0;
  3. **KL warmup** + 适度 β。

存档格式与 `VaeDecoder` 兼容。训练后自检 encode→decode 唯一数(坍缩指标)。

用法:
    python -m puridiff_optim.molecule.train_drug_vae \
        --corpus tdc_data/drd2_vae/smiles_corpus.csv \
        --out puridiff_optim/molecule/assets/vae_drd2_wd/vae.pt \
        --latent-dim 16 --epochs 60 --word-dropout 0.4
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ._molgen import SmilesVAE, Vocab, canonicalize_smiles
# import ._molgen 已把 molgen src 加进 sys.path, 故可直接取其特殊 token id
from data_utils import PAD, BOS, UNK  # noqa: E402


def _encode_corpus(smiles, vocab, max_len):
    return torch.tensor([vocab.encode(s, max_len) for s in smiles], dtype=torch.long)


def _word_dropout(x_in, p, keep_ids=(PAD, BOS)):
    """以概率 p 把 teacher-forcing 输入 token 替成 [UNK](保留 PAD/BOS)."""
    if p <= 0:
        return x_in
    drop = (torch.rand_like(x_in, dtype=torch.float) < p)
    for k in keep_ids:
        drop &= (x_in != k)
    out = x_in.clone()
    out[drop] = UNK
    return out


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    df = pd.read_csv(args.corpus)
    col = "smiles" if "smiles" in df.columns else df.columns[0]
    smiles = [s for s in df[col].astype(str).tolist() if canonicalize_smiles(s)]
    smiles = [canonicalize_smiles(s) for s in smiles]
    vocab = Vocab.build(smiles)
    X = _encode_corpus(smiles, vocab, args.max_len)
    loader = DataLoader(TensorDataset(X), batch_size=args.batch_size, shuffle=True)
    print(f"corpus {len(smiles)} mols | vocab {len(vocab.id_to_token)} | max_len {args.max_len} "
          f"| latent {args.latent_dim} | word_dropout {args.word_dropout} | free_bits {args.free_bits}")

    vae = SmilesVAE(len(vocab.id_to_token), args.max_len, args.latent_dim, args.d_model)
    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr, weight_decay=1e-2)
    fb = float(args.free_bits)
    for epoch in range(1, args.epochs + 1):
        vae.train(); tot = rl = kll = 0.0; nb = 0
        beta = args.beta * min(1.0, epoch / max(1, args.epochs // 3))  # KL warmup
        for (x,) in loader:
            mu, logvar = vae.encode(x)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            x_in = _word_dropout(x[:, :-1], args.word_dropout)
            logits = vae.decode_logits(z, x_in)
            recon = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                    x[:, 1:].reshape(-1), ignore_index=PAD)
            # free-bits KL: 每维 max(kl_dim, fb) 求和
            kl_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())   # (B, d)
            kl = torch.clamp(kl_dim, min=fb).sum(1).mean()
            loss = recon + beta * kl
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0); opt.step()
            tot += float(loss.detach()); rl += float(recon.detach()); kll += float(kl.detach()); nb += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"[wd-vae] epoch {epoch:03d} loss={tot/nb:.4f} recon={rl/nb:.4f} "
                  f"kl={kll/nb:.4f} (beta={beta:.3f})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"model_state": vae.state_dict(),
                "vocab": {"token_to_id": vocab.token_to_id, "id_to_token": vocab.id_to_token},
                "condition_to_id": {"uncond": 0},
                "config": {"max_len": args.max_len, "latent_dim": args.latent_dim, "d_model": args.d_model}},
               args.out)
    print(f"saved {args.out}")
    _selfcheck(args.out, smiles)


def _selfcheck(ckpt, smiles):
    """坍缩自检: encode 多个分子 -> decode, 看唯一数 & 重构匹配 & 随机隐有效/唯一."""
    from .vae_decoder import VaeDecoder
    from ..core.sampling import latin_hypercube
    dec = VaeDecoder(ckpt)
    ref = smiles[:80]
    Zc = dec.encode(ref)
    rec = dec.decode(Zc)
    exact = sum(1 for a, r in zip(ref, rec) if r == a)
    uniq = len({r for r in rec if r})
    d = dec.latent_dim
    Z = latin_hypercube(300, np.full(d, -2.5), np.full(d, 2.5), seed=1)
    rs = [s for s in dec.decode(Z) if s]
    print(f"[selfcheck] encoder mu std={Zc.std():.3f} | encode->decode unique {uniq}/80 exact {exact}/80 "
          f"| LHS[-2.5,2.5] valid {len(rs)}/300 unique {len(set(rs))}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=100)
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--free-bits", type=float, default=0.15)
    ap.add_argument("--word-dropout", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=42)
    train(ap.parse_args(argv))


if __name__ == "__main__":
    main()
