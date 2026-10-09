"""加载训练好的 SmilesVAE, 提供 latent z <-> SMILES 的 encode/decode.

decode 用 greedy (确定性): 同一个 z 总解码到同一分子, 利于优化稳定.
"""

from __future__ import annotations

import numpy as np
import torch

from ._molgen import SmilesVAE, Vocab, canonicalize_smiles


class VaeDecoder:
    def __init__(self, ckpt_path, device="cpu"):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        c = ck["config"]
        self.vocab = Vocab(**ck["vocab"])
        self.max_len = int(c["max_len"])
        self.latent_dim = int(c["latent_dim"])
        self.vae = SmilesVAE(len(self.vocab.id_to_token), c["max_len"],
                             c["latent_dim"], c["d_model"]).to(device)
        self.vae.load_state_dict(ck["model_state"])
        self.vae.eval()
        self.device = device

    @torch.no_grad()
    def decode(self, Z):
        """latent (N, d) -> list[str|None] (canonical SMILES, 无效则 None)."""
        Z = torch.as_tensor(np.atleast_2d(np.asarray(Z, float)),
                            dtype=torch.float32, device=self.device)
        toks = self.vae.generate(Z, greedy=True)
        out = []
        for row in toks.cpu().tolist():
            raw = self.vocab.decode(row)
            out.append(canonicalize_smiles(raw) if raw else None)
        return out

    @torch.no_grad()
    def encode(self, smiles_list):
        """SMILES 列表 -> 后验均值 latent (N, d), 用于 warm-start 优化."""
        ids = [self.vocab.encode(str(s), self.max_len) for s in smiles_list]
        x = torch.tensor(ids, dtype=torch.long, device=self.device)
        mu, _ = self.vae.encode(x)
        return mu.cpu().numpy()
