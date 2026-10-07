"""Step 4: the mixed-type autoencoder family and the classifier built on its encoder (PyTorch).

Encoder  : per-column embeddings for categorical columns, concatenated with continuous columns -> MLP trunk -> latent
Decoder  : latent -> MLP trunk -> one regression head (all continuous columns) + one softmax head per categorical column
Variants :
  ae   plain autoencoder          loss = w_cont*MSE(continuous) + w_cat*mean CE(categorical)
  dae  denoising autoencoder      same loss, but the INPUT is corrupted (Gaussian noise + random 'unknown' masking) and the
                                  CLEAN row must be reconstructed -> encoder cannot just copy; learns feature dependencies
  vae  variational autoencoder    same reconstruction loss + beta * KL(q(z|x) || N(0, I)); stochastic latent, smoother space
Classifier: encoder (latent mean) -> head (linear or small MLP) -> one logit; BCE-with-logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp_trunk(in_dim: int, hidden: list[int], dropout: float = 0.0) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
        d = h
    return nn.Sequential(*layers)


class MixedAutoencoder(nn.Module):
    def __init__(self, n_cont: int, cards: list[int], kind: str = "ae", hidden: int = 128, depth: int = 2,
                 latent: int = 16, dropout: float = 0.0, emb_max: int = 8):
        super().__init__()
        self.kind, self.n_cont, self.cards, self.latent, self.hidden = kind, n_cont, list(cards), latent, hidden
        self.embs = nn.ModuleList([nn.Embedding(c, min(emb_max, max(2, (c + 1) // 2))) for c in cards])
        in_dim = n_cont + sum(e.embedding_dim for e in self.embs)
        self.enc = mlp_trunk(in_dim, [hidden] * depth, dropout)
        self.to_mu = nn.Linear(hidden, latent)
        self.to_logvar = nn.Linear(hidden, latent) if kind == "vae" else None
        self.dec = mlp_trunk(latent, [hidden] * depth, dropout)
        self.cont_head = nn.Linear(hidden, n_cont) if n_cont else None
        self.cat_heads = nn.ModuleList([nn.Linear(hidden, c) for c in cards])

    def _inputs(self, xc, xk):
        parts = [xc] if self.n_cont else []
        parts += [e(xk[:, j]) for j, e in enumerate(self.embs)]
        return torch.cat(parts, dim=1)

    def encode(self, xc, xk):
        h = self.enc(self._inputs(xc, xk))
        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(-10, 10) if self.to_logvar is not None else None
        return mu, logvar

    def decode(self, z):
        h = self.dec(z)
        return (self.cont_head(h) if self.cont_head is not None else None), [hd(h) for hd in self.cat_heads]

    def forward(self, xc, xk):
        mu, logvar = self.encode(xc, xk)
        z = mu
        if logvar is not None and self.training:           # reparameterisation trick
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        cont, logits = self.decode(z)
        return {"mu": mu, "logvar": logvar, "cont": cont, "logits": logits}

    def encoder_parameters(self):
        return [p for m in (self.embs, self.enc, self.to_mu) for p in m.parameters()]


def reconstruction_loss(out, xc, xk, w_cont: float = 1.0, w_cat: float = 1.0):
    """Returns (weighted total, continuous MSE, mean categorical CE)."""
    cont = F.mse_loss(out["cont"], xc) if out["cont"] is not None else xc.new_zeros(())
    cat = torch.stack([F.cross_entropy(lg, xk[:, j]) for j, lg in enumerate(out["logits"])]).mean()
    return w_cont * cont + w_cat * cat, cont.detach(), cat.detach()


def kl_divergence(mu, logvar):
    """KL(q||N(0,I)) averaged over batch AND latent dimensions (keeps beta on the same scale as the per-element recon loss)."""
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()


def corrupt(xc, xk, sigma: float, mask_p: float, unknown_idx):
    """Denoising corruption: Gaussian noise on continuous columns, random replacement by the 'unknown' code on categoricals."""
    xc2 = xc + sigma * torch.randn_like(xc)
    mask = torch.rand(xk.shape, device=xk.device) < mask_p
    return xc2, torch.where(mask, unknown_idx.expand_as(xk), xk)


class SatisfactionClassifier(nn.Module):
    """Encoder + classification layer(s). head_hidden=0 -> a single linear layer (linear probe)."""

    def __init__(self, ae: MixedAutoencoder, head_hidden: int = 32, head_dropout: float = 0.1):
        super().__init__()
        self.ae = ae
        self.head = (nn.Linear(ae.latent, 1) if head_hidden == 0 else
                     nn.Sequential(nn.Linear(ae.latent, head_hidden), nn.GELU(), nn.Dropout(head_dropout), nn.Linear(head_hidden, 1)))

    def forward(self, xc, xk):
        mu, _ = self.ae.encode(xc, xk)
        return self.head(mu).squeeze(-1)


def count_trainable(clf: SatisfactionClassifier, mode: str) -> int:
    head = sum(p.numel() for p in clf.head.parameters())
    enc = 0 if mode == "frozen" else sum(p.numel() for p in clf.ae.encoder_parameters())
    return head + enc
