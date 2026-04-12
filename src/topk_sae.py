"""TopK Sparse Autoencoder for Novae mechinterp.

Mirrors the upstream `arXiv:2603.02952` SAE recipe:
- Linear encoder with pre-bias centering
- TopK sparsification (no ReLU; signed activations are kept)
- Linear decoder with bias, columns L2-renormalized to unit norm after every step
- MSE reconstruction loss
- Optional auxiliary loss to revive dead features (off by default)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    """TopK sparse autoencoder.

    Parameters
    ----------
    d_in : int
        Dimensionality of the activation being decomposed.
    n_features : int
        Dictionary size (number of SAE features).
    k : int
        Number of features kept active per row after the TopK step.
    """

    def __init__(self, d_in: int, n_features: int, k: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.n_features = n_features
        self.k = k

        self.encoder = nn.Linear(d_in, n_features, bias=False)
        self.encoder_bias = nn.Parameter(torch.zeros(n_features))
        self.pre_bias = nn.Parameter(torch.zeros(d_in))
        self.decoder = nn.Linear(n_features, d_in, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            # decoder columns init to unit norm random vectors
            W = torch.randn(self.d_in, self.n_features)
            W /= W.norm(dim=0, keepdim=True).clamp(min=1e-8)
            self.decoder.weight.copy_(W)
            self.decoder.bias.zero_()
            # tie encoder = decoder.T (Anthropic-style init)
            self.encoder.weight.copy_(W.t())
            self.encoder_bias.zero_()
            self.pre_bias.zero_()

    # ------------------------------------------------------------------ encode

    def encode_dense(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-TopK feature activations (signed)."""
        return self.encoder(x - self.pre_bias) + self.encoder_bias

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply TopK on the absolute value, keep signed values.

        Returns
        -------
        z : (B, n_features) sparse tensor — exactly k non-zeros per row
        idx : (B, k) indices of the kept features
        """
        z_dense = self.encode_dense(x)
        # rank by magnitude so signed features compete fairly
        _, idx = z_dense.abs().topk(self.k, dim=-1)
        z = torch.zeros_like(z_dense)
        z.scatter_(-1, idx, z_dense.gather(-1, idx))
        return z, idx

    # ------------------------------------------------------------------ decode

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) + self.pre_bias

    # ------------------------------------------------------------------ forward

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, idx = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z, idx

    # ------------------------------------------------------------------ utils

    @torch.no_grad()
    def renorm_decoder(self) -> None:
        """Project decoder columns back onto the unit sphere."""
        norms = self.decoder.weight.data.norm(dim=0, keepdim=True)
        self.decoder.weight.data /= norms.clamp(min=1e-8)

    @torch.no_grad()
    def feature_alive_mask(self, x: torch.Tensor, n_chunks: int = 16) -> torch.Tensor:
        """Boolean mask of features that fire on at least one row of x."""
        alive = torch.zeros(self.n_features, dtype=torch.bool, device=x.device)
        for chunk in x.chunk(n_chunks):
            _, idx = self.encode(chunk)
            alive[idx.flatten().unique()] = True
        return alive
