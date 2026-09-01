"""Conditional Temporal Fusion Transformer generator.

Maps historical context X_{t-k:t}, macro conditioning C_t and per-step
Gaussian noise Z_t ~ N(0, I) to a stochastic future path:

    Y_hat_{t:t+h} = G(X_{t-k:t}, C_t, Z_t)

Structure (faithful to Lim et al., 2021, adapted to path generation):
  variable selection over inputs -> context-initialised LSTM encoder ->
  noise-driven LSTM decoder -> gate/add/norm -> static enrichment GRN ->
  causal interpretable multi-head attention -> position-wise GRN ->
  final gate with a decoder skip -> output head with noise re-injection.
"""

from __future__ import annotations

import torch
from torch import nn

from .layers import GRN, AddGateNorm, VariableSelection


class TFTGenerator(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_assets: int,
        cond_dim: int,
        lookback: int,
        horizon: int,
        hidden: int = 64,
        heads: int = 4,
        z_dim: int = 16,
        dropout: float = 0.1,
        lstm_layers: int = 1,
    ):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        self.n_features = n_features
        self.n_assets = n_assets
        self.cond_dim = cond_dim
        self.lookback = lookback
        self.horizon = horizon
        self.hidden = hidden
        self.z_dim = z_dim
        self.lstm_layers = lstm_layers

        cond_in = max(cond_dim, 1)
        self.cond_encoder = GRN(cond_in, hidden, dropout=dropout)
        self.ctx_select = GRN(hidden, hidden, dropout=dropout)
        self.ctx_h = GRN(hidden, hidden, dropout=dropout)
        self.ctx_c = GRN(hidden, hidden, dropout=dropout)
        self.ctx_enrich = GRN(hidden, hidden, dropout=dropout)

        self.vsn = VariableSelection(
            n_features, hidden, dropout=dropout, context_size=hidden
        )

        self.enc_lstm = nn.LSTM(hidden, hidden, lstm_layers, batch_first=True)
        self.dec_input = nn.Linear(z_dim + hidden, hidden)
        self.dec_lstm = nn.LSTM(hidden, hidden, lstm_layers, batch_first=True)
        self.gate_enc = AddGateNorm(hidden, hidden)
        self.gate_dec = AddGateNorm(hidden, hidden)

        self.enrich = GRN(hidden, hidden, context_size=hidden, dropout=dropout)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.gate_attn = AddGateNorm(hidden, hidden)
        self.pos_ff = GRN(hidden, hidden, dropout=dropout)
        self.gate_final = AddGateNorm(hidden, hidden)

        self.head = nn.Linear(hidden + z_dim, n_assets)

        mask = torch.full((horizon, lookback + horizon), False, dtype=torch.bool)
        for i in range(horizon):
            mask[i, lookback + i + 1 :] = True
        self.register_buffer("attn_mask", mask, persistent=False)

    def sample_noise(self, batch: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.randn(batch, self.horizon, self.z_dim, device=device)

    def forward(
        self,
        x_hist: torch.Tensor,
        cond: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ):
        B, k, F = x_hist.shape
        if k != self.lookback or F != self.n_features:
            raise ValueError(
                f"x_hist has shape {tuple(x_hist.shape)}, expected "
                f"[B, {self.lookback}, {self.n_features}]"
            )
        if cond is None or cond.shape[-1] == 0:
            cond = torch.zeros(B, 1, device=x_hist.device, dtype=x_hist.dtype)
        if z is None:
            z = self.sample_noise(B, x_hist.device)

        c = self.cond_encoder(cond)
        selected, var_weights = self.vsn(x_hist, self.ctx_select(c))

        h0 = self.ctx_h(c).unsqueeze(0).expand(self.lstm_layers, B, self.hidden).contiguous()
        c0 = self.ctx_c(c).unsqueeze(0).expand(self.lstm_layers, B, self.hidden).contiguous()
        enc_out, enc_state = self.enc_lstm(selected, (h0, c0))
        enc_out = self.gate_enc(enc_out, selected)

        c_dec = c.unsqueeze(1).expand(B, self.horizon, self.hidden)
        dec_in = self.dec_input(torch.cat([z, c_dec], dim=-1))
        dec_out, _ = self.dec_lstm(dec_in, enc_state)
        dec_out = self.gate_dec(dec_out, dec_in)

        seq = torch.cat([enc_out, dec_out], dim=1)
        enriched = self.enrich(seq, self.ctx_enrich(c))
        queries = enriched[:, self.lookback :, :]
        att, att_weights = self.attn(
            queries, enriched, enriched,
            attn_mask=self.attn_mask,
            need_weights=return_diagnostics,
            average_attn_weights=True,
        )
        att = self.gate_attn(att, queries)
        ff = self.pos_ff(att)
        fused = self.gate_final(ff, dec_out)

        y = self.head(torch.cat([fused, z], dim=-1))
        if return_diagnostics:
            return y, {"variable_weights": var_weights, "attention": att_weights}
        return y

    @torch.no_grad()
    def sample(
        self,
        x_hist: torch.Tensor,
        cond: torch.Tensor | None,
        n_samples: int,
        batch_size: int = 256,
    ) -> torch.Tensor:
        """Draw n_samples stochastic paths for ONE context ([1, k, F])."""
        was_training = self.training
        self.eval()
        outs = []
        remaining = n_samples
        while remaining > 0:
            b = min(batch_size, remaining)
            xh = x_hist.expand(b, -1, -1)
            cd = cond.expand(b, -1) if cond is not None and cond.numel() else None
            outs.append(self.forward(xh, cd))
            remaining -= b
        if was_training:
            self.train()
        return torch.cat(outs, dim=0)
