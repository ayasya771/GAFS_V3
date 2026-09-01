"""Temporal Fusion Transformer building blocks.

GLU   gating that lets the network suppress a whole nonlinear branch.
GRN   Gated Residual Network: nonlinear processing with a gated residual
      skip, so simple linear mappings pass through untouched when they are
      sufficient (training stability).
VSN   Variable Selection Network: learned per-time-step softmax weights over
      input variables, focusing on the drivers of the current regime.
"""

from __future__ import annotations

import torch
from torch import nn


class GLU(nn.Module):
    """Gated Linear Unit: (W1 x) * sigmoid(W2 x)."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size * 2)
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class AddGateNorm(nn.Module):
    """LayerNorm(residual + GLU(x)): the TFT gate-add-norm block."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.glu = GLU(input_size, output_size)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.norm(residual + self.glu(x))


class GRN(nn.Module):
    """Gated Residual Network with optional static context injection."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int | None = None,
        context_size: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        output_size = output_size or hidden_size
        self.skip = (
            nn.Linear(input_size, output_size)
            if input_size != output_size
            else nn.Identity()
        )
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ctx = (
            nn.Linear(context_size, hidden_size, bias=False)
            if context_size
            else None
        )
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gate = GLU(hidden_size, output_size)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        h = self.fc1(x)
        if self.ctx is not None and context is not None:
            while context.dim() < h.dim():
                context = context.unsqueeze(-2)
            h = h + self.ctx(context)
        h = self.fc2(self.elu(h))
        h = self.dropout(h)
        return self.norm(self.skip(x) + self.gate(h))


class VariableSelection(nn.Module):
    """Per-time-step soft selection over scalar input variables.

    Input  [B, T, V] raw variables.
    Output [B, T, H] fused representation and [B, T, V] selection weights
    (softmax, summing to 1), interpretable as which drivers matter now.
    """

    def __init__(
        self,
        n_vars: int,
        hidden_size: int,
        dropout: float = 0.1,
        context_size: int | None = None,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.embed = nn.ModuleList(nn.Linear(1, hidden_size) for _ in range(n_vars))
        self.var_grns = nn.ModuleList(
            GRN(hidden_size, hidden_size, dropout=dropout) for _ in range(n_vars)
        )
        self.weight_grn = GRN(
            n_vars * hidden_size,
            hidden_size,
            output_size=n_vars,
            context_size=context_size,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-1] != self.n_vars:
            raise ValueError(f"Expected {self.n_vars} variables, got {x.shape[-1]}")
        embedded = [emb(x[..., i : i + 1]) for i, emb in enumerate(self.embed)]
        flat = torch.cat(embedded, dim=-1)
        weights = torch.softmax(self.weight_grn(flat, context), dim=-1)
        processed = torch.stack(
            [grn(e) for grn, e in zip(self.var_grns, embedded)], dim=-1
        )
        fused = (processed * weights.unsqueeze(-2)).sum(dim=-1)
        return fused, weights
