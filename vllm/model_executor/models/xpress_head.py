# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""xPress released head equations, with AutoModel's raw training weight layout.

Reference: Supercomputing-System-AI-Lab/xPress, refiners/xpress_head.py.
"""

import torch
import torch.nn.functional as F
from torch import nn


class XPressHead(nn.Module):
    """Replicated low-rank correction; no TP collective inside a Jacobi pass."""

    def __init__(self, hidden_size: int, vocab_size: int, block_size: int, rank: int):
        super().__init__()
        self.down_h = nn.Linear(hidden_size, rank, bias=False)
        self.down_g = nn.Linear(hidden_size, rank, bias=False)
        self.w1 = nn.Embedding(vocab_size, rank)
        self.in_proj = nn.Linear(3 * rank, rank, bias=False)
        self.mix = nn.ParameterDict(
            {"L": nn.Parameter(torch.zeros(rank, block_size, block_size))}
        )
        self.mlp = nn.ModuleDict(
            {
                "gate_proj": nn.Linear(rank, 2 * rank, bias=False),
                "up_proj": nn.Linear(rank, 2 * rank, bias=False),
                "down_proj": nn.Linear(2 * rank, rank, bias=False),
            }
        )
        self.w2 = nn.Linear(rank, vocab_size, bias=False)
        self.register_buffer("mixer", torch.empty_like(self.mix["L"]), persistent=False)

    @torch.no_grad()
    def prepare_for_inference(self) -> None:
        """Fold mask and identity once; leave the exported trainable tensor intact."""
        raw = self.mix["L"]
        self.mixer.copy_(
            raw.tril() + torch.eye(raw.shape[-1], device=raw.device, dtype=raw.dtype)
        )

    def prepare_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Cache [requests, full block including anchor, 2 * rank] features."""
        global_hidden = self.down_g(hidden.mean(1, keepdim=True)).expand(
            -1, hidden.shape[1], -1
        )
        return torch.cat((self.down_h(hidden), global_hidden), -1)

    def forward(
        self, prepared: torch.Tensor, previous_ids: torch.Tensor
    ) -> torch.Tensor:
        """Return additive logits from shifted tokens, never unshifted guesses."""
        width = prepared.shape[1]
        fused = self.in_proj(torch.cat((prepared, self.w1(previous_ids)), -1))
        mixed = torch.bmm(
            self.mixer[:, :width, :width], fused.permute(2, 1, 0)
        ).permute(2, 1, 0)
        return self.w2(
            mixed
            + self.mlp["down_proj"](
                F.silu(self.mlp["gate_proj"](mixed)) * self.mlp["up_proj"](mixed)
            )
        )

    def refine(
        self,
        hidden: torch.Tensor,
        base_logits: torch.Tensor,
        anchor_ids: torch.Tensor,
        predecessor_ids: torch.Tensor,
        passes: int,
    ) -> torch.Tensor:
        """Greedy Jacobi proposals; logits cover mask slots, hidden includes anchor."""
        prepared = self.prepare_hidden(hidden)
        selected = base_logits.argmax(-1)
        for _ in range(min(passes, base_logits.shape[1])):
            previous = torch.cat(
                (predecessor_ids[:, None], anchor_ids[:, None], selected[:, :-1]), -1
            )
            selected = (
                base_logits.float() + self(prepared, previous)[:, 1:].float()
            ).argmax(-1)
        return selected
