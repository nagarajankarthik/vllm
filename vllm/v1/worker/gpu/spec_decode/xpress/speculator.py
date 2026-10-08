# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator


class XPressSpeculator(DFlashSpeculator):
    _speculator_name = "xPress"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        spec = vllm_config.speculative_config
        assert spec is not None
        if spec.draft_sample_method != "greedy":
            raise ValueError("xPress currently requires draft_sample_method='greedy'")
        if spec.enable_adaptive_verification:
            raise ValueError("xPress adaptive verification is not implemented")
        super().__init__(vllm_config, device)
        self.predecessor_ids = torch.zeros(
            self.max_num_reqs, dtype=torch.long, device=device
        )
        self.refinement_steps = int(
            self.draft_model_config.hf_config.dflash_config["xpress_refinement_steps"]
        )

    def prepare_context_anchor(
        self, input_batch: InputBatch, num_rejected: torch.Tensor
    ) -> None:
        """The bonus anchor follows the last accepted input, not a rejected guess."""
        n = input_batch.num_reqs
        starts = input_batch.query_start_loc[:n]
        ends = input_batch.query_start_loc[1 : n + 1] - num_rejected[:n]
        self.predecessor_ids.zero_()
        self.predecessor_ids[:n].copy_(
            torch.where(
                ends > starts,
                input_batch.input_ids[(ends - 1).clamp_min(0).long()],
                0,
            )
        )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        width = self.num_query_per_req
        full_hidden = hidden[: num_reqs * width].view(num_reqs, width, hidden.shape[-1])
        logits = self.model.compute_logits(
            full_hidden[:, 1:].reshape(-1, hidden.shape[-1])
        )
        logits = logits.view(num_reqs, self.num_speculative_steps, logits.shape[-1])
        anchors = self.input_buffers.input_ids[: num_reqs * width : width]
        tokens = self.model.model.xpress_head.refine(
            full_hidden,
            logits,
            anchors,
            self.predecessor_ids[:num_reqs],
            self.refinement_steps,
        )
        active = (
            self.sample_idx_mapping[
                : num_reqs * self.num_speculative_steps : self.num_speculative_steps
            ]
            >= 0
        )
        self.draft_tokens[:num_reqs].copy_(torch.where(active[:, None], tokens, 0))
