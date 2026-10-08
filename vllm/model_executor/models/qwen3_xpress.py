# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 grouped convolutions with an xPress full-vocabulary refiner."""

from collections.abc import Iterable

import torch

from vllm.config import VllmConfig

from .qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model
from .qwen3_dflash2 import DFlash2Qwen3DecoderLayer
from .utils import AutoWeightsLoader
from .xpress_head import XPressHead


class XPressModel(DFlashQwen3Model):
    decoder_layer_cls = DFlash2Qwen3DecoderLayer

    def __init__(
        self, *, vllm_config: VllmConfig, start_layer_id: int = 0, prefix: str = ""
    ):
        spec = vllm_config.speculative_config
        assert spec is not None
        config = spec.draft_model_config.hf_config
        geometry = config.dflash_config
        if not 1 <= spec.num_speculative_tokens < geometry["block_size"]:
            raise ValueError(
                "xPress requires 1 <= num_speculative_tokens < trained block_size"
            )
        if geometry["xpress_rank"] <= 0 or geometry["xpress_refinement_steps"] <= 0:
            raise ValueError("xPress rank and refinement steps must be positive")
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        self.xpress_head = XPressHead(
            config.hidden_size,
            config.vocab_size,
            geometry["block_size"],
            geometry["xpress_rank"],
        ).to(dtype=spec.draft_model_config.dtype)


class XPressForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = XPressModel

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if self.draft_id_to_target_id is not None:
            raise ValueError("xPress requires the full target vocabulary")
        # Every TP rank runs the same replicated refiner and needs all logits.
        self.logits_processor.use_all_gather = True

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        head_weights: list[tuple[str, torch.Tensor]] = []
        seen: set[str] = set()

        def backbone_weights():
            for name, value in weights:
                name = name.removeprefix("model.")
                seen.add(name)
                # Keep the head's separate SwiGLU matrices out of the backbone's
                # gate_up_proj packing rules.
                if name.startswith("xpress_head."):
                    head_weights.append((name.removeprefix("xpress_head."), value))
                else:
                    yield name, value

        super().load_weights(backbone_weights())
        head = self.model.xpress_head
        required = {f"xpress_head.{name}" for name, _ in head.named_parameters()}
        required.update(
            name.removeprefix("model.")
            for name, _ in self.named_parameters()
            if ".attention_conv." in name or ".mlp_conv." in name
        )
        if missing := required - seen:
            raise ValueError(f"xPress checkpoint missing tensors: {sorted(missing)}")
        AutoWeightsLoader(head).load_weights(head_weights)
        head.prepare_for_inference()
