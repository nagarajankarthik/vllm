# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only EAGLE-3 drafter with Qwen4Exp QSA attention and a MoE MLP.

Serves Automodel's ``QwenSparseEagle3DraftModel``: one fused decoder layer
whose attention is Qwen4Exp's gated QSA (with its own indexer) on the
``[embeds | hidden]`` concatenation, and whose MLP is a Qwen3-Next-style MoE
with a sigmoid-gated shared expert. Unlike the Qwen4Exp target and MTP layers
it has no hyper-connections and no PLE, and it rotates 1D positions.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    LocalArgmaxMixin,
    SupportsEagle3,
    SupportsHCSP,
)
from vllm.model_executor.models.qwen3_next import Qwen3NextModel
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    get_draft_quant_config,
    maybe_prefix,
    process_eagle_weight,
)
from vllm.multimodal.inputs import NestedTensors
from vllm.platforms import current_platform

from .low_latency_gemm import enable_qwen4_exp_low_latency_gemm
from .model import _QWEN4_EXP_IGNORED_MISSING_SUFFIXES, Qwen4ExpSparseMoeBlock
from .mtp import _make_draft_vllm_config
from .qsa import Qwen4ExpQSAAttention


def _validate(config, vllm_config: VllmConfig) -> None:
    """Reject configurations the drafter does not reproduce faithfully.

    Runs before ``_make_draft_vllm_config``, which re-derives the shared
    parallel and compilation configs from the draft model config.
    """
    if not current_platform.is_cuda():
        raise NotImplementedError("Qwen4Exp EAGLE-3 is NVIDIA-only")
    if not vllm_config.use_v2_model_runner:
        raise NotImplementedError(
            "Qwen4Exp EAGLE-3 requires the V2 model runner: V1's EAGLE proposer "
            "does not support the QSA indexer caches"
        )
    if getattr(config, "parallel_drafting", False):
        raise NotImplementedError("Qwen4Exp EAGLE-3 does not support P-EAGLE")
    if config.attention_bias:
        raise NotImplementedError("Qwen4Exp QSA attention has no bias")
    if config.indexer_kv_heads != 1:
        raise NotImplementedError("Qwen4Exp QSA requires indexer_kv_heads == 1")
    if "mrope_section" in (config.rope_parameters or {}):
        raise ValueError("The Qwen4Exp EAGLE-3 draft uses 1D RoPE, not MRoPE")
    if config.num_experts_per_tok == 1 and config.norm_topk_prob:
        # Training renormalizes the top-k weights only when k > 1, whereas
        # FusedMoE renormalizes whenever asked.
        raise NotImplementedError(
            "norm_topk_prob with num_experts_per_tok == 1 is not supported"
        )
    parallel_config = vllm_config.parallel_config
    if parallel_config.enable_eplb:
        raise NotImplementedError("Qwen4Exp EAGLE-3 does not support EPLB")
    if parallel_config.enable_expert_parallel:
        ep_size = get_ep_group().world_size
        if config.num_experts % ep_size:
            raise ValueError(
                f"num_experts={config.num_experts} must be divisible by the "
                f"expert-parallel size {ep_size}"
            )
    if get_draft_quant_config(vllm_config) is not None:
        raise NotImplementedError("Qwen4Exp EAGLE-3 supports BF16 drafts only")

    num_target_layers = vllm_config.model_config.get_total_num_hidden_layers()
    layer_ids = list(config.eagle_aux_hidden_state_layer_ids)
    if not all(1 <= layer_id <= num_target_layers for layer_id in layer_ids):
        raise ValueError(
            "eagle_aux_hidden_state_layer_ids must lie in "
            f"[1, {num_target_layers}]; got {layer_ids}"
        )


class Qwen4ExpEagle3DecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        eps = config.rms_norm_eps

        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.hidden_norm = RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)
        self.self_attn = Qwen4ExpQSAAttention(
            vllm_config=vllm_config,
            config=config,
            layer_id=extract_layer_index(prefix),
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.self_attn",
            input_size=2 * hidden_size,
        )
        self.mlp = Qwen4ExpSparseMoeBlock(
            vllm_config=vllm_config,
            prefix=f"{prefix}.mlp",
        )
        if self.mlp.shared_expert is None:
            # Fused shared experts would need checkpoint names remapped from
            # layer 0 to this layer's runtime index.
            raise NotImplementedError("Qwen4Exp EAGLE-3 needs a separate shared expert")

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = torch.cat(
            [self.input_layernorm(embeds), self.hidden_norm(hidden_states)],
            dim=-1,
        )
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen4ExpEagle3Model(nn.Module):
    def __init__(
        self,
        *,
        draft_vllm_config: VllmConfig,
        config,
        start_layer_id: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        eps = config.rms_norm_eps

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.num_aux_hidden_states = config.num_aux_hidden_states
        target_hidden_size = config.target_hidden_size
        # EAGLE-3.1 fc_norm: one RMSNorm per aux chunk, before fc.
        self.fc_norm = (
            nn.ModuleList(
                RMSNorm(target_hidden_size, eps=eps)
                for _ in range(self.num_aux_hidden_states)
            )
            if config.fc_norm
            else None
        )
        self.fc = ReplicatedLinear(
            self.num_aux_hidden_states * target_hidden_size,
            hidden_size,
            bias=False,
            quant_config=None,
            prefix=maybe_prefix(prefix, "fc"),
            return_bias=False,
        )

        with set_current_vllm_config(draft_vllm_config, prefix=prefix):
            # Module index 0 matches the checkpoint; the runtime prefix uses
            # the first index after the target layers, as other drafters do.
            self.layers = nn.ModuleList(
                [
                    Qwen4ExpEagle3DecoderLayer(
                        vllm_config=draft_vllm_config,
                        config=config,
                        prefix=f"{prefix}.layers.{start_layer_id}",
                    )
                ]
            )
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.norm_output = bool(config.norm_output)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.fc_norm is not None:
            chunks = hidden_states.chunk(self.num_aux_hidden_states, dim=-1)
            hidden_states = torch.cat(
                [norm(chunk) for norm, chunk in zip(self.fc_norm, chunks)],
                dim=-1,
            )
        return self.fc(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states, residual = self.layers[0](
            positions=positions,
            embeds=inputs_embeds,
            hidden_states=hidden_states,
        )
        hidden_states, hidden_prenorm = self.norm(hidden_states, residual)
        # Sample from the normed state; norm_output also feeds it forward.
        return hidden_states, hidden_states if self.norm_output else hidden_prenorm


class Qwen4ExpEagle3ForCausalLM(
    LocalArgmaxMixin, nn.Module, SupportsEagle3, SupportsHCSP
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    # Stacks q/k/v into qkv_proj and the shared expert's gate/up projections.
    hf_to_vllm_mapper = Qwen3NextModel.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        config = speculative_config.draft_model_config.hf_config
        _validate(config, vllm_config)
        self.config = config

        num_target_layers = vllm_config.model_config.get_total_num_hidden_layers()
        draft_vllm_config = _make_draft_vllm_config(vllm_config, num_target_layers)
        self.model = Qwen4ExpEagle3Model(
            draft_vllm_config=draft_vllm_config,
            config=config,
            start_layer_id=num_target_layers,
            prefix=maybe_prefix(prefix, "model"),
        )

        self.draft_vocab_size = config.draft_vocab_size
        self.lm_head = ParallelLMHead(
            self.draft_vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.draft_vocab_size)
        self.vocab_size = vllm_config.model_config.get_vocab_size()
        self.draft_id_to_target_id: nn.Parameter | None = None
        if self.draft_vocab_size != self.vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        enable_qwen4_exp_low_latency_gemm(self, vllm_config.model_config.dtype)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(input_ids, positions, hidden_states, inputs_embeds)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None or logits is None:
            return logits
        targets = (
            torch.arange(self.draft_vocab_size, device=logits.device)
            + self.draft_id_to_target_id
        )
        full_logits = logits.new_full((logits.shape[0], self.vocab_size), -torch.inf)
        full_logits[:, targets] = logits
        return full_logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        has_embed_tokens = has_lm_head = False

        def remap_weights():
            nonlocal has_embed_tokens, has_lm_head
            for name, weight in weights:
                leaf = name.rsplit(".", 1)[-1]
                if leaf == "t2d":
                    continue
                if leaf == "d2t":
                    name = "draft_id_to_target_id"
                elif not name.startswith(("model.", "lm_head.")):
                    name = f"model.{name}"
                has_embed_tokens |= "embed_tokens" in name
                has_lm_head |= name.startswith("lm_head.")
                name = name.replace(".mlp.shared_experts.", ".mlp.shared_expert.")
                # Automodel stores grouped experts as [E, H, 2I] (gate | up)
                # and [E, I, H]; FusedMoE's fused path takes [E, 2I, H] and
                # [E, H, I].
                if name.endswith(".mlp.experts.gate_and_up_projs"):
                    name = name.removesuffix("gate_and_up_projs") + "gate_up_proj"
                    weight = weight.transpose(-1, -2)
                elif name.endswith(".mlp.experts.down_projs"):
                    name = name.removesuffix("down_projs") + "down_proj"
                    weight = weight.transpose(-1, -2)
                process_eagle_weight(self, name)
                yield name, weight

        mapper = self.hf_to_vllm_mapper
        if self.draft_id_to_target_id is None:
            mapper = mapper | WeightsMapper(
                orig_to_new_substr={"draft_id_to_target_id": None}
            )
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        loaded = loader.load_weights(remap_weights(), mapper=mapper)
        # A drafter without its own embedding or head shares the target's
        # (see eagle/utils.py), which replaces these parameters after loading.
        if not has_embed_tokens:
            loaded.add("model.embed_tokens.weight")
        if not has_lm_head:
            loaded.add("lm_head.weight")
        return loaded


__all__ = ["Qwen4ExpEagle3ForCausalLM"]
