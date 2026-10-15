# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp GDN and PLE short-conv backends that admit adaptive verification.

Adaptive verification trims verification requests on device, so the CPU query
lengths of those requests become an even split of the true ones while the CPU
totals stay exact. The GDN and PLE spec-decode paths already read per-request
offsets from the device ``query_start_loc``; these subclasses declare that, and
keep every decode row on the spec-decode kernels so the single kernel path a
varlen decode graph captures matches every batch that replays it.

They are scoped to the NVIDIA Qwen4Exp layers, whose forward paths were checked
against device-decided lengths; the shared base backends stay opted out.
"""

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import CommonAttentionMetadata, max_decode_query_len
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionBackend,
    PleShortConvAttentionMetadata,
    PleShortConvAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec


def _uses_adaptive_verification(vllm_config: VllmConfig) -> bool:
    spec_config = vllm_config.speculative_config
    return spec_config is not None and spec_config.enable_adaptive_verification


def promote_draftless_decodes(
    common_attn_metadata: CommonAttentionMetadata,
    num_decode_draft_tokens_cpu: torch.Tensor | None,
) -> torch.Tensor | None:
    """Mark one-token decode rows as spec decodes carrying zero drafts.

    The hybrid model state marks a decode row without scheduled drafts with -1,
    which routes it to the plain decode kernels and, next to spec rows, makes the
    batch mixed. A varlen decode graph replays any decode batch, so every decode
    row has to take the spec-decode kernels it was captured with. A zero-draft
    spec row is equivalent to a plain decode: it reads the state at its accepted
    position and writes back one token.
    """
    if num_decode_draft_tokens_cpu is None:
        return None
    m = common_attn_metadata
    num_reqs = m.num_reqs
    query_lens_cpu = torch.diff(m.query_start_loc_cpu[: num_reqs + 1])
    draftless_decode = (num_decode_draft_tokens_cpu[:num_reqs] < 0) & (
        query_lens_cpu == 1
    )
    if m.is_prefilling is not None:
        draftless_decode &= ~m.is_prefilling[:num_reqs].cpu()
    if not bool(draftless_decode.any()):
        return num_decode_draft_tokens_cpu
    promoted = num_decode_draft_tokens_cpu.clone()
    promoted[:num_reqs].masked_fill_(draftless_decode, 0)
    return promoted


class Qwen4ExpGDNAttentionMetadataBuilder(GDNAttentionMetadataBuilder):
    adaptive_verification: bool = False

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # The base enables block-table reuse for its exact type only, to keep
        # out subclasses with per-group metadata; this one adds none.
        if (
            not self.supports_update_block_table
            and vllm_config.use_v2_model_runner
            and device.type == "cuda"
        ):
            self.supports_update_block_table = True
            self.mamba_aligned_state_indices = None
        self.adaptive_verification = _uses_adaptive_verification(vllm_config)
        self.keep_draftless_spec_decodes = self.adaptive_verification

    @classmethod
    def get_varlen_cudagraph_max_query_len(
        cls: type["Qwen4ExpGDNAttentionMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> int | None:
        # Spec rows index the conv/recurrent state through the device cu_seqlens
        # within a fixed num_spec + 1 window, so the graph replays any 1..k+1 mix.
        return max_decode_query_len(vllm_config)

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        if self.adaptive_verification:
            num_decode_draft_tokens_cpu = promote_draftless_decodes(
                common_attn_metadata, num_decode_draft_tokens_cpu
            )
        return super().build(
            common_prefix_len,
            common_attn_metadata,
            num_accepted_tokens,
            num_decode_draft_tokens_cpu,
            fast_build,
        )


class Qwen4ExpGDNAttentionBackend(GDNAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[Qwen4ExpGDNAttentionMetadataBuilder]:
        return Qwen4ExpGDNAttentionMetadataBuilder

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        # Spec rows take their offsets from the device query_start_loc; only
        # totals and request classes come from the CPU, and those stay exact.
        return True


class Qwen4ExpPleShortConvAttentionMetadataBuilder(
    PleShortConvAttentionMetadataBuilder
):
    adaptive_verification: bool = False

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.adaptive_verification = _uses_adaptive_verification(vllm_config)

    @classmethod
    def get_varlen_cudagraph_max_query_len(
        cls: type["Qwen4ExpPleShortConvAttentionMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> int | None:
        # ple_conv locates each token's request by searching the device offsets
        # and writes back only that request's tokens.
        return max_decode_query_len(vllm_config)

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> PleShortConvAttentionMetadata:
        if self.adaptive_verification:
            num_decode_draft_tokens_cpu = promote_draftless_decodes(
                common_attn_metadata, num_decode_draft_tokens_cpu
            )
        return super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            **kwargs,
        )


class Qwen4ExpPleShortConvAttentionBackend(PleShortConvAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[Qwen4ExpPleShortConvAttentionMetadataBuilder]:
        return Qwen4ExpPleShortConvAttentionMetadataBuilder

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        # Same contract as Qwen4ExpGDNAttentionBackend.
        return True
