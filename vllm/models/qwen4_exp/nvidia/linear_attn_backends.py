# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp GDN and PLE short-conv backends that admit adaptive verification.

Adaptive verification trims verification requests on device, so the CPU query
lengths of those requests become an even split of the true ones while the CPU
totals stay exact. The GDN and PLE spec-decode paths already read per-request
offsets from the device ``query_start_loc``; these subclasses declare that.
Every decode row, drafted or not, reaches them as a spec row, so the spec
kernels a varlen decode graph captures match every batch that replays it.

They are scoped to the NVIDIA Qwen4Exp layers, whose forward paths were checked
against device-decided lengths; the shared base backends stay opted out.
"""

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import max_decode_query_len
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionBackend,
    PleShortConvAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec


class Qwen4ExpGDNAttentionMetadataBuilder(GDNAttentionMetadataBuilder):
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

    @classmethod
    def get_varlen_cudagraph_max_query_len(
        cls: type["Qwen4ExpGDNAttentionMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> int | None:
        # Spec rows index the conv/recurrent state through the device cu_seqlens
        # within a fixed num_spec + 1 window, so the graph replays any 1..k+1 mix.
        return max_decode_query_len(vllm_config)


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
    @classmethod
    def get_varlen_cudagraph_max_query_len(
        cls: type["Qwen4ExpPleShortConvAttentionMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> int | None:
        # ple_conv locates each token's request by searching the device offsets
        # and writes back only that request's tokens.
        return max_decode_query_len(vllm_config)


class Qwen4ExpPleShortConvAttentionBackend(PleShortConvAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[Qwen4ExpPleShortConvAttentionMetadataBuilder]:
        return Qwen4ExpPleShortConvAttentionMetadataBuilder

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        # Same contract as Qwen4ExpGDNAttentionBackend.
        return True
