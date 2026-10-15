# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adaptive-verification contract of the Qwen4Exp GDN and PLE backends."""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig, VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.models.qwen4_exp.nvidia.linear_attn_backends import (
    Qwen4ExpGDNAttentionBackend,
    Qwen4ExpGDNAttentionMetadataBuilder,
    Qwen4ExpPleShortConvAttentionBackend,
    Qwen4ExpPleShortConvAttentionMetadataBuilder,
    promote_draftless_decodes,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend
from vllm.v1.attention.backends.short_conv_attn import PleShortConvAttentionBackend
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")
NUM_SPEC = 3


def test_only_the_qwen4_exp_backends_accept_device_query_lens() -> None:
    for backend in (Qwen4ExpGDNAttentionBackend, Qwen4ExpPleShortConvAttentionBackend):
        assert backend.supports_device_cpu_query_lens_mismatch()
    # The shared bases keep the SSM opt-out for every other model.
    for backend in (GDNAttentionBackend, PleShortConvAttentionBackend):
        assert not backend.supports_device_cpu_query_lens_mismatch()
    assert (
        Qwen4ExpGDNAttentionBackend.get_builder_cls()
        is Qwen4ExpGDNAttentionMetadataBuilder
    )
    assert (
        Qwen4ExpPleShortConvAttentionBackend.get_builder_cls()
        is Qwen4ExpPleShortConvAttentionMetadataBuilder
    )
    # Names and layer bookkeeping stay those of the base backends.
    assert Qwen4ExpGDNAttentionBackend.get_name() == GDNAttentionBackend.get_name()
    assert (
        Qwen4ExpPleShortConvAttentionBackend.get_name()
        == PleShortConvAttentionBackend.get_name()
    )


@pytest.mark.parametrize(
    "builder_cls",
    [Qwen4ExpGDNAttentionMetadataBuilder, Qwen4ExpPleShortConvAttentionMetadataBuilder],
)
def test_varlen_cudagraph_bound_is_one_verification_request(builder_cls) -> None:
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=NUM_SPEC, parallel_drafting=False
        ),
        use_v2_model_runner=True,
    )
    bound = builder_cls.get_varlen_cudagraph_max_query_len(vllm_config, None)
    assert bound == NUM_SPEC + 1


def _common(query_lens: list[int], is_prefilling: list[bool] | None):
    return SimpleNamespace(
        num_reqs=len(query_lens),
        query_start_loc_cpu=torch.tensor(
            [0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32
        ),
        is_prefilling=None
        if is_prefilling is None
        else torch.tensor(is_prefilling, dtype=torch.bool),
    )


def test_promote_draftless_decodes() -> None:
    # decode without drafts, verification, prefill chunk, graph padding
    common = _common([1, 3, 5, 0], [False, False, True, False])
    drafts = torch.tensor([-1, 2, -1, -1], dtype=torch.int32)

    promoted = promote_draftless_decodes(common, drafts)

    assert promoted is not None
    assert promoted.tolist() == [0, 2, -1, -1]
    # The caller's tensor is left alone.
    assert drafts.tolist() == [-1, 2, -1, -1]


def test_promote_draftless_decodes_keeps_one_token_prefill_chunks() -> None:
    common = _common([1, 1], [True, False])
    promoted = promote_draftless_decodes(
        common, torch.tensor([-1, -1], dtype=torch.int32)
    )
    assert promoted is not None
    assert promoted.tolist() == [-1, 0]


def test_promote_draftless_decodes_is_a_noop_without_draftless_rows() -> None:
    drafts = torch.tensor([3, 0], dtype=torch.int32)
    assert promote_draftless_decodes(_common([4, 1], [False, False]), drafts) is drafts
    assert promote_draftless_decodes(_common([1], [False]), None) is None


def _gdn_builder(
    adaptive: bool, full_cuda_graph: bool = False
) -> Qwen4ExpGDNAttentionMetadataBuilder:
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B", block_size=BLOCK_SIZE
    )
    vllm_config.compilation_config.cudagraph_mode = (
        CUDAGraphMode.FULL_AND_PIECEWISE if full_cuda_graph else CUDAGraphMode.NONE
    )
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=NUM_SPEC
    )
    vllm_config.speculative_config.enable_adaptive_verification = adaptive
    return Qwen4ExpGDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE, shapes=((16, 64),), dtypes=(torch.float16,)
        ),
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def _gdn_build(builder, query_lens: list[int], drafts: list[int]):
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[q + 32 for q in query_lens], query_lens=query_lens),
        BLOCK_SIZE,
        DEVICE,
    ).replace(is_prefilling=torch.zeros(len(query_lens), dtype=torch.bool))
    return builder.build(
        0,
        common,
        num_accepted_tokens=torch.ones(len(query_lens), dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.tensor(drafts, dtype=torch.int32),
    )


@pytest.mark.parametrize("adaptive", [False, True])
def test_gdn_draftless_decode_next_to_verification(adaptive: bool) -> None:
    # The device has trimmed the middle row to its bonus token, so the CPU sees
    # an even split; the hybrid state marks the draftless row with -1.
    meta = _gdn_build(_gdn_builder(adaptive), [1, 4, 1], [-1, NUM_SPEC, -1])

    if adaptive:
        # Pure spec batch: the shape a varlen decode graph captured.
        assert meta.num_spec_decodes == 3
        assert meta.num_prefills == 0
    else:
        # Unchanged: draftless decodes are reclassified as prefills.
        assert meta.num_spec_decodes == 1
        assert meta.num_prefills == 2
    assert meta.num_decodes == 0


@pytest.mark.parametrize("adaptive", [False, True])
def test_gdn_capture_of_one_token_rows(adaptive: bool) -> None:
    # A varlen decode graph is captured with one token per request whenever
    # num_tokens <= max_num_reqs; it must still record the spec kernels.
    builder = _gdn_builder(adaptive, full_cuda_graph=True)
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[1] * 4, query_lens=[1] * 4), BLOCK_SIZE, DEVICE
    ).replace(is_prefilling=torch.zeros(4, dtype=torch.bool))

    meta = builder.build_for_cudagraph_capture(common)

    if adaptive:
        assert meta.num_spec_decodes == 4
        assert meta.num_decodes == 0
    else:
        assert meta.num_spec_decodes == 0
        assert meta.num_decodes == 4


@pytest.mark.parametrize("adaptive", [False, True])
def test_ple_draftless_decode_next_to_verification(adaptive: bool) -> None:
    config = VllmConfig()
    config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=NUM_SPEC
    )
    config.speculative_config.enable_adaptive_verification = adaptive
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    config.compilation_config.max_cudagraph_capture_size = 32
    builder = Qwen4ExpPleShortConvAttentionMetadataBuilder(
        MambaSpec(block_size=16, shapes=((16, 64),), dtypes=(torch.float32,)),
        ["ple"],
        config,
        DEVICE,
    )
    query_lens = [1, 4, 1]
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[q + 16 for q in query_lens], query_lens=query_lens),
        16,
        DEVICE,
        arange_block_indices=True,
    )
    common.is_prefilling = torch.zeros(3, dtype=torch.bool)

    meta = builder.build(
        0,
        common,
        num_accepted_tokens=torch.ones(3, dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.tensor([-1, NUM_SPEC, -1], dtype=torch.int32),
    )

    if adaptive:
        assert meta.num_spec_decodes == 3
        assert meta.num_decodes == 0
        # Pure spec rows are staged into the persistent graph buffers.
        assert (
            meta.spec_query_start_loc.data_ptr()
            == builder.spec_query_start_loc.data_ptr()
        )
        assert meta.spec_query_start_loc[:4].tolist() == [0, 1, 5, 6]
    else:
        assert meta.num_spec_decodes == 1
        assert meta.num_decodes == 2
