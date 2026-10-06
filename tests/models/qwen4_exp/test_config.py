# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen4ExpConfig, Qwen4ExpTextConfig

from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models.config import (
    Qwen3_5ForConditionalGenerationConfig,
    Qwen4ExpForConditionalGenerationConfig,
)
from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

from ...utils import spawn_new_process_for_each_test


def _text_config(**kwargs) -> Qwen4ExpTextConfig:
    values = {
        "vocab_size": 64,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        # `Qwen4ExpTextConfig` requires an EOS token whenever PLE is enabled.
        "eos_token_id": 1,
        "hc_count": 2,
        "hc_lowrank": 4,
        "ple_layer_ids": [1],
        "mtp_num_hidden_layers": 1,
        "mtp": {"hybrid": True},
    }
    values.update(kwargs)
    return Qwen4ExpTextConfig(**values)


def test_qwen4_exp_mtp_returns_sample_and_multi_streams() -> None:
    from vllm.models.qwen4_exp.nvidia.mtp import (
        Qwen4ExpMultiTokenPredictor,
    )

    model = object.__new__(Qwen4ExpMultiTokenPredictor)
    torch.nn.Module.__init__(model)
    model.hc_count = 2
    model.hidden_size = 4
    model.num_mtp_layers = 1
    model.layers = [
        lambda **kwargs: (
            kwargs["hidden_states"],
            kwargs["hidden_states"],
            torch.zeros(kwargs["hidden_states"].shape[0], 2),
        ),
    ]
    model.hyper_connection_mixer = SimpleNamespace(
        combine_and_mix=lambda hidden_states, block_output, injection: (
            hidden_states,
            hidden_states.unflatten(-1, (2, 4)).mean(-2),
            None,
        ),
    )
    multi_hidden = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=True)

    with patch(
        "vllm.models.qwen4_exp.nvidia.mtp.get_pp_group",
        return_value=pp_group,
    ):
        sample_hidden, returned_multi_hidden = model.forward(
            input_ids=None,
            positions=torch.arange(2),
            intermediate_tensors={"hidden_states": multi_hidden},
        )

    torch.testing.assert_close(
        sample_hidden,
        multi_hidden.unflatten(-1, (2, 4)).mean(dim=-2),
    )
    assert returned_multi_hidden is multi_hidden


@spawn_new_process_for_each_test
@pytest.mark.parametrize("backend", ["amd", "nvidia"])
def test_qwen4_exp_mtp_remaps_mixed_precision_layer_indices(backend: str) -> None:
    mtp_module = import_module(f"vllm.models.qwen4_exp.{backend}.mtp")

    quantized_layers = {
        "model.language_model.layers.0.mlp.experts": {"quant_algo": "NVFP4"},
        "mtp.layers.0.mlp.experts": {
            "quant_algo": "FP8_BLOCK_SCALES",
            "group_size": 128,
        },
    }

    assert mtp_module._remap_quantized_layers(quantized_layers, 48) == {
        "model.language_model.layers.0.mlp.experts": {"quant_algo": "NVFP4"},
        "mtp.layers.48.mlp.experts": {
            "quant_algo": "FP8_BLOCK_SCALES",
            "group_size": 128,
        },
    }


@pytest.mark.parametrize("wrapped_config", [False, True])
def test_qwen4_exp_mtp_override_sets_draft_config(
    wrapped_config: bool,
) -> None:
    text_config = _text_config(
        architectures=["Qwen4ExpForCausalLM"],
        index_share_for_mtp_iteration=True,
    )
    config = (
        Qwen4ExpConfig(
            architectures=["Qwen4ExpForConditionalGeneration"],
            text_config=text_config,
        )
        if wrapped_config
        else text_config
    )

    draft_config = SpeculativeConfig.hf_config_override(config)

    assert draft_config.index_share_for_mtp_iteration is True
    assert draft_config.model_type == "qwen4_exp_mtp"
    assert draft_config.architectures == ["Qwen4ExpMTP"]
    assert draft_config.hc_mult == 2
    assert draft_config.n_predict == 1


@pytest.mark.parametrize("ple_layer_ids", [[1], []])
def test_qwen4_exp_rejects_pipeline_parallel_only_with_ple(ple_layer_ids) -> None:
    """PLE needs raw input_ids, which non-first pipeline ranks never see. The
    rest of the architecture is PP-capable, so the refusal must be conditional
    -- and must land before the engine spends time loading weights."""
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=_text_config(ple_layer_ids=ple_layer_ids),
            multimodal_config=None,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2, enable_dbo=False, ubatch_size=1
        ),
        speculative_config=None,
    )
    with patch.object(
        Qwen3_5ForConditionalGenerationConfig, "verify_and_update_config"
    ):
        if ple_layer_ids:
            with pytest.raises(NotImplementedError, match="pipeline_parallel_size=1"):
                Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(
                    vllm_config
                )
        else:
            Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(vllm_config)


@pytest.mark.parametrize("method", ["dflash", "mtp", "ngram", "eagle3"])
def test_qwen4_exp_speculative_methods(method):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=_text_config(), multimodal_config=None
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, enable_dbo=False, ubatch_size=1
        ),
        speculative_config=SimpleNamespace(method=method),
    )
    with patch.object(
        Qwen3_5ForConditionalGenerationConfig, "verify_and_update_config"
    ):
        if method == "eagle3":
            with pytest.raises(NotImplementedError, match="supports DFlash"):
                Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(config)
        else:
            Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(config)


class _CaptureMixer:
    """Nonuniform learned read/write surrogate; kernel parity is in test_hc_ops."""

    def __init__(self):
        self.read_weight = torch.randn(8, 8)
        self.inject_weight = torch.randn(8, 2)
        self.calls = 0

    def mix(self, hidden):
        self.calls += 1
        gates = (hidden @ self.read_weight).sigmoid().unflatten(-1, (2, 4))
        feature = (hidden.unflatten(-1, (2, 4)) * gates).mean(-2)
        return hidden, feature, hidden @ self.inject_weight

    def combine(self, hidden, output, injection):
        if output is None:
            return hidden
        return (
            hidden.unflatten(-1, (2, 4))
            + output.unsqueeze(-2) * (2 * (injection / 2).sigmoid()).unsqueeze(-1)
        ).flatten(-2)

    def combine_and_mix(self, hidden, output, injection):
        return self.mix(self.combine(hidden, output, injection))


@spawn_new_process_for_each_test
@pytest.mark.parametrize("backend", ["amd", "nvidia"])
def test_qwen4_exp_capture_uses_next_learned_readout_before_ple(backend):
    """Capture is H-wide, ordered, and does not change actor/MTP outputs."""
    module = import_module(f"vllm.models.qwen4_exp.{backend}.model")
    torch.manual_seed(17)
    model = object.__new__(module.Qwen4ExpModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(hc_count=2, hidden_size=4)
    model.start_layer, model.end_layer = 0, 3
    model._mtp_hidden_buffer = torch.empty(5, 8)
    model.hyper_connection_mixer = _CaptureMixer()
    model._start_layer_ple_prefetch = lambda *args: None
    layers = []
    for i in range(3):
        layer = object.__new__(module.Qwen4ExpDecoderLayer)
        torch.nn.Module.__init__(layer)
        layer.layer_type = "linear_attention"
        layer.attn_hyper_connection = _CaptureMixer()
        layer.mlp_hyper_connection = _CaptureMixer()
        layer.linear_attn = lambda hidden_states: hidden_states * 0.3
        layer.mlp = lambda hidden: hidden * -0.2
        # AMD's PLE returns the delta; NVIDIA's PLE also applies the addition.
        layer.ple = (
            (
                (lambda hidden, *args: hidden + 2)
                if backend == "nvidia"
                else (lambda hidden, *args: torch.full_like(hidden, 2))
            )
            if i == 1
            else None
        )
        layers.append(layer)
    model.layers = torch.nn.ModuleList(layers)
    pp = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    embedded = torch.randn(5, 4)
    inputs = dict(
        input_ids=torch.arange(5),
        positions=torch.arange(5),
        inputs_embeds=embedded,
        query_start_loc=torch.tensor([0, 5]),
        ngram_context=torch.zeros(1, 2),
    )
    with patch.object(module, "get_pp_group", return_value=pp):
        plain = model.forward(**inputs)
        mtp = model._mtp_hidden_buffer.clone()
        wrapper = object.__new__(module.Qwen4ExpForCausalLM)
        torch.nn.Module.__init__(wrapper)
        wrapper.model = model
        wrapper.set_aux_hidden_state_layers((0, 1, 2, 3))
        captured_output, captured = model.forward(**inputs)

    torch.testing.assert_close(captured_output, plain, rtol=0, atol=0)
    torch.testing.assert_close(model._mtp_hidden_buffer, mtp, rtol=0, atol=0)
    torch.testing.assert_close(captured[0], embedded)
    assert all(value.shape == (5, 4) for value in captured)
    assert [layer.attn_hyper_connection.calls for layer in layers] == [2, 3, 2]
    # Independently step the real layers, reading each completed state with
    # the *next* mixer before its PLE, rather than averaging the HC streams.
    hidden = embedded.repeat(1, 2)
    for index, layer in enumerate(layers):
        hidden, output, injection = layer(
            hidden,
            None,
            None,
            inputs["positions"],
            input_ids=inputs["input_ids"],
            query_start_loc=inputs["query_start_loc"],
            ngram_context=inputs["ngram_context"],
        )
        hidden = layer.mlp_hyper_connection.combine(hidden, output, injection)
        mixer = (
            layers[index + 1].attn_hyper_connection
            if index < 2
            else model.hyper_connection_mixer
        )
        expected = mixer.mix(hidden)[1]
        torch.testing.assert_close(captured[index + 1], expected)
        assert not torch.allclose(expected, hidden.unflatten(-1, (2, 4)).mean(-2))
    assert captured[-1] is captured_output


def test_qwen4_exp_model_state_prepares_ngram_context() -> None:
    model_state = object.__new__(Qwen4ExpModelState)
    model_state.uses_ngram_embedding = True
    model_state.ngram_context_len = 3
    model_state.ngram_eos_token_id = 99
    model_state.ngram_context = torch.empty((8, 3), dtype=torch.int32)
    model_state.ngram_context_offsets = torch.arange(-3, 0, dtype=torch.int64)
    model_state.ple_query_start_loc = torch.empty(9, dtype=torch.int32)

    input_batch = SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=3,
        idx_mapping=torch.tensor([1, 0]),
        query_start_loc=torch.tensor([0, 2, 3, 3], dtype=torch.int32),
    )
    req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([3, 1])),
        all_token_ids=SimpleNamespace(
            gpu=torch.tensor([[1, 2, 3, 4], [20, 21, 22, 23]], dtype=torch.int32)
        ),
    )

    with patch.object(MambaHybridModelState, "prepare_inputs", return_value={}):
        model_inputs = model_state.prepare_inputs(input_batch, req_states)

    expected_query_start_loc = torch.full((9,), 3, dtype=torch.int32)
    expected_query_start_loc[0] = 0
    expected_query_start_loc[1] = 2
    torch.testing.assert_close(
        model_inputs["query_start_loc"], expected_query_start_loc
    )
    expected_context = torch.full((8, 3), 99, dtype=torch.int32)
    expected_context[:2] = torch.tensor([[99, 99, 20], [1, 2, 3]])
    torch.testing.assert_close(model_inputs["ngram_context"], expected_context)

    # Retain the views to detect reallocations as the request layout changes.
    query_start_loc = model_inputs["query_start_loc"]
    ngram_context = model_inputs["ngram_context"]
    input_batch.num_reqs = 1
    input_batch.num_reqs_after_padding = 1
    input_batch.idx_mapping = torch.tensor([0])
    input_batch.query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    with patch.object(MambaHybridModelState, "prepare_inputs", return_value={}):
        model_inputs = model_state.prepare_inputs(input_batch, req_states)

    expected_query_start_loc.fill_(3)
    expected_query_start_loc[0] = 0
    torch.testing.assert_close(
        model_inputs["query_start_loc"], expected_query_start_loc
    )
    expected_context.fill_(99)
    expected_context[0] = torch.tensor([1, 2, 3])
    torch.testing.assert_close(model_inputs["ngram_context"], expected_context)
    assert model_inputs["query_start_loc"].data_ptr() == query_start_loc.data_ptr()
    assert model_inputs["ngram_context"].data_ptr() == ngram_context.data_ptr()


def test_qwen4_exp_model_state_prepares_stable_dummy_ngram_inputs() -> None:
    model_state = object.__new__(Qwen4ExpModelState)
    model_state.uses_ngram_embedding = True
    model_state.ngram_eos_token_id = 99
    model_state.ngram_context = torch.empty((8, 3), dtype=torch.int32)
    model_state.ple_query_start_loc = torch.empty(9, dtype=torch.int32)

    with patch.object(MambaHybridModelState, "prepare_dummy_inputs", return_value={}):
        first = model_state.prepare_dummy_inputs(num_reqs=3, num_tokens=4)
        # Dummy runs establish the addresses used during CUDA graph capture.
        query_start_loc_ptr = first["query_start_loc"].data_ptr()
        ngram_context_ptr = first["ngram_context"].data_ptr()
        second = model_state.prepare_dummy_inputs(num_reqs=3, num_tokens=4)

    expected_query_start_loc = torch.full((9,), 4, dtype=torch.int32)
    expected_query_start_loc[:4] = torch.tensor([0, 1, 2, 4], dtype=torch.int32)
    torch.testing.assert_close(second["query_start_loc"], expected_query_start_loc)
    torch.testing.assert_close(
        second["ngram_context"], torch.full((8, 3), 99, dtype=torch.int32)
    )
    assert second["query_start_loc"].data_ptr() == query_start_loc_ptr
    assert second["ngram_context"].data_ptr() == ngram_context_ptr
