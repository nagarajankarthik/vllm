# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import Qwen4ExpTextConfig

from vllm.config.speculative import SpeculativeConfig
from vllm.transformers_utils.configs.qwen_sparse_eagle3 import (
    QwenSparseEagle3DraftConfig,
)

_ARCHITECTURE = "QwenSparseEagle3DraftModel"
_MROPE_PARAMETERS = {
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
    "rope_type": "default",
    "rope_theta": 10_000_000,
    "partial_rotary_factor": 0.25,
}


def _explicit_config(**kwargs) -> QwenSparseEagle3DraftConfig:
    """An Automodel explicit draft config.json, as the training YAML writes it."""
    values = {
        "architectures": [_ARCHITECTURE],
        "rope_parameters": dict(_MROPE_PARAMETERS),
        "fc_norm": True,
        "norm_output": True,
        "aux_layer_ids": [1, 33, 57],
    }
    values.update(kwargs)
    return QwenSparseEagle3DraftConfig(**values)


def test_explicit_draft_config_is_normalized() -> None:
    hf_config = SpeculativeConfig.hf_config_override(_explicit_config())

    assert hf_config.architectures == [_ARCHITECTURE]
    assert hf_config.num_hidden_layers == 1
    assert "mrope_section" not in hf_config.rope_parameters
    assert "mrope_interleaved" not in hf_config.rope_parameters
    assert hf_config.rope_parameters["partial_rotary_factor"] == 0.25
    # Automodel id i (output of decoder layer i) is vLLM capture id i + 1.
    assert hf_config.eagle_aux_hidden_state_layer_ids == [2, 34, 58]
    assert hf_config.draft_vocab_size == hf_config.vocab_size
    assert hf_config.target_hidden_size == hf_config.hidden_size
    assert getattr(hf_config, "hc_mult", 1) == 1


def test_explicit_aux_layer_ids_take_precedence() -> None:
    hf_config = SpeculativeConfig.hf_config_override(
        _explicit_config(eagle_aux_hidden_state_layer_ids=[3, 35, 59])
    )

    assert hf_config.eagle_aux_hidden_state_layer_ids == [3, 35, 59]


@pytest.mark.parametrize(
    "overrides",
    [
        {"aux_layer_ids": None},
        {"aux_layer_ids": [33, 1, 57]},
        {"aux_layer_ids": [1, 33]},
        {"eagle_aux_hidden_state_layer_ids": [2, 34, 34]},
    ],
)
def test_invalid_aux_layer_ids_are_rejected(overrides) -> None:
    with pytest.raises(ValueError, match="aux"):
        SpeculativeConfig.hf_config_override(_explicit_config(**overrides))


def test_target_derived_draft_config_is_not_rewritten_to_mtp() -> None:
    hf_config = Qwen4ExpTextConfig(
        architectures=[_ARCHITECTURE],
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        num_experts=4,
        num_experts_per_tok=2,
        eos_token_id=1,
        hc_count=2,
        hc_lowrank=4,
        ple_layer_ids=[1],
        mtp_num_hidden_layers=1,
        rope_parameters=dict(_MROPE_PARAMETERS),
    )
    hf_config.aux_layer_ids = [0]
    hf_config.num_aux_hidden_states = 1

    hf_config = SpeculativeConfig.hf_config_override(hf_config)

    assert hf_config.architectures == [_ARCHITECTURE]
    assert getattr(hf_config, "hc_mult", 1) == 1
    assert hf_config.num_hidden_layers == 1
    assert "mrope_section" not in hf_config.rope_parameters
    assert hf_config.eagle_aux_hidden_state_layer_ids == [1]
