# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.models.xpress_head import XPressHead
from vllm.v1.worker.gpu.spec_decode.xpress.speculator import XPressSpeculator


def _head(dtype=torch.float32):
    torch.manual_seed(19)
    head = XPressHead(16, 32, 5, 8).to(dtype)
    with torch.no_grad():
        head.mix["L"].normal_(std=0.3)
    head.prepare_for_inference()
    return head


def _reference(head, hidden, previous):
    global_hidden = head.down_g(hidden.mean(1, keepdim=True)).expand(
        -1, hidden.shape[1], -1
    )
    x = head.in_proj(
        torch.cat((head.down_h(hidden), global_hidden, head.w1(previous)), -1)
    )
    size = x.shape[1]
    mixing = head.mix["L"][:, :size, :size].tril() + torch.eye(size, dtype=x.dtype)
    x = torch.bmm(mixing, x.permute(2, 1, 0)).permute(2, 1, 0)
    return head.w2(
        x
        + head.mlp["down_proj"](
            F.silu(head.mlp["gate_proj"](x)) * head.mlp["up_proj"](x)
        )
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_exported_head_and_greedy_fixed_point(dtype):
    """Folded serving math equals raw training weights, including anchor/global mean."""
    head = _head(dtype)
    hidden = torch.randn(2, 5, 16, dtype=dtype)
    base = torch.randn(2, 4, 32, dtype=dtype) * 0.02
    anchor, previous = torch.tensor([1, 2]), torch.tensor([8, 9])
    serial = torch.zeros(2, 4, dtype=torch.long)
    for step in range(4):
        shifted = torch.cat((previous[:, None], anchor[:, None], serial[:, :-1]), -1)
        bias = _reference(head, hidden, shifted)
        torch.testing.assert_close(
            head(head.prepare_hidden(hidden), shifted), bias, atol=0, rtol=0
        )
        serial[:, step] = (base.float() + bias[:, 1:].float())[:, step].argmax(-1)
    actual = head.refine(hidden, base, anchor, previous, passes=4)
    torch.testing.assert_close(actual, serial)
    assert (actual != base.argmax(-1)).any()
    restored = _head(dtype)
    restored.load_state_dict(head.state_dict(), strict=True)
    restored.prepare_for_inference()
    torch.testing.assert_close(
        restored.refine(hidden, base, anchor, previous, passes=4), actual
    )


def test_predecessor_uses_accepted_context_and_not_neighbour_request():
    """Rejected guesses and an empty request cannot become the token before anchor."""
    spec = object.__new__(XPressSpeculator)
    spec.predecessor_ids = torch.full((5,), 31, dtype=torch.long)
    batch = SimpleNamespace(
        num_reqs=3,
        query_start_loc=torch.tensor([0, 3, 3, 6]),
        input_ids=torch.tensor([10, 11, 12, 20, 21, 22]),
    )
    spec.prepare_context_anchor(batch, torch.tensor([1, 0, 0]))
    assert spec.predecessor_ids.tolist() == [11, 0, 22, 0, 0]


def test_generation_keeps_anchor_hidden_and_padding_inert():
    """The shared DFlash runner gives the refiner its full block, not only masks."""
    head = _head()
    hidden = torch.randn(3 * 5, 16)
    projection = torch.randn(16, 32)
    spec = object.__new__(XPressSpeculator)
    spec.num_query_per_req, spec.num_speculative_steps = 5, 4
    spec.refinement_steps = 4
    spec.predecessor_ids = torch.tensor([6, 7, 0])
    spec.sample_idx_mapping = torch.tensor([0] * 4 + [2] * 4 + [-1] * 4)
    ids = torch.tensor([1, 31, 31, 31, 31, 2, 31, 31, 31, 31, 0, 0, 0, 0, 0])
    spec.input_buffers = SimpleNamespace(input_ids=ids)
    spec.draft_tokens = torch.full((3, 4), 31, dtype=torch.long)
    spec.model = SimpleNamespace(
        model=SimpleNamespace(xpress_head=head),
        compute_logits=lambda x: x @ projection,
    )
    spec._run_model = lambda *args: hidden
    spec._generate_draft(3, 15, None, None, None)
    full = hidden.view(3, 5, 16)
    expected = head.refine(
        full, full[:, 1:] @ projection, ids[::5], spec.predecessor_ids, 4
    )
    torch.testing.assert_close(spec.draft_tokens[:2], expected[:2])
    assert spec.draft_tokens[2].eq(0).all()


def test_unimplemented_sampling_is_rejected_before_model_loading():
    spec = SimpleNamespace(draft_sample_method="probabilistic")
    with pytest.raises(ValueError, match="greedy"):
        XPressSpeculator(SimpleNamespace(speculative_config=spec), torch.device("cpu"))
    spec.draft_sample_method, spec.enable_adaptive_verification = "greedy", True
    with pytest.raises(ValueError, match="adaptive verification"):
        XPressSpeculator(SimpleNamespace(speculative_config=spec), torch.device("cpu"))


@pytest.mark.parametrize("missing", [None, "head", "convolution"])
def test_checkpoint_load_keeps_refiner_and_convolutions(monkeypatch, missing):
    """Use the real loader, bypassing only GPU backbone construction/KV preparation."""
    from vllm.distributed import parallel_state
    from vllm.model_executor.models.qwen3_xpress import XPressForCausalLM, XPressModel

    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )
    # Loader-only fixture: bypass the abstract model interface and GPU setup.
    model = XPressForCausalLM.__new__(XPressForCausalLM)  # type: ignore[type-abstract]
    nn.Module.__init__(model)
    model.model = XPressModel.__new__(XPressModel)
    nn.Module.__init__(model.model)
    model.model.xpress_head = _head()
    layer = nn.Module()
    layer.attention_conv = nn.Linear(2, 2, bias=False)
    model.model.layers = nn.ModuleList([layer])
    model.model.use_aux_hidden_state = True
    model.model.has_separate_mask_embedding = False
    monkeypatch.setattr(model, "_read_mask_embedding", lambda: None)
    monkeypatch.setattr(model.model, "_build_fused_kv_buffers", lambda: None)
    weights = {key: value.clone() for key, value in model.model.state_dict().items()}
    if missing:
        key = (
            "xpress_head.w2.weight"
            if missing == "head"
            else "layers.0.attention_conv.weight"
        )
        del weights[key]
        with pytest.raises(ValueError, match="missing tensors"):
            model.load_weights(weights.items())
    else:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        model.load_weights(weights.items())
        for name, parameter in model.model.named_parameters():
            torch.testing.assert_close(parameter, weights[name], atol=0, rtol=0)
        raw = weights["xpress_head.mix.L"]
        torch.testing.assert_close(
            model.model.xpress_head.mixer, raw.tril() + torch.eye(5)
        )


def test_dflash_wrapper_retains_xpress_architecture():
    from transformers import Qwen3Config

    from vllm.config import VllmConfig
    from vllm.transformers_utils.configs.eagle import EAGLEConfig

    wrapped = EAGLEConfig(
        Qwen3Config(architectures=["XPressDraftModel"]), method="dflash"
    )
    assert wrapped.architectures == ["XPressDraftModel"]
    # Runner V1 must not silently skip the correction head.
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="dflash", draft_model_config=wrapped)
    )
    assert VllmConfig._is_dflash_candidate_draft(config)
