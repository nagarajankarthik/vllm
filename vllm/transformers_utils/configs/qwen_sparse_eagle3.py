# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config of the Qwen Sparse Attention (QSA) + MoE EAGLE-3 draft.

Automodel saves explicitly configured ``QwenSparseEagle3DraftModel`` drafts
with ``model_type="qwen_sparse_eagle3_draft"``. The draft is one decoder layer
with Qwen4Exp gated QSA attention (plus its indexer) and a Qwen3-Next-style
MoE. ``SpeculativeConfig.hf_config_override`` normalizes the loaded config for
serving (single layer, 1D RoPE, defaults for keys training does not record).
"""

from typing import Any

from transformers.configuration_utils import PreTrainedConfig


class QwenSparseEagle3DraftConfig(PreTrainedConfig):
    model_type = "qwen_sparse_eagle3_draft"
    # Training copies the target's interleaved MRoPE layout; it reduces to 1D
    # RoPE for text positions and is stripped before the draft is built.
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    def __init__(
        self,
        *,
        vocab_size: int = 248320,
        hidden_size: int = 4096,
        num_hidden_layers: int = 1,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 2,
        head_dim: int = 256,
        attention_bias: bool = False,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 262144,
        rope_parameters: dict[str, Any] | None = None,
        initializer_range: float = 0.02,
        indexer_n_heads: int = 4,
        indexer_kv_heads: int = 1,
        indexer_head_dim: int = 128,
        indexer_budget: int = 2048,
        indexer_compress_ratio: int = 4,
        num_experts: int = 512,
        num_experts_per_tok: int = 10,
        moe_intermediate_size: int = 1024,
        shared_expert_intermediate_size: int = 1024,
        norm_topk_prob: bool = True,
        num_aux_hidden_states: int = 3,
        draft_vocab_size: int | None = None,
        target_hidden_size: int | None = None,
        fc_norm: bool = False,
        norm_output: bool = False,
        parallel_drafting: bool = False,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.attention_bias = attention_bias
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_parameters = dict(
            rope_parameters
            or {
                "rope_type": "default",
                "rope_theta": 10_000_000.0,
                "partial_rotary_factor": 0.25,
            }
        )
        self.initializer_range = initializer_range
        self.indexer_n_heads = indexer_n_heads
        self.indexer_kv_heads = indexer_kv_heads
        self.indexer_head_dim = indexer_head_dim
        self.indexer_budget = indexer_budget
        self.indexer_compress_ratio = indexer_compress_ratio
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.num_aux_hidden_states = num_aux_hidden_states
        self.draft_vocab_size = (
            vocab_size if draft_vocab_size is None else draft_vocab_size
        )
        self.target_hidden_size = (
            hidden_size if target_hidden_size is None else target_hidden_size
        )
        self.fc_norm = fc_norm
        self.norm_output = norm_output
        self.parallel_drafting = parallel_drafting
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["QwenSparseEagle3DraftConfig"]
