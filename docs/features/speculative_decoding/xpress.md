# DFlash2 + xPress

`XPressDraftModel` retains DFlash2 grouped convolutions and adds the
[released xPress head](https://github.com/Supercomputing-System-AI-Lab/xPress/blob/main/refiners/xpress_head.py).
It uses Model Runner V2 and the existing DFlash context-cache and verification path.

```bash
vllm serve /path/to/target --dtype bfloat16 \
    --speculative-config '{"method":"dflash","model":"/path/to/xpress","num_speculative_tokens":15,"draft_sample_method":"greedy"}'
```

Use the trained `block_size - 1` prediction length first. The block includes the
known anchor. The refiner receives the entire block's hidden states and the true
token before that anchor, taken from accepted target context after rejection.
It never substitutes an anchor or a neighbouring request's token for that input.

The head is replicated across TP ranks. Base logits are gathered once; each
fixed-count Jacobi pass refines full-vocabulary logits without repeating the
backbone or introducing a TP collective. Head weights stay in floating-point
model dtype. `dflash_config.xpress_refinement_steps` controls passes; fewer than
the prediction depth gives an approximate parallel proposal, still verified by
the normal target verifier.

The export must retain `XPressDraftModel`, `dflash_config.xpress_*`, convolution
tensors and all nine `xpress_head.*` parameters. `mix.L` is the raw trainable
matrix. Loading folds its causal mask and identity into a nonpersistent buffer.
Checkpoint and config names follow the AutoModel export, not the linked
`vllm_xpress` package's plain-DFlash configuration.

Probabilistic proposal sampling and adaptive verification are not implemented.
Training-time VAT and D-PACE remain training features, not serving guarantees.
No compatibility is claimed with the linked xPress runtime unchanged.

CPU head, token-coordinate and padding tests are provided. GPU model loading,
TP, CUDA graphs, end-to-end acceptance and throughput still need qualification.
