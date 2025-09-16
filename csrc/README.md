# Fusion Operators (CSRC)

High-performance CUDA kernels for accelerating model training, with specialized support for multimodal and MoE architectures.

## Operators

### Asymmetric Dual Expert GEMM
- `asym_dual_gmm`: Simultaneous matrix multiplication for two experts
- Supports all transpose combinations (NN, TN, NT, TT)

### Token Permutation
- `permute`: Token permutation for MoE routing
- `unpermute`: Token recovery after expert computation
- `unpermute_bwd`: Backward pass for token recovery

### Multimodal RoPE
- `rope`: Rotary Position Embedding forward pass
- `rope_bwd`: RoPE backward pass
- `rope_index`: Generates position indices for multimodal RoPE
- `rot_pos_emb`: Fused rotary position embedding computation

### Vision Transformer Optimization
- `get_window_index`: Window attention index generation


## Acknowledgments

The `permute` and `unpermute` operators are adapted from [fanshiqing/grouped_gemm](https://github.com/fanshiqing/grouped_gemm). Thanks for their open-source contributions.
