# Experimental DeepGEMM Hopper training backend

Install the companion [DeepGEMM fork](https://github.com/Shomvel/DeepGEMM/tree/sonic-hopper-training), at commit `fde59707727ba614085849f01715c845d8896501`, including submodules, before importing this backend. Its current adapter/native changes are required: direct BF16 weight gradients, no-C overwrite, strided FP8 column output and bounded gathered-RHS weight gradients. QuACK remains the package default.

```python
from sonicmoe import deepgemm_swiglu
# BF16: x[T,D], w1[E,2I,D] (interleaved gate/up), w2[E,D,I]
# scores[T,K]: FP32/BF16; ids[T,K]: int32/int64
out = deepgemm_swiglu(x, w1, w2, scores, ids, precision="fp8")
# Opt in to lower backward memory; performance is shape dependent.
out = deepgemm_swiglu(x, w1, w2, scores, ids,
                     precision="fp8", recompute_dh=True)
```

`MoE.forward` accepts `KernelBackendMoE.deepgemm_bf16` and `deepgemm_fp8`, preserving router gradients and auxiliary loss. Pass `recompute_dh=True` as a keyword with `deepgemm_fp8`. Recompute requires chunked reduction and no up-projection bias (`bias1=None`); down-projection bias is supported. Unsupported combinations raise an error instead of silently retaining the large buffers.

## Training and memory contracts

Single-GPU Hopper SM90, BF16 inputs/parameter tensors, D/I divisible by 128, fixed top-k assignments, interleaved SwiGLU and first derivatives. Empty experts, duplicate routes, signed and zero scores are tested. Counts copy to the host once; CUDA graph capture is rejected. Expert parallel communication, higher derivatives and other activations are outside this implementation.

Backward forms unweighted `U=dO@W2`, then `ds=dot(U,A)` (plus `dot(dO,b2)` when present). This preserves router gradients at zero scores. dH already includes scores, so dX does not multiply them again. Saved tensors contain X/H, parameters and routing metadata, not expert output Y. There is no cross-step weight cache.

Default `reduction="chunked"` reuses a BF16 route/column projection workspace. Automatic sizing targets 512 MiB: at most 4096 columns in BF16 or 2048 in FP8, with minimum 128; `chunk_size` overrides the column count. When D fits, a full-width intermediate still exists. Each tile is immediately combined into the token output.

`reduction="atomic"` directly scatters FP32 GEMM accumulators to a zeroed token-sized buffer and casts to BF16, completely eliminating expanded projection outputs. Atomics are nondeterministic and reject deterministic modes. They are slower in these measurements. `reduction="materialize"` retains full expanded outputs for comparison; legacy `scatter=True/False` selects atomic/materialize.

FP8 gathers tokens directly into quantizers, emits native scale layouts and computes all six GEMMs using E4M3 operands. Native weight gradients keep FP32 register accumulation and write BF16. Fresh wgrad output uses no-C overwrite; dW1 uses 2048-column gathered-RHS chunks, writing directly to final output slices. BF16 preserves supported transposed weight views. Bias gradients use bounded partial FP32 sums.

With `recompute_dh=True`, backward generates K-group FP8 dH, computes dW1, releases it, recomputes row-group FP8 dH and ds, computes dX, releases that encoding, then generates quantized score*A for dW2. Neither full BF16 dH nor AP is allocated. Both BF16 rounding boundaries and the original unweighted ds reduction are preserved. The fixed late-dW2 schedule trades extra activation work for memory; it is not selected automatically by benchmark shape.

FP8 uses straight-through approximate gradients, not derivatives through the quantizer. It is not numerically equivalent to BF16. BF16 parameter tensors receive BF16 gradients; an external FP32 master optimizer can cast weights for each forward.

## H100 production acceptance: opt-r15

A clean source build passed 42 strict-FP32 DeepGEMM cases, 3 original Sonic cases, 16 recompute quantizer checks, 9 recompute full-gradient/retained-graph/accumulation cases, wrappers and 20-step FP32-master SGD. TF32 is explicitly disabled. Native regressions include 40 multicast boundaries, 120 M-tail combinations, overwrite/strided output contracts and real addresses exceeding 2³¹ elements.

Maximum whole-tensor relative L2 error across the 42 cases: BF16 **0.4630%**, FP8 **6.6957%**. The 9 recompute cases reached 6.5640% versus FP32 and were **bitwise identical to default FP8 for every output and gradient**. Recompute and default FP8 also produced identical final outputs/parameters after 20 SGD updates. Separate earlier candidate inputs reached 7.4652%; these observed maxima are not universal error bounds. The final loss after 20 updates was 0.0111980308 (FP32), 0.0112803960 (BF16), 0.0112843160 (both FP8 modes). This small fixed task does not establish long-run convergence or real-model quality.

The table reports synchronized whole-layer forward+backward **ms / incremental peak MiB**, with distinct random expert selections per token. Default FP8, recompute and Sonic were each measured 25 times before and 25 times after the other comparisons; table latency is the pooled median. DG BF16 has 25 samples. Memory is the maximum of paired records. Routing, D2H counts, padding, quantization, all GEMMs and reductions are included; optimizer, router linear/top-k and JIT warmup are excluded.

| T / I / E / K (D=4096) | Sonic BF16 | DG BF16 | DG FP8 default | DG FP8 recompute |
|---|---:|---:|---:|---:|
| 4096 / 512 / 64 / 8 | 3.074 / 1152.52 | 3.532 / 1261.29 | 4.037 / 1085.57 | 4.403 / 1069.43 |
| 32768 / 512 / 64 / 8 | 18.288 / 3844.13 | 23.279 / 4405.80 | 18.737 / 3129.85 | 20.251 / 3012.59 |
| 32768 / 512 / 256 / 8 | 20.138 / 6148.13 | 31.831 / 6851.19 | 22.927 / 5517.98 | 24.734 / 5395.84 |
| 32768 / 512 / 64 / 16 | 36.293 / 6408.13 | 44.507 / 7492.05 | 35.982 / 5498.67 | 39.592 / 4718.67 |
| 32768 / 1536 / 64 / 8 | 51.531 / 6404.13 | 59.728 / 8025.55 | 42.256 / 7284.88 | 47.414 / 5858.26 |

At I1536, recompute reduces time by about 8.0% and incremental peak by 8.5% versus Sonic, at FP8 numerical precision. Default FP8 is faster there but uses more memory. K16 default saves about 14.2% memory; its sub-1% timing difference does not establish a reliable speedup. Short-T and E256 remain slower than Sonic. Pure BF16 DeepGEMM has no overall advantage here. Total allocation and retained-forward memory are also in the raw report; incremental layer peak is not full-model training memory.

[Raw production results](deepgemm-h100-optimized-results.json) contain all 55 measurements, numerical errors and training trajectories. [Provenance](deepgemm-h100-optimized-provenance.json) contains source/install/native hashes, pinned image, environment, independent static review, completed job and cleanup receipts. Earlier materialized/scatter JSON assets remain historical evidence, not the current acceptance result.

```sh
NVIDIA_TF32_OVERRIDE=0 python -m pytest tests/test_deepgemm_training.py tests/test_deepgemm_recompute.py -q
NVIDIA_TF32_OVERRIDE=0 python benchmarks/deepgemm_training.py --output results --iterations 10 --extended-shapes
```

Executed on H100 80GB, CUDA 13.0, NVIDIA Torch 2.10.0a0+b558c986e8.nv25.11, Triton 3.5.0+gitde3506d2, QuACK 0.6.4 and CUTLASS DSL 4.6.2. Base protobuf 6.33.5/upb; isolated environment 6.33.6/upb. The source-import experiment uses a prerelease below the declared torch>=2.11 requirement; it does not establish general 2.10 support. H800/H200, distributed training, dynamic-routing training and long-run model quality are untested.
