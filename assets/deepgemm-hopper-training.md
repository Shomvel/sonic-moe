# Experimental DeepGEMM Hopper training backend

The companion fork is https://github.com/Shomvel/DeepGEMM, branch
`sonic-hopper-training`. Install that checkout (including its submodules) before
using this backend. The implementation is `sonicmoe/deepgemm.py`; the existing
QuACK backend remains the default.

```python
from sonicmoe import deepgemm_swiglu
# x: BF16 [T,D]; w1: BF16 [E,2I,D], gate/up interleaved
# w2: BF16 [E,D,I]; scores: FP32/BF16 [T,K]; ids: int32/int64 [T,K]
y = deepgemm_swiglu(x, w1, w2, scores, ids, precision="bf16")
# precision="fp8" requantizes all six GEMMs, including weight gradients.
```

The `MoE` wrapper accepts `KernelBackendMoE.deepgemm_bf16` and
`KernelBackendMoE.deepgemm_fp8` and preserves its routing/auxiliary loss interface.
The wrapper test covers both modes, BF16 router scores, expert biases, router
parameter gradients and auxiliary loss. The Torch reference backend explicitly
casts routed expert outputs to FP32 before its FP32 `index_add` reduction.

Scope: single-GPU Hopper SM90, BF16 master tensors, interleaved SwiGLU, optional
expert biases, D/I divisible by 128, fixed top-k assignments, and first gradients.
No token rounding is applied. Empty experts and zero routing scores are supported.
The launch copies expert counts to the host, so CUDA graph capture is rejected.
Higher derivatives, expert parallel communication, other activations, arbitrary
hidden dimensions, and custom CUDA epilogues are outside this implementation.

For each route, backward computes unweighted `U = dO @ W2`, then `ds = dot(U,A)`
and `dH = dSwiGLU(H, s*U)`. With down-projection bias, `dot(dO,b2)` is added to ds.
`dW2 = dO.T @ (s*A)`; saved preactivations allow recomputing A without saving Y.
The implementation saves X/H and parameters/route metadata; padded X/A/Y/dX are
still temporary workspaces. It materializes gather buffers and uses separate
activation kernels. Those costs can outweigh DeepGEMM's GEMM throughput advantage.

FP8 mode uses E4M3, FP32 block scales and straight-through gradients. The training
objective and gradients approximate BF16; a local tolerance test is not evidence of
long-run model convergence. Bias/router reductions use FP32. BF16 gradients are
returned to BF16 parameters, as with the original interface.

GPU validation:

```sh
python -m pytest tests/test_deepgemm_training.py -q
python benchmarks/deepgemm_training.py --output results --iterations 5
```

The second command checks outputs and all gradients against an independent FP32
expression, empty/skewed experts, exact zero scores, biases, short SGD training and
original SonicMoE, then benchmarks forward+backward. Synchronized wall times include
routing, D2H count sync, padding, packing, quantization, all GEMMs and reductions.
Each step uses fresh conversions without cached/obsolete weights. It excludes the
optimizer update, router linear/top-k shared outside the fixed-route expert layer,
and JIT/autotuning warmup. Report GPU peak allocated memory alongside time.

## H100 measurement, 2026-10-23

One H100 80GB (SM90), Python 3.12.3, CUDA 13.0, NVIDIA PyTorch
`2.10.0a0+b558c986e8.nv25.11`, Triton `3.5.0+gitde3506d2`, QuACK 0.6.4 and
CUTLASS DSL 4.6.2 passed the adapter, numerical, wrapper and 20-step SGD checks.
This source-import experiment uses a PyTorch prerelease below the package's
declared `torch>=2.11.0` requirement; it does not establish general 2.10 support.
The base image has protobuf 6.33.5/upb; the isolated benchmark environment reports
6.33.6/upb. The full numerical and timing data are in
[deepgemm-h100-results.json](deepgemm-h100-results.json); source hashes, image digest,
job identity and terminal/cleanup evidence are in
[deepgemm-h100-provenance.json](deepgemm-h100-provenance.json).

Times are forward+backward milliseconds: median of 3 samples, 5 iterations per
sample, after 2 warmups. T/D/I/E/K are tokens, hidden width, intermediate width,
experts and top-k. Routing is random with fixed assignments during each benchmark.

| T/D/I/E/K | Sonic BF16 | DeepGEMM BF16 | DeepGEMM FP8 |
| --- | ---: | ---: | ---: |
| 512/256/512/4/2 | 1.815 | 1.796 | 2.453 |
| 2048/1024/2048/8/2 | 1.945 | 2.115 | 2.515 |
| 4096/4096/2048/8/2 | 2.386 | 5.855 | 4.818 |

Incremental peak allocated memory in MiB (peak minus live allocation before the
timed loop; includes transient gradients and conversion workspaces):

| T/D/I/E/K | Sonic BF16 | DeepGEMM BF16 | DeepGEMM FP8 |
| --- | ---: | ---: | ---: |
| 512/256/512/4/2 | 6.29 | 17.40 | 17.40 |
| 2048/1024/2048/8/2 | 144.07 | 397.85 | 397.85 |
| 4096/4096/2048/8/2 | 576.14 | 1566.20 | 1566.20 |

The largest shape is 2.45x slower in BF16 and 2.02x slower in FP8 than Sonic.
The smallest BF16 difference is within the sample variation. These measurements
do not show a throughput or peak-memory advantage for this adapter. Gather,
padding, transposes, quantization and launch overhead remain unfused; no profiler
breakdown has yet attributed the measured runtime to individual operations.

Against the independent FP32 expert expression, the largest relative L2 error
across the tested nonempty outputs and gradients was 0.46% in BF16 and 6.51% in
FP8. The MoE wrapper comparison against the Torch backend reached 0.63% and 6.60%
respectively, including router parameter gradients. Auxiliary losses matched.
All three fixed-routing SGD runs reduced the loss; 20 steps do not establish
long-run convergence. Only H100 was executed; H800/H200 remain untested.
