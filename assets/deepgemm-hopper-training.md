# Experimental DeepGEMM Hopper training backend

The companion fork is https://github.com/Shomvel/DeepGEMM, branch
`sonic-hopper-training`. The scatter implementation requires companion commit
`deaf34ddb6436dd4ac344abf2431780cc3410f2d` (or a compatible later revision).
Install that checkout (including its submodules) before
using this backend. The implementation is `sonicmoe/deepgemm.py`; the existing
QuACK backend remains the default.

```python
from sonicmoe import deepgemm_swiglu
# x: BF16 [T,D]; w1: BF16 [E,2I,D], gate/up interleaved
# w2: BF16 [E,D,I]; scores: FP32/BF16 [T,K]; ids: int32/int64 [T,K]
y = deepgemm_swiglu(x, w1, w2, scores, ids, precision="bf16")
# scatter=False retains the materialized output path for comparisons.
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
hidden dimensions, and arbitrary custom CUDA epilogues are outside this implementation.

For each route, backward computes unweighted `U = dO @ W2`, then `ds = dot(U,A)`
and `dH = dSwiGLU(H, s*U)`. With down-projection bias, `dot(dO,b2)` is added to ds.
`dW2 = dO.T @ (s*A)`; saved preactivations allow recomputing A without saving Y.
The implementation saves X/H and parameters/route metadata. By default, the
forward down-projection and backward input-gradient GEMMs scatter their FP32
accumulators directly into token-sized buffers before a BF16 cast. They do not
materialize route-expanded Y or dX. `scatter=False` retains the previous path.
The atomic reduction is nondeterministic and rejects deterministic modes. It
materializes the other gather buffers and uses separate
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

## Historical materialized baseline, 2026-10-23

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

## Scatter implementation

Both SM90 BF16 and scaled FP8 WGMMA kernels have a shared CUDA scatter epilogue.
For each valid row the forward epilogue computes `(accumulator + bias[e]) * score`
and uses FP32 `atomicAdd` into its original token. Backward dX omits bias and
score because dH is already weighted. Padding never contributes. Backward keeps
`ds = dot(U,A) + dot(dO,b2)` and does not regenerate Y. The FP32 accumulator is
zeroed before launch and converted once to the public BF16 output.

For T=32768, D=4096 and top-k=8 without padding, the removed BF16 Y is 2 GiB;
the replacement FP32 accumulator is 512 MiB. Both paths additionally need the
256 MiB final BF16 output. Packing, preactivations, other gradients and optional
bias-gradient reductions can still dominate whole-layer peak memory. Atomic
contention can also cost more than the removed sequential output traffic.
`benchmarks/deepgemm_training.py` measures both paths from the same source, with
whole-layer synchronized forward+backward times and a separate pre-grouped
projection test at this shape. No long-run convergence or H800/H200 claim follows.

## Scatter H100 measurement, 2026-10-23

A clean source build passed raw scatter/ordinary GEMM checks, 10 whole-layer
numerical cases, both MoE wrappers and 20-step fixed-routing SGD in BF16/FP8.
The same-job comparison below includes the original Sonic backend and both
DeepGEMM paths. [Raw results](deepgemm-h100-scatter-results.json),
[adapter checks](deepgemm-h100-scatter-adapters.json) and
[source/build/job provenance](deepgemm-h100-scatter-provenance.json) are retained.
The worker completed with exit 0 and was deleted; all task jobs are absent.

Forward+backward milliseconds (synchronized wall time, 2 warmups, 3 samples of
5 iterations; includes routing/packing/quantization, excludes optimizer/JIT):

| T/D/I/E/K | Sonic BF16 | Materialize BF16 | Scatter BF16 | Materialize FP8 | Scatter FP8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/256/512/4/2 | 1.603 | 1.586 | 1.541 | 2.349 | 2.245 |
| 2048/1024/2048/8/2 | 1.596 | 2.037 | 2.267 | 2.929 | 2.402 |
| 4096/4096/2048/8/2 | 2.372 | 5.955 | 6.208 | 4.872 | 5.113 |
| 4096/4096/512/8/8 | 2.350 | 4.457 | 5.620 | 5.686 | 6.946 |

Incremental peak allocated memory in MiB:

| T/D/I/E/K | Sonic BF16 | Materialize BF16 | Scatter BF16 | Materialize FP8 | Scatter FP8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/256/512/4/2 | 6.29 | 17.40 | 16.78 | 17.40 | 16.78 |
| 2048/1024/2048/8/2 | 144.07 | 397.85 | 388.60 | 397.85 | 388.60 |
| 4096/4096/2048/8/2 | 576.14 | 1566.20 | 1496.20 | 1566.20 | 1496.20 |
| 4096/4096/512/8/8 | 480.52 | 1248.75 | 992.75 | 1290.82 | 1066.82 |

The top-k=8 whole-layer case saves 256 MiB in BF16 (20.5%) and 224 MiB in FP8
(17.4%), but takes 26.1% and 22.2% longer than the corresponding materialized
path. Both remain slower and use more memory than original Sonic. The smaller
wall-time differences vary by shape; these samples do not establish a general
speedup. No kernel-level profiler attribution was performed.

The separate down-projection uses T/D/I/E/K=32768/4096/256/8/8, with all 8 experts
selected in expert order for each token. Inputs and routing are already grouped;
quantization, accumulator zeroing, reduction and final BF16 cast are timed with
CUDA events. Inputs/weights are excluded from incremental peak, while new output
and temporary allocations are included. There is no bias in these timings.

| Precision | Path | ms | Incremental peak / MiB |
| --- | --- | ---: | ---: |
| BF16 | materialize | 2.029 | 2304.00 |
| BF16 | scatter | 7.375 | 768.00 |
| FP8 | materialize | 2.178 | 2304.00 |
| FP8 | scatter | 7.656 | 842.00 |

The BF16 Y buffer is 2048 MiB and is replaced by a 512 MiB FP32 accumulator;
both paths need a 256 MiB final BF16 output. Measured BF16 incremental peak drops
from 2304 to 768 MiB (66.7%); FP8 drops to 842 MiB including conversion buffers.
Projection time increases 3.64x in BF16 and 3.51x in FP8. Scatter versus
materialized output relative L2 error is 0.262% in both precisions, consistent
with removing intermediate BF16 rounding. The result establishes a memory/time
tradeoff, not a throughput improvement. Other I dimensions or route schedules
may change the tradeoff; this shape is not a full large-model training run.

Raw BF16 FP32 scatter relative L2 error is at most 1.29e-7; FP8 is 3.67% against
independent FP32 GEMM/reduction. Full-layer outputs and gradients reach 0.463%
(BF16) and 6.501% (FP8), including top-k=8 signed scores and duplicate routes.
Wrapper maxima are 0.622%/6.591%; auxiliary losses match. SGD losses decrease
for all modes; final outputs differ from FP32 by 0.380%/6.494%. Deterministic
mode rejection, invalid route dtype/length, nonzero padding, zero scores and
empty experts/batches are checked. This does not establish long-run convergence.

The environment matches the earlier baseline description (including the NVIDIA
PyTorch prerelease caveat). Only H100 was tested. A bounded-scratch producer/
reduction design is a possible next experiment; it is not implemented here.
