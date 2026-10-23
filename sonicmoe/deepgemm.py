"""Experimental Hopper SwiGLU training backend using all six DeepGEMM GEMMs.

Public expert weights use [E,output,input]. Routing is fixed top-k: padding never
adds or drops a route. The only host synchronization copies E route counts, which
SM90 DeepGEMM's K-grouped launch currently requires. Saved activations are X and
BF16 preactivations H; A is recomputed. Output reductions use bounded columns
by default, with an optional direct atomic scatter epilogue.
"""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable


@tr.jit
def _gather(X, MAP, Y, M: tl.constexpr, N: tl.constexpr, TOPK: tl.constexpr,
            B: tl.constexpr, R: tl.constexpr):
    rows = (tl.program_id(0) * R + tl.arange(0, R)).to(tl.int64)
    c = tl.program_id(1) * B + tl.arange(0, B)
    route = tl.load(MAP + rows, rows < M, -1)
    v = tl.load(X + (route[:, None] // TOPK) * N + c[None, :],
                (route[:, None] >= 0) & (c[None, :] < N), 0)
    tl.store(Y + rows[:, None] * N + c[None, :], v,
             (rows[:, None] < M) & (c[None, :] < N))


@tr.jit
def _act(H, A, BIAS, EXPERT, MAP, I: tl.constexpr, B: tl.constexpr, HAS_BIAS: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, B)
    valid = tl.load(MAP + r) >= 0
    g = tl.load(H + r * 2 * I + 2 * c, c < I, 0).to(tl.float32)
    u = tl.load(H + r * 2 * I + 2 * c + 1, c < I, 0).to(tl.float32)
    if HAS_BIAS:
        e = tl.load(EXPERT + r).to(tl.int64)
        g += tl.load(BIAS + e * 2 * I + 2 * c, c < I, 0).to(tl.float32)
        u += tl.load(BIAS + e * 2 * I + 2 * c + 1, c < I, 0).to(tl.float32)
    g = tl.where(valid, g, 0).to(H.dtype.element_ty).to(tl.float32)
    u = tl.where(valid, u, 0).to(H.dtype.element_ty).to(tl.float32)
    tl.store(H + r * 2 * I + 2 * c, g, c < I)
    tl.store(H + r * 2 * I + 2 * c + 1, u, c < I)
    a = g * tl.sigmoid(g) * u
    tl.store(A + r * I + c, a, c < I)


@tr.jit
def _combine_columns(Y, INVERSE, SCORES, IDS, BIAS, O,
                     T: tl.constexpr, D: tl.constexpr, C: tl.constexpr, STRIDE: tl.constexpr,
                     START: tl.constexpr, TOPK: tl.constexpr, B: tl.constexpr, R: tl.constexpr,
                     WEIGHTED: tl.constexpr, HAS_BIAS: tl.constexpr):
    t = (tl.program_id(0) * R + tl.arange(0, R)).to(tl.int64)
    c = tl.program_id(1) * B + tl.arange(0, B)
    mask = (t[:, None] < T) & (c[None, :] < C)
    acc = tl.full((R, B), 0, tl.float32)
    for k in range(TOPK):
        route = t * TOPK + k
        row = tl.load(INVERSE + route, t < T, 0)
        v = tl.load(Y + row[:, None] * STRIDE + c[None, :], mask, 0).to(tl.float32)
        if HAS_BIAS:
            e = tl.load(IDS + route, t < T, 0).to(tl.int64)
            v += tl.load(BIAS + e[:, None] * D + START + c[None, :], mask, 0).to(tl.float32)
        if WEIGHTED:
            v *= tl.load(SCORES + route, t < T, 0).to(tl.float32)[:, None]
        acc += v
    tl.store(O + t[:, None] * D + START + c[None, :], acc, mask)


def _project(gemms, a, w, rowmap, inverse, ids, scores, bias, reduction, chunk_size):
    t, topk = ids.shape
    if reduction == "atomic":
        return gemms.m_gemm_scatter(a, w, rowmap, t, topk, scores, bias)
    out = torch.empty((t, w.size(1)), device=a.device, dtype=torch.bfloat16)
    if not t:
        return out
    if reduction == "materialize":
        y = gemms.m_gemm(a, w)
        _combine_columns[(tr.cdiv(t, 4), tr.cdiv(w.size(1), 256))](
            y, inverse, scores, ids, bias, out, t, w.size(1), w.size(1), y.stride(0),
            0, topk, 256, 4, scores is not None, bias is not None)
    else:
        for start, y in gemms.m_gemm_columns(a, w, chunk_size):
            _combine_columns[(tr.cdiv(t, 4), tr.cdiv(y.size(1), 256))](
                y, inverse, scores, ids, bias, out, t, w.size(1), y.size(1), y.stride(0),
                start, topk, 256, 4, scores is not None, bias is not None)
    return out


@tr.jit
def _act_backward(H, U, MAP, SCORES, DH, AP, DS, I: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    route = tl.load(MAP + r)
    c = tl.arange(0, B)
    g = tl.load(H + r * 2 * I + 2 * c, c < I, 0).to(tl.float32)
    up = tl.load(H + r * 2 * I + 2 * c + 1, c < I, 0).to(tl.float32)
    v = tl.load(U + r * I + c, c < I, 0).to(tl.float32)
    s = tl.load(SCORES + route, route >= 0, 0).to(tl.float32)
    sig = tl.sigmoid(g)
    a = (g * sig * up).to(AP.dtype.element_ty).to(tl.float32)
    # Router derivative uses the UNWEIGHTED U, including for zero routing scores.
    ds = tl.sum(tl.where(c < I, v * a, 0), 0)
    tl.store(DS + route, ds, route >= 0)
    grad = v * s
    dg = grad * up * sig * (1 + g * (1 - sig))
    du = grad * g * sig
    tl.store(DH + r * 2 * I + 2 * c, dg, c < I)
    tl.store(DH + r * 2 * I + 2 * c + 1, du, c < I)
    tl.store(AP + r * I + c, a * s, c < I)


@tr.jit
def _bias_router_grad(DO, BIAS, IDS, DS, D: tl.constexpr, TOPK: tl.constexpr, B: tl.constexpr):
    route = tl.program_id(0).to(tl.int64)
    e = tl.load(IDS + route).to(tl.int64)
    c = tl.arange(0, B)
    v = tl.load(DO + route // TOPK * D + c, c < D, 0).to(tl.float32)
    bias = tl.load(BIAS + e * D + c, c < D, 0).to(tl.float32)
    tl.store(DS + route, tl.load(DS + route) + tl.sum(v * bias, 0))



@tr.jit
def _bias_partials(X, MAP, SCORES, PARTIAL, N: tl.constexpr, TOPK: tl.constexpr,
                   GATHER: tl.constexpr, B: tl.constexpr):
    row = (tl.program_id(0) * 128 + tl.arange(0, 128)).to(tl.int64)
    col = tl.program_id(1) * B + tl.arange(0, B)
    route = tl.load(MAP + row)
    source = route // TOPK if GATHER else row
    value = tl.load(X + source[:, None] * N + col[None, :],
                    (route[:, None] >= 0) & (col[None, :] < N), 0).to(tl.float32)
    if GATHER:
        value *= tl.load(SCORES + route, route >= 0, 0).to(tl.float32)[:, None]
    tl.store(PARTIAL + tl.program_id(0).to(tl.int64) * N + col, tl.sum(value, 0), col < N)


@tr.jit
def _bias_finish(PARTIAL, OFFSETS, OUT, N: tl.constexpr, B: tl.constexpr):
    e = tl.program_id(0).to(tl.int64)
    start = tl.load(OFFSETS + e).to(tl.int64) // 128
    end = tl.load(OFFSETS + e + 1).to(tl.int64) // 128
    col = tl.program_id(1) * B + tl.arange(0, B)
    acc = tl.full((B,), 0, tl.float32)
    for base in range(start, end, 128):
        row = base + tl.arange(0, 128)
        value = tl.load(PARTIAL + row[:, None] * N + col[None, :],
                        (row[:, None] < end) & (col[None, :] < N), 0)
        acc += tl.sum(value, 0)
    tl.store(OUT + e * N + col, acc, col < N)


def _bias_grad(x, layout, rowmap, topk, scores=None):
    # One partial row per 128 routes, never a full FP32 route-expanded input.
    out = torch.zeros((len(layout.ks), x.size(1)), device=x.device, dtype=x.dtype)
    if layout.rows:
        partial = torch.empty((layout.rows // 128, x.size(1)), device=x.device, dtype=torch.float32)
        grid = (layout.rows // 128, tr.cdiv(x.size(1), 64))
        _bias_partials[grid](x, rowmap, scores, partial, x.size(1), topk, scores is not None, 64)
        _bias_finish[(len(layout.ks), tr.cdiv(x.size(1), 64))](partial, layout.offsets, out, x.size(1), 64)
    return out


def _gather_rows(x, row_to_route, topk):
    out = torch.empty((row_to_route.numel(), x.size(1)), device=x.device, dtype=x.dtype)
    if out.numel():
        _gather[(tr.cdiv(out.size(0), 16), tr.cdiv(out.size(1), 256))](
            x, row_to_route, out, out.size(0), x.size(1), topk, 256, 16, num_warps=4)
    return out


@tr.jit
def _route_histogram(IDS, HIST, N: tl.constexpr, E: tl.constexpr, H: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(IDS + row, row < N, 0)
    # E is a dedicated invalid-ID bin; padded lanes are excluded entirely.
    value = tl.where((value >= 0) & (value < E), value, E).to(tl.int32)
    counts = tl.histogram(value, H, mask=row < N)
    tl.store(HIST + tl.program_id(0) * H + tl.arange(0, H), counts)


@tr.jit
def _route_prefix(HIST, PREFIX, BLOCKS: tl.constexpr, H: tl.constexpr, B: tl.constexpr):
    block = tl.arange(0, B)
    e = tl.program_id(0)
    counts = tl.load(HIST + block * H + e, block < BLOCKS, 0)
    prefix = tl.cumsum(counts, 0) - counts
    tl.store(PREFIX + block * H + e, prefix, block < BLOCKS)


@tr.jit
def _route_scatter(IDS, PREFIX, OFFSETS, MAP, INV, N: tl.constexpr,
                   H: tl.constexpr, E: tl.constexpr, B: tl.constexpr):
    lane = tl.arange(0, B)
    row = tl.program_id(0) * B + lane
    expert = tl.load(IDS + row, row < N, 0).to(tl.int64)
    # Unique integer keys preserve original route order within each expert.
    key = tl.where(row < N, expert * (N + 1) + row, 0x7fffffffffffffff)
    key = tl.sort(key, descending=False)
    e, route = key // (N + 1), key % (N + 1)
    previous = tl.gather(e, tl.maximum(lane - 1, 0), axis=0)
    first = (lane == 0) | (e != previous)
    start = tl.associative_scan(tl.where(first, lane, 0), axis=0, combine_fn=_route_max)
    valid = e < E
    base = tl.load(OFFSETS + e, valid, 0)
    prefix = tl.load(PREFIX + tl.program_id(0) * H + e, valid, 0)
    dest = base + prefix + lane - start
    tl.store(MAP + dest, route, valid)
    tl.store(INV + route, dest, valid)


@tr.jit
def _route_max(a, b):
    return tl.maximum(a, b)


def _routing(ids, experts):
    from deep_gemm.sonic_training import GroupedTrainingLayout
    flat = ids.flatten()
    if not flat.numel():
        layout = GroupedTrainingLayout.from_counts([0] * experts, ids.device)
        empty = torch.empty((0,), device=ids.device, dtype=torch.int64)
        return layout, empty, empty
    blocks, bins = tr.cdiv(flat.numel(), 1024), tr.next_power_of_2(experts + 1)
    hist = torch.empty((blocks, bins), device=ids.device, dtype=torch.int32)
    _route_histogram[(blocks,)](flat, hist, flat.numel(), experts, bins, 1024)
    host = hist.sum(0).cpu().tolist()
    if host[experts]:
        raise ValueError("expert IDs out of range")
    layout = GroupedTrainingLayout.from_counts(host[:experts], ids.device)
    prefix = torch.empty_like(hist)
    _route_prefix[(experts,)](hist, prefix, blocks, bins, tr.next_power_of_2(blocks))
    rowmap = torch.full((layout.rows,), -1, device=ids.device, dtype=torch.int64)
    inverse = torch.empty((flat.numel(),), device=ids.device, dtype=torch.int64)
    _route_scatter[(blocks,)](flat, prefix, layout.offsets, rowmap, inverse, flat.numel(), bins, experts, 1024)
    return layout, rowmap, inverse


class _DeepGemmSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, scores, ids, b1, b2, precision, reduction, chunk_size, recompute_dh):
        from deep_gemm.sonic_training import GroupedTrainingGemm
        layout, rowmap, inverse = _routing(ids, w1.size(0))
        gemms = GroupedTrainingGemm(layout, precision)
        if chunk_size is None:
            # Bound the BF16 route workspace to 512 MiB, subject to the native
            # minimum of 128 columns. Wider BF16 tiles avoid costly weight packs.
            budget_columns = (512 * 1024 * 1024 // (2 * max(layout.rows, 1))) // 128 * 128
            column_limit = 4096 if precision == "bf16" else 2048
            chunk_size = max(128, min(column_limit, budget_columns))
        if precision == "fp8":
            h = gemms.m_gemm_gather(x, w1, rowmap, ids.size(1))
        else:
            xp = _gather_rows(x, rowmap, ids.size(1))
            h = gemms.m_gemm(xp, w1)
            del xp
        a = torch.empty((layout.rows, w2.size(2)), device=x.device, dtype=x.dtype)
        if layout.rows:
            _act[(layout.rows,)](h, a, b1, layout.m_indices, rowmap, a.size(1), tr.next_power_of_2(a.size(1)), b1 is not None)
        out = _project(gemms, a, w2, rowmap, inverse, ids, scores, b2, reduction, chunk_size)
        ctx.reduction, ctx.chunk_size = reduction, chunk_size
        ctx.recompute_dh = recompute_dh
        ctx.save_for_backward(x, w1, w2, scores, ids, h, rowmap, inverse, b1, b2)
        ctx.gemms = gemms
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        if ctx.recompute_dh:
            from ._deepgemm_recompute import recompute_backward
            dx, dw1, dw2, ds, db2 = recompute_backward(ctx, dout)
            return dx, dw1, dw2, ds, None, None, db2, None, None, None, None
        x, w1, w2, scores, ids, h, rowmap, inverse, b1, b2 = ctx.saved_tensors
        gemms, layout = ctx.gemms, ctx.gemms.layout
        topk, d, i = ids.size(1), x.size(1), w2.size(2)
        dout = dout.contiguous()
        if gemms.precision == "fp8":
            u = gemms.m_gemm_gather(dout, w2.transpose(1, 2), rowmap, topk)
        else:
            dop = _gather_rows(dout, rowmap, topk)
            u = gemms.m_gemm(dop, w2.transpose(1, 2))
        dh, ap = torch.empty_like(h), torch.empty((layout.rows, i), device=x.device, dtype=x.dtype)
        ds = torch.empty_like(scores, dtype=torch.float32)
        if layout.rows:
            _act_backward[(layout.rows,)](h, u, rowmap, scores, dh, ap, ds, i, tr.next_power_of_2(i))
        del u
        if b2 is not None and ids.numel():
            _bias_router_grad[(ids.numel(),)](dout, b2, ids, ds, d, topk, tr.next_power_of_2(d))
        if gemms.precision == "fp8":
            dw2 = gemms.wgrad(dout, ap, row_to_route=rowmap, topk=topk, gather_lhs=True, output_dtype=w2.dtype)
        else:
            dw2 = gemms.wgrad(dop, ap, output_dtype=w2.dtype)
            del dop
        del ap
        dx = _project(gemms, dh, w1.transpose(1, 2), rowmap, inverse, ids,
                      None, None, ctx.reduction, ctx.chunk_size)
        if gemms.precision == "fp8":
            dw1 = gemms.wgrad_gather_rhs_columns(dh, x, rowmap, topk, output_dtype=w1.dtype)
        else:
            xp = _gather_rows(x, rowmap, topk)
            dw1 = gemms.wgrad(dh, xp, output_dtype=w1.dtype)
            del xp
        db1 = db2 = None
        if b1 is not None:
            db1 = _bias_grad(dh, layout, rowmap, topk)
        if b2 is not None:
            db2 = _bias_grad(dout, layout, rowmap, topk, scores)
        return dx, dw1, dw2, ds.to(scores.dtype), None, db1, db2, None, None, None, None


def deepgemm_swiglu(x, w1, w2, scores, expert_ids, *, bias1=None, bias2=None, precision="bf16",
                     reduction="chunked", chunk_size=None, scatter=None, recompute_dh=False):
    """Fixed top-k SwiGLU MoE, including gradients of X/W1/W2/scores/biases.

    X [T,D], W1 [E,2I,D] (gate/up interleaved), W2 [E,D,I], scores/IDs [T,K].
    BF16 tensors, FP32/BF16 scores, D/I divisible by 128; SM90 only. FP8 mode
    requantizes all GEMM operands (also wgrad) and uses straight-through gradients.
    reduction="chunked" reuses a column workspace (default: 512 MiB budget,
    up to 4096 BF16 / 2048 FP8 columns, minimum 128); "atomic" scatters
    FP32 accumulators and "materialize" retains full route-expanded outputs.
    Legacy scatter=True/False selects atomic/materialize. Only atomics are nondeterministic.
    recompute_dh=True avoids full BF16 dH/AP by recomputing directly into FP8;
    it requires FP8, chunked reduction and no bias1, and may cost extra runtime.
    Routing counts require a host sync; CUDA graph capture and higher gradients
    are intentionally unsupported in this first implementation.
    """
    if scatter is not None:
        reduction = "atomic" if scatter else "materialize"
    if reduction not in ("chunked", "atomic", "materialize"):
        raise ValueError("reduction must be chunked, atomic or materialize")
    if not isinstance(recompute_dh, bool):
        raise ValueError("recompute_dh must be a boolean")
    if recompute_dh and (precision != "fp8" or reduction != "chunked" or bias1 is not None):
        raise ValueError("recompute_dh requires fp8, chunked reduction and bias1=None")
    if chunk_size is not None and (not isinstance(chunk_size, int) or chunk_size <= 0 or chunk_size % 128):
        raise ValueError("chunk_size must be None or a positive multiple of 128")
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)[0] != 9:
        raise ValueError("DeepGEMM training backend currently requires Hopper SM90")
    if x.ndim != 2 or w1.ndim != 3 or w2.ndim != 3 or scores.ndim != 2 or expert_ids.shape != scores.shape:
        raise ValueError("expected X[ T,D ], expert weights[ E,N,K ], scores/IDs[ T,topk ]")
    e, d, i = w2.shape
    if w1.shape != (e, 2 * i, d) or x.size(1) != d or scores.size(0) != x.size(0):
        raise ValueError("inconsistent expert, token or hidden dimensions")
    if e < 1 or scores.size(1) < 1 or scores.size(1) > e or d % 128 or i % 128:
        raise ValueError("requires 1 <= topk <= E and D/I divisible by 128")
    tensors = [x, w1, w2] + [b for b in (bias1, bias2) if b is not None]
    if any(v.dtype != torch.bfloat16 or v.device != x.device for v in tensors):
        raise ValueError("activations, weights and biases must be BF16 on the same GPU")
    if scores.dtype not in (torch.bfloat16, torch.float32) or scores.device != x.device or expert_ids.device != x.device:
        raise ValueError("scores must be BF16/FP32 on the input GPU")
    if expert_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("expert IDs must be int32/int64")
    if (bias1 is not None and bias1.shape != (e, 2 * i)) or (bias2 is not None and bias2.shape != (e, d)):
        raise ValueError("invalid expert bias shape")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("SM90 K-grouped host counts prevent CUDA graph capture")
    with torch.cuda.device(x.device):
        return _DeepGemmSwiGLU.apply(x.contiguous(), w1, w2, scores.contiguous(), expert_ids.contiguous(),
                                    None if bias1 is None else bias1.contiguous(),
                                    None if bias2 is None else bias2.contiguous(), precision, reduction, chunk_size,
                                    recompute_dh)
