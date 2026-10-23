"""Experimental Hopper SwiGLU training backend using all six DeepGEMM GEMMs.

Public expert weights use [E,output,input]. Routing is fixed top-k: padding never
adds or drops a route. The only host synchronization copies E route counts, which
SM90 DeepGEMM's K-grouped launch currently requires. Saved activations are X and
BF16 preactivations H; A is recomputed and output GEMMs scatter directly to tokens.
"""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable


@tr.jit
def _gather(X, MAP, Y, N: tl.constexpr, TOPK: tl.constexpr, B: tl.constexpr):
    r, b = tl.program_id(0), tl.program_id(1)
    route = tl.load(MAP + r)
    c = b * B + tl.arange(0, B)
    v = tl.load(X + (route // TOPK) * N + c, (route >= 0) & (c < N), 0)
    tl.store(Y + r * N + c, v, c < N)


@tr.jit
def _act(H, A, BIAS, EXPERT, MAP, I: tl.constexpr, B: tl.constexpr, HAS_BIAS: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, B)
    valid = tl.load(MAP + r) >= 0
    g = tl.load(H + r * 2 * I + 2 * c, c < I, 0).to(tl.float32)
    u = tl.load(H + r * 2 * I + 2 * c + 1, c < I, 0).to(tl.float32)
    if HAS_BIAS:
        e = tl.load(EXPERT + r)
        g += tl.load(BIAS + e * 2 * I + 2 * c, c < I, 0).to(tl.float32)
        u += tl.load(BIAS + e * 2 * I + 2 * c + 1, c < I, 0).to(tl.float32)
    g = tl.where(valid, g, 0).to(H.dtype.element_ty).to(tl.float32)
    u = tl.where(valid, u, 0).to(H.dtype.element_ty).to(tl.float32)
    tl.store(H + r * 2 * I + 2 * c, g, c < I)
    tl.store(H + r * 2 * I + 2 * c + 1, u, c < I)
    a = g * tl.sigmoid(g) * u
    tl.store(A + r * I + c, a, c < I)


@tr.jit
def _combine(Y, INVERSE, SCORES, IDS, BIAS, O, D: tl.constexpr,
             TOPK: tl.constexpr, B: tl.constexpr, WEIGHTED: tl.constexpr, HAS_BIAS: tl.constexpr):
    t, block = tl.program_id(0), tl.program_id(1)
    c = block * B + tl.arange(0, B)
    acc = tl.full((B,), 0, tl.float32)
    for k in range(TOPK):
        route = t * TOPK + k
        row = tl.load(INVERSE + route)
        v = tl.load(Y + row * D + c, c < D, 0).to(tl.float32)
        if HAS_BIAS:
            e = tl.load(IDS + route)
            v += tl.load(BIAS + e * D + c, c < D, 0).to(tl.float32)
        if WEIGHTED:
            v *= tl.load(SCORES + route).to(tl.float32)
        acc += v
    tl.store(O + t * D + c, acc, c < D)


@tr.jit
def _act_backward(H, U, MAP, SCORES, DH, AP, DS, I: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0)
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
    route = tl.program_id(0)
    e = tl.load(IDS + route)
    c = tl.arange(0, B)
    v = tl.load(DO + route // TOPK * D + c, c < D, 0).to(tl.float32)
    bias = tl.load(BIAS + e * D + c, c < D, 0).to(tl.float32)
    tl.store(DS + route, tl.load(DS + route) + tl.sum(v * bias, 0))


def _gather_rows(x, row_to_route, topk):
    out = torch.empty((row_to_route.numel(), x.size(1)), device=x.device, dtype=x.dtype)
    if out.numel():
        _gather[(out.size(0), tr.cdiv(out.size(1), 256))](x, row_to_route, out, x.size(1), topk, 256)
    return out


def _routing(ids, experts):
    from deep_gemm.sonic_training import GroupedTrainingLayout
    flat = ids.flatten().long()
    # A bounded histogram and min/max check share one D2H synchronization. Never
    # silently discard invalid IDs; a padded expert row must still represent one route.
    valid = ((flat >= 0) & (flat < experts)).all()
    counts = torch.bincount(flat.clamp(0, experts - 1), minlength=experts).to(torch.int32)
    host = torch.cat((counts, valid.view(1).to(torch.int32))).cpu().tolist()
    if not host[-1]:
        raise ValueError("expert IDs out of range")
    layout = GroupedTrainingLayout.from_counts(host[:-1], ids.device)
    order = flat.argsort(stable=True)
    logical = torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1]))
    gaps = torch.repeat_interleave(layout.offsets[:-1] - logical, counts, output_size=flat.numel())
    rows = torch.arange(flat.numel(), device=ids.device) + gaps
    row_to_route = torch.full((layout.rows,), -1, device=ids.device, dtype=torch.int64)
    row_to_route[rows] = order
    inverse = torch.empty_like(flat)
    inverse[order] = rows
    return layout, row_to_route, inverse


class _DeepGemmSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, scores, ids, b1, b2, precision, scatter):
        from deep_gemm.sonic_training import GroupedTrainingGemm
        layout, rowmap, inverse = _routing(ids, w1.size(0))
        gemms = GroupedTrainingGemm(layout, precision)
        xp = _gather_rows(x, rowmap, ids.size(1))
        h = gemms.m_gemm(xp, w1)
        a = torch.empty((layout.rows, w2.size(2)), device=x.device, dtype=x.dtype)
        if layout.rows:
            _act[(layout.rows,)](h, a, b1, layout.m_indices, rowmap, a.size(1), tr.next_power_of_2(a.size(1)), b1 is not None)
        if scatter:
            out = gemms.m_gemm_scatter(a, w2, rowmap, x.size(0), ids.size(1), scores, b2)
        else:
            y = gemms.m_gemm(a, w2)
            out = torch.empty_like(x)
            if x.size(0):
                _combine[(x.size(0), tr.cdiv(x.size(1), 256))](
                    y, inverse, scores, ids, b2, out, x.size(1), ids.size(1), 256, True, b2 is not None)
        ctx.scatter = scatter
        ctx.save_for_backward(x, w1, w2, scores, ids, h, rowmap, inverse, b1, b2)
        ctx.gemms = gemms
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        x, w1, w2, scores, ids, h, rowmap, inverse, b1, b2 = ctx.saved_tensors
        gemms, layout = ctx.gemms, ctx.gemms.layout
        topk, d, i = ids.size(1), x.size(1), w2.size(2)
        dout = dout.contiguous()
        dop = _gather_rows(dout, rowmap, topk)
        u = gemms.m_gemm(dop, w2.transpose(1, 2))
        dh, ap = torch.empty_like(h), torch.empty((layout.rows, i), device=x.device, dtype=x.dtype)
        ds = torch.empty_like(scores, dtype=torch.float32)
        if layout.rows:
            _act_backward[(layout.rows,)](h, u, rowmap, scores, dh, ap, ds, i, tr.next_power_of_2(i))
        if b2 is not None and ids.numel():
            _bias_router_grad[(ids.numel(),)](dout, b2, ids, ds, d, topk, tr.next_power_of_2(d))
        dw2 = gemms.wgrad(dop, ap)
        del u, ap
        if ctx.scatter:
            dx = gemms.m_gemm_scatter(dh, w1.transpose(1, 2), rowmap, x.size(0), topk)
        else:
            dxp = gemms.m_gemm(dh, w1.transpose(1, 2))
        xp = _gather_rows(x, rowmap, topk)
        dw1 = gemms.wgrad(dh, xp)
        if not ctx.scatter:
            dx = torch.empty_like(x)
            if x.size(0):
                _combine[(x.size(0), tr.cdiv(d, 256))](dxp, inverse, scores, ids, None, dx, d, topk, 256, False, False)
        db1 = db2 = None
        if b1 is not None:
            db1 = torch.zeros_like(b1, dtype=torch.float32)
            db1.index_add_(0, layout.m_indices.long(), dh.float())
        if b2 is not None:
            # Reduce only real routes, without allocating another padded D-wide tensor.
            route_dout = dout.repeat_interleave(topk, dim=0).float() * scores.flatten()[:, None]
            db2 = torch.zeros_like(b2, dtype=torch.float32)
            db2.index_add_(0, ids.flatten().long(), route_dout)
        return dx, dw1.to(w1.dtype), dw2.to(w2.dtype), ds.to(scores.dtype), None, db1, db2, None, None


def deepgemm_swiglu(x, w1, w2, scores, expert_ids, *, bias1=None, bias2=None, precision="bf16", scatter=True):
    """Fixed top-k SwiGLU MoE, including gradients of X/W1/W2/scores/biases.

    X [T,D], W1 [E,2I,D] (gate/up interleaved), W2 [E,D,I], scores/IDs [T,K].
    BF16 tensors, FP32/BF16 scores, D/I divisible by 128; SM90 only. FP8 mode
    requantizes all GEMM operands (also wgrad) and uses straight-through gradients.
    scatter=True fuses FP32 atomic token reduction into the two output GEMMs.
    scatter=False retains materialized outputs for comparison. Atomics are nondeterministic.
    Routing counts require a host sync; CUDA graph capture and higher gradients
    are intentionally unsupported in this first implementation.
    """
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
                                    None if bias2 is None else bias2.contiguous(), precision, scatter)
