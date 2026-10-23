"""FP8 backward with recomputed, directly quantized activation gradients.

The fixed schedule computes dW1 first, dX second, and dW2 last. K-group and
row-group FP8 dH layouts never coexist, and no full BF16 dH is materialized.
All quantizers preserve the original BF16 rounding boundaries. Saved tensors
are read-only, including when the graph is retained for another backward.
"""
import torch
import triton as tr
import triton.language as tl

from deep_gemm.sonic_training import quantize_k_groups, quantize_weights
from . import deepgemm as sm


@tr.jit
def _dh_values(g, up, v, score):
    sig = tl.sigmoid(g)
    grad = v * score
    dg = grad * up * sig * (1 + g * (1 - sig))
    du = grad * g * sig
    # This is a semantic rounding boundary, matching the BF16 stores in
    # _act_backward; quantizing the FP32 expressions directly changes training.
    return dg.to(tl.bfloat16).to(tl.float32), du.to(tl.bfloat16).to(tl.float32)


@tr.jit
def _dh_k_quant(H, U, SCORES, MAP, OFFSETS, Q, S, I: tl.constexpr, BN: tl.constexpr):
    expert = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1).to(tl.int64)
    feature_block = tl.program_id(2).to(tl.int64)
    start = tl.load(OFFSETS + expert).to(tl.int64)
    end = tl.load(OFFSETS + expert + 1).to(tl.int64)
    local_row = block * 128 + tl.arange(0, 128).to(tl.int64)
    row = start + local_row
    feature = feature_block * (BN // 2) + tl.arange(0, BN // 2).to(tl.int64)
    route = tl.load(MAP + row, row < end, -1)
    score = tl.load(SCORES + route, (row < end) & (route >= 0), 0).to(tl.float32)
    mask = (row[:, None] < end) & (feature[None, :] < I)
    address = row[:, None] * (2 * I) + 2 * feature[None, :]
    g = tl.load(H + address, mask, 0).to(tl.float32)
    up = tl.load(H + address + 1, mask, 0).to(tl.float32)
    v = tl.load(U + row[:, None] * I + feature[None, :], mask, 0).to(tl.float32)
    dg, du = _dh_values(g, up, v, score[:, None])
    dh = tl.reshape(tl.join(dg, du), (128, BN))
    scale = tl.maximum(tl.max(tl.abs(dh), axis=0), 1.e-4) / 448.
    column = feature_block * BN + tl.arange(0, BN).to(tl.int64)
    dest = start * (2 * I) + column[None, :] * (end - start) + local_row[:, None]
    tl.store(Q + dest, (dh / scale[None, :]).to(Q.dtype.element_ty),
             (row[:, None] < end) & (column[None, :] < 2 * I))
    tl.store(S + column + (start // 128 + block) * (2 * I), scale,
             (start + block * 128 < end) & (column < 2 * I))


def _quantize_dh_k(h, u, scores, rowmap, layout):
    block_features = 64
    n = h.size(1)
    q = torch.empty((layout.rows * n,), device=h.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty_strided((n, layout.rows // 128), (1, n),
                                 device=h.device, dtype=torch.float32)
    if layout.rows:
        _dh_k_quant[(len(layout.ks), max(layout.ks) // 128, tr.cdiv(n, block_features))](
            h, u, scores, rowmap, layout.offsets, q, scales, n // 2, block_features,
            num_warps=4)
    return q, scales


def _project_prequantized(gemms, qa, weights, inverse, ids, columns):
    """dX consumes row-quantized dH; routing weights are already inside dH."""
    from deep_gemm import m_grouped_fp8_gemm_nt_contiguous

    t, topk = ids.shape
    out = torch.empty((t, weights.size(1)), device=weights.device, dtype=torch.bfloat16)
    if not t:
        return out
    columns = min(columns, weights.size(1))
    workspace = torch.empty((gemms.layout.rows, columns), device=weights.device, dtype=torch.bfloat16)
    for start in range(0, weights.size(1), columns):
        w = weights[:, start:start + columns]
        tile = workspace[:, :w.size(1)]
        m_grouped_fp8_gemm_nt_contiguous(
            qa, quantize_weights(w), tile, gemms.layout.m_indices,
            recipe=(1, 128, 128), disable_ue8m0_cast=True)
        sm._combine_columns[(tr.cdiv(t, 4), tr.cdiv(tile.size(1), 256))](
            tile, inverse, None, ids, None, out, t, weights.size(1), tile.size(1),
            tile.stride(0), start, topk, 256, 4, False, False)
    return out


@tr.jit
def _dh_row_quant_ds(H, U, SCORES, MAP, Q, S, DS,
                      I: tl.constexpr, B: tl.constexpr, SS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    route = tl.load(MAP + row)
    feature = tl.arange(0, B).to(tl.int64)
    address = row * (2 * I) + 2 * feature
    g = tl.load(H + address, feature < I, 0).to(tl.float32)
    up = tl.load(H + address + 1, feature < I, 0).to(tl.float32)
    v = tl.load(U + row * I + feature, feature < I, 0).to(tl.float32)
    score = tl.load(SCORES + route, route >= 0, 0).to(tl.float32)
    # Keep exactly the original B-wide reduction and BF16 activation rounding.
    # Computing this from FP8 values, or a sum of block partials, changes DS.
    sig = tl.sigmoid(g)
    a = (g * sig * up).to(tl.bfloat16).to(tl.float32)
    ds = tl.sum(tl.where(feature < I, v * a, 0), 0)
    tl.store(DS + route, ds, route >= 0)
    grad = v * score
    dg = (grad * up * sig * (1 + g * (1 - sig))).to(tl.bfloat16).to(tl.float32)
    du = (grad * g * sig).to(tl.bfloat16).to(tl.float32)
    # join preserves [dg0,du0,dg1,du1,...]. Reshaping does not change the
    # quantization axis: each row below is exactly 128 interleaved features.
    dh = tl.reshape(tl.join(dg, du), (B // 64, 128))
    scale = tl.maximum(tl.max(tl.abs(dh), axis=1), 1.e-4) / 448.
    block = tl.arange(0, B // 64).to(tl.int64)
    column = block[:, None] * 128 + tl.arange(0, 128)[None, :].to(tl.int64)
    tl.store(Q + row * (2 * I) + column,
             (dh / scale[:, None]).to(Q.dtype.element_ty), column < 2 * I)
    tl.store(S + row + block * SS, scale, block < tl.cdiv(2 * I, 128))


def _quantize_dh_rows(h, u, scores, rowmap, layout, *, ds):
    n = h.size(1)
    q = torch.empty((layout.rows, n), device=h.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty_strided((layout.rows, tr.cdiv(n, 128)),
                                 (1, tr.cdiv(layout.rows, 4) * 4),
                                 device=h.device, dtype=torch.float32)
    if layout.rows:
        _dh_row_quant_ds[(layout.rows,)](
            h, u, scores, rowmap, q, scales, ds, n // 2,
            tr.next_power_of_2(n // 2), scales.stride(1), num_warps=4)
    return q, scales


@tr.jit
def _ap_k_quant(H, SCORES, MAP, OFFSETS, Q, S, I: tl.constexpr, BN: tl.constexpr):
    expert = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1).to(tl.int64)
    nc = tl.program_id(2).to(tl.int64)
    # Both source H and transposed grouped destinations may exceed 2**31
    # elements. Cast offsets/indices before multiplying by feature strides.
    start = tl.load(OFFSETS + expert).to(tl.int64)
    end = tl.load(OFFSETS + expert + 1).to(tl.int64)
    row = block * 128 + tl.arange(0, 128).to(tl.int64)
    col = nc * BN + tl.arange(0, BN).to(tl.int64)
    valid = start + row < end
    route = tl.load(MAP + start + row, valid, -1)
    mask = valid[:, None] & (col[None, :] < I)
    address = (start + row[:, None]) * (2 * I) + 2 * col[None, :]
    g = tl.load(H + address, mask, 0).to(tl.float32)
    u = tl.load(H + address + 1, mask, 0).to(tl.float32)
    score = tl.load(SCORES + route, valid & (route >= 0), 0).to(tl.float32)
    # Match _act_backward: first A is rounded, then score*A is rounded to AP.
    a = (g * tl.sigmoid(g) * u).to(tl.bfloat16).to(tl.float32)
    ap = (a * score[:, None]).to(tl.bfloat16).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(ap), axis=0), 1.e-4) / 448.
    dest = start * I + col[None, :] * (end - start) + row[:, None]
    tl.store(Q + dest, (ap / scale[None, :]).to(Q.dtype.element_ty), mask)
    tl.store(S + col + (start // 128 + block) * I, scale,
             (start + block * 128 < end) & (col < I))


def _quantize_ap(h, scores, rowmap, layout):
    block_features = 32
    i = h.size(1) // 2
    q = torch.empty((layout.rows * i,), device=h.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty_strided((i, layout.rows // 128), (1, i),
                                 device=h.device, dtype=torch.float32)
    if layout.rows:
        _ap_k_quant[(len(layout.ks), max(layout.ks) // 128, tr.cdiv(i, block_features))](
            h, scores, rowmap, layout.offsets, q, scales, i, block_features, num_warps=4)
    return q, scales


def recompute_backward(ctx, dout):
    """Return dX/dW1/dW2/dScores/db2 for the validated FP8 chunked path.

    The public entry point enforces FP8, chunked reduction and no bias1. DS
    uses unweighted BF16 U and BF16 A, with the original FP32 reduction tree.
    Native dW1 writes its final column slices without an extra output tile.
    """
    from deep_gemm import k_grouped_fp8_gemm_nt_contiguous

    x, w1, w2, scores, ids, h, rowmap, inverse, _, b2 = ctx.saved_tensors
    gemms, layout = ctx.gemms, ctx.gemms.layout
    topk, d = ids.size(1), x.size(1)
    dout = dout.contiguous()
    u = gemms.m_gemm_gather(dout, w2.transpose(1, 2), rowmap, topk)
    qdh = _quantize_dh_k(h, u, scores, rowmap, layout)
    dw1 = gemms.wgrad_gather_rhs_columns(
        None, x, rowmap, topk, quantized_lhs=qdh, output_dtype=w1.dtype)
    del qdh
    ds = torch.empty_like(scores, dtype=torch.float32)
    qdh = _quantize_dh_rows(h, u, scores, rowmap, layout, ds=ds)
    del u
    if b2 is not None and ids.numel():
        sm._bias_router_grad[(ids.numel(),)](
            dout, b2, ids, ds, d, topk, tr.next_power_of_2(d))
    dx = _project_prequantized(gemms, qdh, w1.transpose(1, 2), inverse, ids, ctx.chunk_size)
    del qdh
    dw2 = torch.empty_like(w2, memory_format=torch.contiguous_format)
    if layout.rows:
        qap = _quantize_ap(h, scores, rowmap, layout)
        qdo = quantize_k_groups(dout, layout, rowmap, topk)
        k_grouped_fp8_gemm_nt_contiguous(
            qdo, qap, dw2, list(layout.ks), layout.k_sizes, None, recipe=(1, 1, 128))
        del qap, qdo
    else:
        dw2.zero_()
    db2 = sm._bias_grad(dout, layout, rowmap, topk, scores) if b2 is not None else None
    return dx, dw1, dw2, ds.to(scores.dtype), db2
