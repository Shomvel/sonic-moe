"""GPU acceptance for opt-in FP8 dH recomputation, independent of candidates."""
import gc
import math
import os

import pytest
import torch
import triton as tr

from deep_gemm.sonic_training import quantize_k_groups, quantize_rows
import sonicmoe.deepgemm as sm
from sonicmoe import _deepgemm_recompute as recompute
from test_deepgemm_training import metrics, reference


def _same_bytes(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def _strict_precision():
    if os.environ.get("NVIDIA_TF32_OVERRIDE") == "1":
        raise RuntimeError("FP32 numerical acceptance cannot use NVIDIA_TF32_OVERRIDE=1")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    return dict(float32_matmul_precision=torch.get_float32_matmul_precision(),
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                nvidia_tf32_override=os.environ.get("NVIDIA_TF32_OVERRIDE"))


def validate_recompute_quantizers():
    """Compare both FP8 layouts, AP and DS to the original BF16 intermediates."""
    torch.manual_seed(981)
    reports = []
    for i in (128, 512, 1536, 2048):
        for skew in (False, True):
            t, e, k = 257, 64, 8
            ids = torch.rand(t, e, device="cuda").topk(k, -1).indices
            if skew:
                ids.fill_(0)
                ids[-1, -1] = 2
            layout, rowmap, _ = sm._routing(ids, e)
            h = torch.randn(layout.rows, 2 * i, device="cuda", dtype=torch.bfloat16)
            u = torch.randn(layout.rows, i, device="cuda", dtype=torch.bfloat16)
            # Real forward masks padding; reproduce that contract for AP.
            h[rowmap < 0] = 0
            h_copy, u_copy = h.clone(), u.clone()
            for dtype in (torch.float32, torch.bfloat16):
                scores = (torch.rand(t, k, device="cuda") * 2 - 1).to(dtype)
                scores[:, 0] = 0
                scores[::3, 1] = 1.e-8
                score_copy = scores.clone()
                dh, ap = torch.empty_like(h), torch.empty_like(u)
                ds_ref = torch.empty_like(scores, dtype=torch.float32)
                sm._act_backward[(layout.rows,)](
                    h, u, rowmap, scores, dh, ap, ds_ref, i, tr.next_power_of_2(i))
                ds = torch.empty_like(ds_ref)
                encodings = (
                    (recompute._quantize_dh_k(h, u, scores, rowmap, layout), quantize_k_groups(dh, layout)),
                    (recompute._quantize_dh_rows(h, u, scores, rowmap, layout, ds=ds), quantize_rows(dh)),
                    (recompute._quantize_ap(h, scores, rowmap, layout), quantize_k_groups(ap, layout)),
                )
                for actual, expected in encodings:
                    for a, b in zip(actual, expected):
                        _same_bytes(a, b)
                _same_bytes(ds, ds_ref)
                _same_bytes(h, h_copy)
                _same_bytes(u, u_copy)
                _same_bytes(scores, score_copy)
                reports.append(dict(i=i, skew=skew, score_dtype=str(dtype),
                                    k_group_exact=True, row_group_exact=True, ap_exact=True,
                                    ds_exact=True, inputs_unchanged=True))
    return reports


def validate_recompute_large_offsets():
    """Exercise actual H/packed-dH addresses beyond 2**31 elements on H100.

    Called separately by the GPU runner; the small numerical suite does not
    allocate these buffers. Only one pair of quantized encodings is live at a
    time, and source snapshots contain seven rows rather than full H/U clones.
    """
    from deep_gemm.sonic_training import GroupedTrainingLayout

    routes, i = 2**19, 2048
    layout = GroupedTrainingLayout.from_counts([routes, 0, 1], torch.device("cuda"))
    tail_start = routes * (2 * i)
    assert tail_start == 2**31 and layout.rows * (2 * i) > 2**31
    rowmap = torch.full((layout.rows,), -1, device="cuda", dtype=torch.int64)
    rowmap[:routes + 1] = torch.arange(routes + 1, device="cuda", dtype=torch.int64)
    scores = torch.zeros(routes + 1, device="cuda", dtype=torch.float32)
    selected = torch.tensor([0, 127, 128, routes - 129, routes - 128, routes - 1, routes],
                            device="cuda", dtype=torch.int64)
    selected_scores = torch.tensor([1., -.25, 0., .75, -1.5, 0., -.5], device="cuda")
    scores[selected] = selected_scores
    h = torch.zeros((layout.rows, 2 * i), device="cuda", dtype=torch.bfloat16)
    u = torch.zeros((layout.rows, i), device="cuda", dtype=torch.bfloat16)
    gate = torch.linspace(-1.5, 1.5, i, device="cuda")
    for index, row in enumerate(selected.tolist()):
        h[row, 0::2] = (gate + index * .03125).bfloat16()
        h[row, 1::2] = (1.25 + gate * .125 + index * .0625).bfloat16()
        u[row] = (1. + gate * .0625 + index * .125).bfloat16()
    saved_h, saved_u = h[selected], u[selected]
    dh, ap = torch.empty_like(h), torch.empty_like(u)
    ds_ref = torch.empty_like(scores)
    print("RECOMPUTE_LARGE_BEGIN", "reference", flush=True)
    sm._act_backward[(layout.rows,)](h, u, rowmap, scores, dh, ap, ds_ref, i, i)
    torch.cuda.synchronize()

    # AP reads the same H source above 2**31. Its smaller packed destination
    # does not cross that boundary. Release AP before either dH encoding.
    print("RECOMPUTE_LARGE_BEGIN", "ap", flush=True)
    actual = recompute._quantize_ap(h, scores, rowmap, layout)
    expected = quantize_k_groups(ap, layout)
    for a, b in zip(actual, expected):
        _same_bytes(a, b)
    del a, b, actual, expected, ap
    torch.cuda.synchronize()

    # The third expert begins at exactly 2**31 packed FP8 dH elements; its
    # nonzero valid row makes an incorrect 32-bit source/destination observable.
    print("RECOMPUTE_LARGE_BEGIN", "k_group", flush=True)
    actual = recompute._quantize_dh_k(h, u, scores, rowmap, layout)
    expected = quantize_k_groups(dh, layout)
    for a, b in zip(actual, expected):
        _same_bytes(a, b)
    tail = actual[0][tail_start:].view(2 * i, layout.ks[-1])
    assert tail[:, 0].float().abs().max() > 0
    assert torch.count_nonzero(tail[:, 1:].float()) == 0
    del a, b, actual, expected, tail
    torch.cuda.synchronize()

    print("RECOMPUTE_LARGE_BEGIN", "row_group_ds", flush=True)
    ds = torch.empty_like(ds_ref)
    actual = recompute._quantize_dh_rows(h, u, scores, rowmap, layout, ds=ds)
    expected = quantize_rows(dh)
    for a, b in zip(actual, expected):
        _same_bytes(a, b)
    _same_bytes(ds, ds_ref)
    assert ds[128].abs() > 0 and ds[routes - 1].abs() > 0
    assert actual[0][routes].float().abs().max() > 0
    _same_bytes(h[selected], saved_h)
    _same_bytes(u[selected], saved_u)
    _same_bytes(scores[selected], selected_scores)
    del a, b, actual, expected, dh, h, u
    torch.cuda.synchronize()
    print("RECOMPUTE_LARGE_DONE", flush=True)
    return dict(counts=[routes, 0, 1], rows=layout.rows, i=i, width=2 * i,
                h_source_tail_start=tail_start, k_packed_tail_start=tail_start,
                max_h_element_offset=layout.rows * (2 * i) - 1,
                selected_rows=selected.tolist(), k_group_exact=True, row_group_exact=True,
                ap_exact=True, ds_exact=True, zero_score_ds_nonzero=True,
                source_samples_unchanged=True)


def _inputs(case):
    t, d, i, e, k = case["shape"]
    torch.manual_seed(1234)
    x = (torch.randn(t, d, device="cuda", dtype=torch.bfloat16) * .5).requires_grad_()
    w1 = (torch.randn(e, 2 * i, d, device="cuda", dtype=torch.bfloat16) / math.sqrt(d)).requires_grad_()
    if case.get("noncontiguous"):
        w1 = w1.transpose(1, 2).contiguous().transpose(1, 2).detach().requires_grad_()
    w2 = (torch.randn(e, d, i, device="cuda", dtype=torch.bfloat16) / math.sqrt(i)).requires_grad_()
    scores = torch.rand(t, k, device="cuda")
    if case.get("signed"):
        scores = scores * 2 - 1
    if case.get("zero"):
        scores[:, 0] = 0
        scores[::3, 1] = 1.e-8
    if case.get("bf16_scores"):
        scores = scores.bfloat16()
    scores.requires_grad_()
    ids = torch.rand(t, e, device="cuda").topk(k, -1).indices
    if case.get("skew"):
        ids.fill_(0)
        if t:
            ids[-1, -1] = 2
    params = [x, w1, w2, scores]
    if case.get("bias2"):
        params.append((torch.randn(e, d, device="cuda", dtype=torch.bfloat16) * .03).requires_grad_())
    return params, ids, torch.randn_like(x) * .1


def validate_recompute_case(case):
    _strict_precision()
    params, ids, dout = _inputs(case)
    names = ["output", "dx", "dw1", "dw2", "ds"] + (["db2"] if case.get("bias2") else [])
    def run(enabled):
        return sm.deepgemm_swiglu(*params[:4], ids, bias2=params[4] if case.get("bias2") else None,
                                  precision="fp8", recompute_dh=enabled)
    refs = [p.detach().float().requires_grad_() for p in params]
    ref_y = reference(*refs[:4], ids, None, refs[4] if case.get("bias2") else None)
    ref_g = torch.autograd.grad(ref_y, refs, dout.float())
    baseline = run(False)
    baseline_g = torch.autograd.grad(baseline, params, dout)
    saved = []
    def pack(tensor):
        saved.append(tensor)
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        y = run(True)
    saved_h = saved[5].clone()
    first = torch.autograd.grad(y, params, dout, retain_graph=True)
    second = torch.autograd.grad(y, params, dout, retain_graph=True)
    for actual, expected in zip(first, second):
        _same_bytes(actual, expected)
    _same_bytes(saved[5], saved_h)
    for p in params:
        p.grad = None
    y.backward(dout, retain_graph=True)
    y.backward(dout)
    for p, expected in zip(params, first):
        _same_bytes(p.grad, expected + expected)
        p.grad = None
    _same_bytes(saved[5], saved_h)
    vs_baseline, vs_fp32 = {}, {}
    for name, actual, base, fp32 in zip(names, [y, *first], [baseline, *baseline_g], [ref_y, *ref_g]):
        vs_baseline[name] = dict(exact=bool(torch.equal(actual, base)), **metrics(actual, base))
        if name == "dw1":
            # Native column tiling may alter accumulation; record actual error.
            assert vs_baseline[name]["rrmse"] < .01, (name, vs_baseline[name])
        else:
            _same_bytes(actual, base)
        vs_fp32[name] = metrics(actual, fp32)
        assert vs_fp32[name]["rrmse"] < .15, (name, vs_fp32[name])
    if case.get("skew"):
        assert torch.count_nonzero(first[1][1]) == 0
        assert torch.count_nonzero(first[2][1]) == 0
    if case.get("zero") and case["shape"][0]:
        assert first[3][:, 0].abs().max() > 0, "zero-score routes lost router gradients"
    return dict(case=case, vs_baseline=vs_baseline, vs_fp32=vs_fp32,
                retain_graph_exact=True, accumulation_exact=True, saved_h_unchanged=True)


def validate_recompute_invalid():
    params, ids, _ = _inputs(dict(shape=(1, 128, 128, 4, 2)))
    bias1 = torch.zeros(4, 256, device="cuda", dtype=torch.bfloat16)
    cases = (
        ("bf16", dict(precision="bf16")),
        ("atomic", dict(precision="fp8", reduction="atomic")),
        ("materialize", dict(precision="fp8", reduction="materialize")),
        ("scatter_true", dict(precision="fp8", scatter=True)),
        ("scatter_false", dict(precision="fp8", scatter=False)),
        ("bias1", dict(precision="fp8", bias1=bias1)),
    )
    reports = []
    for name, options in cases:
        with pytest.raises(ValueError) as exc:
            sm.deepgemm_swiglu(*params, ids, recompute_dh=True, **options)
        reports.append(dict(case=name, rejected=True, error=str(exc.value)))
    return reports


CASES = (
    dict(shape=(129, 2176, 128, 4, 2), signed=True, zero=True, noncontiguous=True),
    dict(shape=(129, 384, 512, 64, 8), bias2=True, skew=True, signed=True, zero=True),
    dict(shape=(0, 2176, 128, 4, 2), bias2=True),
    dict(shape=(257, 384, 512, 64, 8)),
    dict(shape=(129, 384, 128, 256, 8), signed=True, zero=True),
    dict(shape=(129, 384, 128, 256, 16), signed=True, zero=True),
    dict(shape=(129, 384, 512, 64, 16), signed=True, zero=True, bf16_scores=True),
    dict(shape=(129, 2176, 1536, 4, 2), zero=True, noncontiguous=True, bf16_scores=True),
    dict(shape=(129, 4096, 1536, 64, 8), signed=True, zero=True),
)


def validate_recompute():
    """Return a JSON-friendly report for the production integration GPU job."""
    result = dict(reference_precision=_strict_precision(), kernel_checks=validate_recompute_quantizers(),
                  invalid_combinations=validate_recompute_invalid(), numerical=[])
    for case in CASES:
        print("RECOMPUTE_CASE", case, flush=True)
        result["numerical"].append(validate_recompute_case(case))
        gc.collect()
        torch.cuda.empty_cache()
    result["passed"] = True
    return result


def test_recompute_quantizers():
    validate_recompute_quantizers()


def test_recompute_invalid_combinations():
    validate_recompute_invalid()


@pytest.mark.parametrize("case", CASES)
def test_recompute_gradients(case):
    validate_recompute_case(case)
