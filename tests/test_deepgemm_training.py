"""GPU numerical acceptance against an independent FP32 PyTorch expression."""
import math
import os

import pytest
import torch

from sonicmoe.deepgemm import deepgemm_swiglu

# These tests use an independent FP32 reference, including FP32 GEMMs.
if os.environ.get("NVIDIA_TF32_OVERRIDE") == "1":
    raise RuntimeError("FP32 numerical acceptance requires NVIDIA_TF32_OVERRIDE unset or 0")
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False


def reference(x, w1, w2, s, ids, b1=None, b2=None):
    out = torch.zeros_like(x, dtype=torch.float32)
    for e in range(w1.size(0)):
        t, k = (ids == e).nonzero(as_tuple=True)
        h = x[t].float() @ w1[e].float().T
        if b1 is not None:
            h = h + b1[e].float()
        a = torch.nn.functional.silu(h[:, ::2]) * h[:, 1::2]
        y = a @ w2[e].float().T
        if b2 is not None:
            y = y + b2[e].float()
        out = out.index_add(0, t, y * s[t, k, None].float())
    return out


def metrics(actual, expected):
    a, b = actual.float().flatten(), expected.float().flatten()
    assert torch.isfinite(a).all(), "nonfinite output/gradient"
    error = (a - b).norm() / b.norm().clamp_min(1e-8)
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=0) if b.norm() > 1e-8 else a.new_tensor(1.)
    return {"rrmse": error.item(), "cosine": cosine.item(), "max_abs": (a-b).abs().max().item() if a.numel() else 0.}


def validate_case(precision, *, t=257, bias=False, skew=False, zero_scores=False, scatter=None,
                  k=2, signed=False, reduction="chunked", chunk_size=None, d=256, i=512,
                  e=None, random_routes=False, backend="deepgemm"):
    torch.manual_seed(1234)
    e = max(4, k) if e is None else e
    if not 1 <= k <= e or backend not in ("deepgemm", "sonic"):
        raise ValueError("expected 1 <= k <= e and deepgemm/sonic backend")
    if backend == "sonic" and precision != "bf16":
        raise ValueError("the original Sonic comparison uses BF16")
    x = (torch.randn(t, d, device="cuda", dtype=torch.bfloat16) * .5).requires_grad_()
    w1 = (torch.randn(e, 2*i, d, device="cuda", dtype=torch.bfloat16) / math.sqrt(d)).requires_grad_()
    w2 = (torch.randn(e, d, i, device="cuda", dtype=torch.bfloat16) / math.sqrt(i)).requires_grad_()
    # Strided inputs exercise parameter layouts used by wrappers.
    if backend == "deepgemm":
        w1 = w1.transpose(1, 2).contiguous().transpose(1, 2).detach().requires_grad_()
    s = torch.rand(t, k, device="cuda", dtype=torch.float32)
    if signed:
        s = s * 2 - 1
    if zero_scores:
        s[:, 0] = 0
        if k > 1:
            s[::3, 1] = 1e-8
    s.requires_grad_()
    ids = (torch.rand(t, e, device="cuda").topk(k, -1).indices if random_routes else
           torch.arange(t*k, device="cuda").view(t, k) % e)
    if skew:
        ids.fill_(0)  # Duplicate routes remain distinct; most experts are empty.
        if t and e > 2:
            ids[-1, -1] = 2
    elif random_routes and k > 1:
        ordered = ids.sort(-1).values
        assert torch.all(ordered[:, 1:] != ordered[:, :-1])
    b1 = (torch.randn(e, 2*i, device="cuda", dtype=torch.bfloat16) * .03).requires_grad_() if bias else None
    b2 = (torch.randn(e, d, device="cuda", dtype=torch.bfloat16) * .03).requires_grad_() if bias else None
    params = [x, w1, w2, s] + ([b1, b2] if bias else [])
    ref_params = [v.detach().clone().float().requires_grad_() for v in params]
    rx, rw1, rw2, rs = ref_params[:4]
    rb1, rb2 = ref_params[4:] if bias else (None, None)
    ref = reference(rx, rw1, rw2, rs, ids, rb1, rb2)
    saved_shapes = []
    def pack(v):
        saved_shapes.append(list(v.shape))
        return v
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda v: v):
        if backend == "sonic":
            from sonicmoe.functional import moe_general_routing_inputs
            from sonicmoe.enums import ActivationType
            out = moe_general_routing_inputs(
                x=x, router_scores=s.flatten(),
                token_indices=torch.arange(t, device="cuda", dtype=torch.int32).repeat_interleave(k),
                expert_indices=ids.flatten().int(), w1=w1.permute(1, 2, 0),
                b1=b1, w2=w2.permute(1, 2, 0), b2=b2, E=e,
                stream_id=torch.cuda.current_stream().cuda_stream,
                activation_type=ActivationType.SWIGLU)[0]
        else:
            out = deepgemm_swiglu(x, w1, w2, s, ids, bias1=b1, bias2=b2, precision=precision,
                                  scatter=scatter, reduction=reduction, chunk_size=chunk_size)
    do = torch.randn_like(out) * .1
    grads = torch.autograd.grad(out, params, do)
    ref_grads = torch.autograd.grad(ref, ref_params, do.float())
    report = {"output": metrics(out, ref)}
    for name, a, b in zip(["dx", "dw1", "dw2", "ds", "db1", "db2"], grads, ref_grads):
        report[name] = metrics(a, b)
    tolerance = .025 if precision == "bf16" else .15
    for name, value in report.items():
        limit = .03 if backend == "sonic" and name != "output" else tolerance
        assert value["rrmse"] < limit, (backend, precision, t, e, k, bias, skew, zero_scores, name, value)
    if skew and e > 1:
        assert torch.count_nonzero(grads[1][1]) == 0
        assert torch.count_nonzero(grads[2][1]) == 0
    if zero_scores and t:
        assert grads[3][:, 0].abs().max() > 0, "zero route scores must still receive router gradients"
    if backend == "deepgemm":
        assert len(saved_shapes) == 8 + (2 if bias else 0), saved_shapes
        # Neither route-expanded activation nor expert output is retained.
        assert not t or not any(shape == [saved_shapes[5][0], i] for shape in saved_shapes)
        assert not t or not any(shape == [saved_shapes[5][0], d] for shape in saved_shapes)
    report["saved_shapes"] = saved_shapes
    report["case"] = dict(backend=backend, precision=precision, t=t, bias=bias, skew=skew,
                          zero_scores=zero_scores, scatter=scatter, e=e, k=k, signed=signed,
                          reduction=reduction, chunk_size=chunk_size, d=d, i=i,
                          random_routes=random_routes, distinct_routes=not skew)
    return report


def validate_sparse_training(steps=20):
    """Fixed distinct E64/K8 routes, with a genuinely FP32 SGD reference."""
    torch.manual_seed(222)
    t, d, i, e, k = 129, 256, 512, 64, 8
    initial = [torch.randn(t, d, device="cuda", dtype=torch.bfloat16) * .5,
               torch.randn(e, 2*i, d, device="cuda", dtype=torch.bfloat16) / math.sqrt(d),
               torch.randn(e, d, i, device="cuda", dtype=torch.bfloat16) / math.sqrt(i),
               torch.randn(t, k, device="cuda", dtype=torch.float32)]
    ids = torch.rand(t, e, device="cuda").topk(k, -1).indices
    ordered = ids.sort(-1).values
    assert torch.all(ordered[:, 1:] != ordered[:, :-1])
    target = torch.randn(t, d, device="cuda", dtype=torch.float32) * .1
    final, runs = {}, []
    low_precision_modes = ("bf16", "fp8", "fp8_recompute")
    for mode in ("reference", *low_precision_modes):
        # FP32 master weights give both paths the same SGD update precision;
        # low-precision forward casts still backpropagate through their inputs.
        x = initial[0].float() if mode == "reference" else initial[0]
        w1, w2, logits = [v.detach().clone().float() for v in initial[1:]]
        trainable = [v.requires_grad_() for v in (w1, w2, logits)]
        optim = torch.optim.SGD(trainable, lr=20.)

        def forward():
            scores = logits.softmax(-1)
            if mode == "reference":
                return reference(x, w1, w2, scores, ids)
            return deepgemm_swiglu(x, w1.bfloat16(), w2.bfloat16(), scores, ids,
                                  precision="bf16" if mode == "bf16" else "fp8",
                                  recompute_dh=mode == "fp8_recompute")

        losses = []
        for _ in range(steps):
            optim.zero_grad(set_to_none=True)
            loss = (forward().float() - target).square().mean()
            loss.backward()
            optim.step()
            losses.append(loss.item())
        with torch.no_grad():
            out = forward()
            post_update_loss = (out.float() - target).square().mean().item()
        assert all(math.isfinite(v) for v in [*losses, post_update_loss])
        assert post_update_loss < losses[0], (mode, losses, post_update_loss)
        final[mode] = (out.detach(), *(v.detach() for v in trainable))
        runs.append(dict(mode=mode, losses=losses, post_update_loss=post_update_loss))
    drift = {mode: metrics(final[mode][0], final["reference"][0]) for mode in low_precision_modes}
    assert drift["bf16"]["rrmse"] < .05, drift
    assert drift["fp8"]["rrmse"] < .15, drift
    assert drift["fp8_recompute"]["rrmse"] < .15, drift
    for actual, expected in zip(final["fp8_recompute"], final["fp8"]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    parameter_drift = {mode: {name: metrics(a, b) for name, a, b in
                             zip(("w1", "w2", "logits"), final[mode][1:], final["reference"][1:])}
                       for mode in low_precision_modes}
    return dict(shape=dict(t=t, d=d, i=i, e=e, k=k), distinct_routes=True,
                reference_dtype="float32", optimizer_dtype="float32", learning_rate=20.,
                steps=steps, runs=runs, drift=drift,
                parameter_drift=parameter_drift)


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
@pytest.mark.parametrize("reduction", ["chunked", "materialize", "atomic"])
def test_sparse_full_gradients(precision, reduction):
    validate_case(precision, t=129, e=64, k=8, random_routes=True,
                  bias=True, zero_scores=True, signed=True, reduction=reduction)


def test_sparse_sonic_reference():
    validate_case("bf16", t=129, e=64, k=8, random_routes=True, backend="sonic")


def test_sparse_sgd():
    validate_sparse_training()


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
@pytest.mark.parametrize("bias,skew,zero_scores", [(False,False,False), (True,True,True)])
def test_against_fp32(precision, bias, skew, zero_scores):
    validate_case(precision, bias=bias, skew=skew, zero_scores=zero_scores)


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
def test_empty_batch(precision):
    validate_case(precision, t=0)


def validate_wrapper(precision="bf16", *, recompute_dh=False):
    from sonicmoe import MoE, KernelBackendMoE
    from sonicmoe.enums import ActivationType
    torch.manual_seed(17)
    model=MoE(64 if recompute_dh else 4,8 if recompute_dh else 2,256,512,
              ActivationType.SWIGLU,not recompute_dh,.04).cuda().bfloat16().train()
    x=torch.randn(2,128,256,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    backend = KernelBackendMoE.deepgemm_bf16 if precision == "bf16" else KernelBackendMoE.deepgemm_fp8
    y,aux=model(x,backend,recompute_dh=recompute_dh)
    ref,raux=model(x,KernelBackendMoE.torch)
    do=torch.randn_like(y)*.1
    params=[x,*model.parameters()]
    grads=torch.autograd.grad((y.float()*do).sum()+aux*.01,params)
    refs=torch.autograd.grad((ref.float()*do).sum()+raux*.01,params)
    result={'output':metrics(y,ref),'aux':metrics(aux,raux)}
    for name,a,b in zip(['x',*[name for name,_ in model.named_parameters()]],grads,refs):
        result[name]=metrics(a,b)
    tolerance = .03 if precision == "bf16" else .15
    assert all(v['rrmse']<tolerance for v in result.values()),result
    if recompute_dh:
        base,base_aux=model(x,backend)
        base_grads=torch.autograd.grad((base.float()*do).sum()+base_aux*.01,params)
        for actual,expected in zip((y,aux,*grads),(base,base_aux,*base_grads)):
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        for unsupported in (KernelBackendMoE.torch,KernelBackendMoE.sonicmoe,
                            KernelBackendMoE.deepgemm_bf16):
            with pytest.raises(ValueError,match="recompute_dh"):
                model(x,unsupported,recompute_dh=True)
    return result


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
def test_moe_wrapper(precision):
    validate_wrapper(precision)


def test_moe_recompute_wrapper():
    validate_wrapper("fp8",recompute_dh=True)


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
def test_scatter_high_topk(precision):
    validate_case(precision, k=8, bias=True, skew=True, signed=True)


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
def test_column_workspace(precision):
    validate_case(precision, bias=True, skew=True, zero_scores=True,
                  reduction="chunked", chunk_size=256, d=384)


def validate_routing():
    from sonicmoe.deepgemm import _routing
    cases = ((0,4,2),(17,1,1),(257,4,2),(2057,17,2),
             (257,64,8),(2057,256,8),(137,64,16))
    for t,e,k in cases:
        torch.manual_seed(t)
        ids = (torch.rand(t,e,device="cuda").topk(k,-1).indices if e >= 64 else
               torch.randint(e,(t,k),device="cuda",dtype=torch.int64))
        if (t,e,k)==(257,4,2):
            ids.fill_(0)
            ids[-1,-1]=2
        layout,rowmap,inverse=_routing(ids,e)
        flat=ids.flatten()
        counts=torch.bincount(flat,minlength=e).cpu().tolist()
        order=flat.argsort(stable=True)
        expected=torch.full_like(rowmap,-1)
        logical=0
        for group,count in enumerate(counts):
            start=sum(layout.ks[:group])
            expected[start:start+count]=order[logical:logical+count]
            logical+=count
        torch.testing.assert_close(rowmap,expected,rtol=0,atol=0)
        if t:
            torch.testing.assert_close(rowmap[inverse],torch.arange(t*k,device='cuda'),rtol=0,atol=0)
        for group,size in enumerate(layout.ks):
            start=sum(layout.ks[:group])
            assert torch.all(layout.m_indices[start:start+size]==group)
    for invalid in (-1,4,2**32):
        ids=torch.tensor([[0,invalid]],device='cuda',dtype=torch.int64)
        try:
            _routing(ids,4)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid expert {invalid}")
    return {"stable_routing_cases":len(cases),"sparse_distinct_cases":3,"invalid_ids_rejected":3}


def test_stable_routing():
    validate_routing()


def validate_large_gather_offsets():
    """Top-16 padded route buffers may exceed signed-int32 element addressing."""
    from deep_gemm.sonic_training import GroupedTrainingLayout
    from sonicmoe.deepgemm import _gather_rows, _bias_grad
    width, routes, topk = 4096, 2**19, 16
    layout = GroupedTrainingLayout.from_counts([routes, 0, 1], torch.device('cuda'))
    tokens = torch.tensor([1., 2.], device='cuda', dtype=torch.bfloat16)[:, None].expand(2, width).contiguous()
    rowmap = torch.zeros(layout.rows, device='cuda', dtype=torch.int64)
    rowmap[routes] = topk
    rowmap[routes+1:] = -1
    expanded = _gather_rows(tokens, rowmap, topk)
    torch.testing.assert_close(expanded[0], tokens[0], rtol=0, atol=0)
    torch.testing.assert_close(expanded[routes-1], tokens[0], rtol=0, atol=0)
    torch.testing.assert_close(expanded[routes], tokens[1], rtol=0, atol=0)
    assert torch.count_nonzero(expanded[routes+1:]) == 0
    # This also crosses the 2**31 source index in the non-gather bias path.
    actual = _bias_grad(expanded, layout, rowmap, topk)
    expected = torch.tensor([float(routes), 0., 2.], device='cuda', dtype=torch.bfloat16)[:, None].expand_as(actual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.cuda.synchronize()
    return {'rows': layout.rows, 'width': width, 'tail_start': routes*width}


def test_large_gather_offsets():
    validate_large_gather_offsets()
