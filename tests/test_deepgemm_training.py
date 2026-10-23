"""GPU numerical acceptance against an independent FP32 PyTorch expression."""
import math

import pytest
import torch

from sonicmoe.deepgemm import deepgemm_swiglu


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


def validate_case(precision, *, t=257, bias=False, skew=False, zero_scores=False, scatter=True, k=2, signed=False):
    torch.manual_seed(1234)
    e, d, i = max(4, k), 256, 512
    x = (torch.randn(t, d, device="cuda", dtype=torch.bfloat16) * .5).requires_grad_()
    w1 = (torch.randn(e, 2*i, d, device="cuda", dtype=torch.bfloat16) / math.sqrt(d)).requires_grad_()
    w2 = (torch.randn(e, d, i, device="cuda", dtype=torch.bfloat16) / math.sqrt(i)).requires_grad_()
    # Strided inputs exercise parameter layouts used by wrappers.
    w1 = w1.transpose(1, 2).contiguous().transpose(1, 2).detach().requires_grad_()
    s = torch.rand(t, k, device="cuda", dtype=torch.float32)
    if signed:
        s = s * 2 - 1
    if zero_scores:
        s[:, 0] = 0
        s[::3, 1] = 1e-8
    s.requires_grad_()
    ids = torch.arange(t*k, device="cuda").view(t, k) % e
    if skew:
        ids.fill_(0)  # experts 1..3 empty; duplicate routes remain distinct.
        if t:
            ids[-1, -1] = 2
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
        out = deepgemm_swiglu(x, w1, w2, s, ids, bias1=b1, bias2=b2, precision=precision, scatter=scatter)
    do = torch.randn_like(out) * .1
    grads = torch.autograd.grad(out, params, do)
    ref_grads = torch.autograd.grad(ref, ref_params, do.float())
    report = {"output": metrics(out, ref)}
    for name, a, b in zip(["dx", "dw1", "dw2", "ds", "db1", "db2"], grads, ref_grads):
        report[name] = metrics(a, b)
    tolerance = .025 if precision == "bf16" else .15
    for name, value in report.items():
        assert value["rrmse"] < tolerance, (precision, t, bias, skew, zero_scores, name, value)
    if skew:
        assert torch.count_nonzero(grads[1][1]) == 0
        assert torch.count_nonzero(grads[2][1]) == 0
    if zero_scores and t:
        assert grads[3][:, 0].abs().max() > 0, "zero route scores must still receive router gradients"
    assert len(saved_shapes) == 8 + (2 if bias else 0), saved_shapes
    # In this shape, neither [padded routes,I] activation nor [padded routes,D]
    # expert output may be retained across forward/backward.
    assert not t or not any(shape == [saved_shapes[5][0], i] for shape in saved_shapes)
    assert not t or not any(shape == [saved_shapes[5][0], d] for shape in saved_shapes)
    report["saved_shapes"] = saved_shapes
    report["case"] = dict(precision=precision, t=t, bias=bias, skew=skew, zero_scores=zero_scores, scatter=scatter, k=k, signed=signed)
    return report


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
@pytest.mark.parametrize("bias,skew,zero_scores", [(False,False,False), (True,True,True)])
def test_against_fp32(precision, bias, skew, zero_scores):
    validate_case(precision, bias=bias, skew=skew, zero_scores=zero_scores)


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
def test_empty_batch(precision):
    validate_case(precision, t=0)


def validate_wrapper(precision="bf16"):
    from sonicmoe import MoE, KernelBackendMoE
    from sonicmoe.enums import ActivationType
    torch.manual_seed(17)
    model=MoE(4,2,256,512,ActivationType.SWIGLU,True,.04).cuda().bfloat16().train()
    x=torch.randn(2,128,256,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    backend = KernelBackendMoE.deepgemm_bf16 if precision == "bf16" else KernelBackendMoE.deepgemm_fp8
    y,aux=model(x,backend)
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
    return result


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
def test_moe_wrapper(precision):
    validate_wrapper(precision)


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
def test_scatter_high_topk(precision):
    validate_case(precision, k=8, bias=True, skew=True, signed=True)
