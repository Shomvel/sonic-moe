"""Acceptance and synchronized wall-time benchmark (includes routing and packing)."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import google.protobuf
from google.protobuf.internal.api_implementation import Type

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_deepgemm_training import validate_case, validate_wrapper, validate_routing, validate_sparse_training, metrics
from sonicmoe.deepgemm import deepgemm_swiglu, _routing, _project
from deep_gemm.sonic_training import GroupedTrainingGemm


MAIN_SHAPES = ((4096,4096,512,64,8), (32768,4096,512,64,8),
               (32768,4096,512,256,8), (32768,4096,512,64,16))
EXTENDED_SHAPES = ((32768,4096,1536,64,8),)
DEBUG_SHAPES = ((512,256,512,4,2), (2048,1024,2048,8,2),
                (4096,4096,2048,8,2), (4096,4096,512,8,8),
                (32768,4096,2048,8,2), (32768,4096,512,8,8))


def benchmark_shapes(debug_shapes=False, extended_shapes=False):
    return MAIN_SHAPES + (EXTENDED_SHAPES if extended_shapes else ()) + (DEBUG_SHAPES if debug_shapes else ())


def time_training(fn, params, do, iterations):
    def step():
        out = fn()
        torch.autograd.grad(out, params, do)
    for _ in range(2):
        step()
    torch.cuda.synchronize()
    # Capture memory while forward output and saved autograd tensors coexist.
    initial = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    retained_forward = torch.cuda.memory_allocated() - initial
    forward_peak = torch.cuda.max_memory_allocated() - initial
    torch.autograd.grad(out, params, do)
    del out
    torch.cuda.synchronize()
    initial = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            step()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / iterations)
    return {"median_ms": statistics.median(samples), "samples_ms": samples,
            "timing": "synchronized_wall", "retained_forward_bytes": retained_forward,
            "forward_peak_bytes": forward_peak,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "incremental_peak_bytes": torch.cuda.max_memory_allocated() - initial}


def projection_benchmark(iterations, debug_shapes=False, extended_shapes=False):
    results = []
    shapes = benchmark_shapes(debug_shapes, extended_shapes)
    if debug_shapes:
        shapes += ((32768,4096,256,8,8),)
    for shape in shapes:
        results.extend(_projection_case(shape, iterations))
    return results


def _projection_case(shape, iterations):
    """Isolate the removed D-wide output; inputs/routing are already grouped."""
    results = []
    torch.manual_seed(71)
    t, d, i, e, k = shape
    ids = torch.rand(t, e, device="cuda").topk(k, -1).indices
    layout, rowmap, inverse = _routing(ids, e)
    a = torch.randn(layout.rows, i, device="cuda", dtype=torch.bfloat16) * .1
    w = torch.randn(e, d, i, device="cuda", dtype=torch.bfloat16) / math.sqrt(i)
    scores = torch.rand(t, k, device="cuda", dtype=torch.float32) / k
    for precision in ("bf16", "fp8"):
        gemms = GroupedTrainingGemm(layout, precision)
        budget = (512 * 1024 * 1024 // (2 * max(layout.rows, 1))) // 128 * 128
        columns = max(128, min(4096 if precision == "bf16" else 2048, budget))
        def materialize():
            return _project(gemms,a,w,rowmap,inverse,ids,scores,None,"materialize",columns)
        def scatter():
            return gemms.m_gemm_scatter(a,w,rowmap,t,k,scores)
        def chunked():
            return _project(gemms,a,w,rowmap,inverse,ids,scores,None,"chunked",columns)
        old, new = materialize(), scatter()
        check = metrics(new,old)
        assert check["rrmse"] < .01, check
        tiled = chunked()
        column_check = metrics(tiled, old)
        assert column_check["rrmse"] < 1e-4, column_check
        del old,new,tiled
        for name, fn in (("materialize",materialize),("scatter",scatter),("chunked",chunked)):
            for _ in range(2): fn()
            torch.cuda.synchronize()
            initial = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            samples, wall_samples = [], []
            for _ in range(3):
                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                wall_start = time.perf_counter()
                start.record()
                for _ in range(iterations): fn()
                end.record(); end.synchronize()
                samples.append(start.elapsed_time(end)/iterations)
                wall_samples.append((time.perf_counter()-wall_start)*1000/iterations)
            value=dict(mode=name,precision=precision,t=t,d=d,i=i,e=e,k=k,rows=layout.rows,
                       columns=columns, distinct_routes=True, timing="cuda_event",
                       samples_ms=samples,median_ms=statistics.median(samples), wall_samples_ms=wall_samples,
                       wall_median_ms=statistics.median(wall_samples),
                       comparison=column_check if name=="chunked" else check if name=="scatter" else None,
                       comparison_reference="same_precision_materialized",
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       incremental_peak_bytes=torch.cuda.max_memory_allocated()-initial,
                       theoretical_y_bytes=layout.rows*d*2,theoretical_acc_bytes=t*d*4)
            print("PROJECTION",json.dumps(value),flush=True)
            results.append(value)
    return results


def run(output, iterations, debug_shapes=False, extended_shapes=False):
    output.mkdir(parents=True, exist_ok=True)
    report = {"environment": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
              "protobuf": google.protobuf.__version__, "protobuf_backend": Type(),
              "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
              "reference_matmul_precision": torch.get_float32_matmul_precision(),
              "reference_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
              "packages": {}}, "numerical": [], "training": [], "benchmarks": [],
              "shape_options": dict(debug_shapes=debug_shapes, extended_shapes=extended_shapes),
              "numerical_contract": {"reference": "float32", "bf16_rrmse_limit": .025,
                                     "sonic_bf16_gradient_rrmse_limit": .03,
                                     "fp8_rrmse_limit": .15,
                                     "fp8": "approximate quantized training with straight-through gradients"}}
    for package in ("deep-gemm", "quack-kernels", "nvidia-cutlass-dsl",
                    "nvidia-cutlass-dsl-libs-core", "nvidia-cutlass-dsl-libs-base",
                    "nvidia-cutlass-dsl-libs-cu13", "pytorch-triton"):
        try:
            report["environment"]["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    def save():
        (output / "results.json").write_text(json.dumps(report, indent=2))
    save()
    report["routing"] = validate_routing()
    save()
    for precision in ("bf16", "fp8"):
        for options in ({}, {"bias": True, "skew": True, "zero_scores": True}, {"t": 0}, {"t": 0, "bias": True},
                        {"k": 8, "bias": True, "skew": True, "signed": True},
                        {"scatter": False, "bias": True, "zero_scores": True},
                        {"scatter": True, "k": 8, "bias": True, "skew": True, "signed": True},
                        {"chunk_size": 256, "d": 384, "bias": True, "skew": True, "zero_scores": True}):
            print("NUMERICAL", precision, options, flush=True)
            value = validate_case(precision, **options)
            report["numerical"].append(value)
            save()
            print(json.dumps(value), flush=True)
    # Small T/D keeps the independent FP32 graph bounded. Match the sparse
    # expert counts and top-k of performance shapes, and check every gradient.
    sparse_pairs = {(shape[3], shape[4]) for shape in benchmark_shapes(debug_shapes, extended_shapes)}
    for e, k in sorted(sparse_pairs):
        for precision in ("bf16", "fp8"):
            for reduction in ("chunked", "materialize", "atomic"):
                options = dict(t=129, d=256, i=512, e=e, k=k, random_routes=True,
                               bias=True, zero_scores=True, signed=True, reduction=reduction)
                print("SPARSE_NUMERICAL", precision, options, flush=True)
                report["numerical"].append(validate_case(precision, **options))
                save()
        report["numerical"].append(validate_case("bf16", t=129, e=e, k=k,
                                                  random_routes=True, backend="sonic"))
        save()
    training = validate_sparse_training()
    report["training"] = training.pop("runs")
    report["training_drift"] = training.pop("drift")
    report["training_acceptance"] = training
    save()
    from sonicmoe.functional import moe_general_routing_inputs
    from sonicmoe.enums import ActivationType
    report["wrapper"] = {mode:validate_wrapper(mode) for mode in ("bf16", "fp8")}
    report["wrapper"]["fp8_recompute"] = validate_wrapper("fp8", recompute_dh=True)
    save()
    for t,d,i,e,k in benchmark_shapes(debug_shapes, extended_shapes):
        torch.manual_seed(11)
        x = torch.randn(t,d,device="cuda",dtype=torch.bfloat16,requires_grad=True)
        w1 = (torch.randn(e,2*i,d,device="cuda",dtype=torch.bfloat16)/math.sqrt(d)).requires_grad_()
        w2 = (torch.randn(e,d,i,device="cuda",dtype=torch.bfloat16)/math.sqrt(i)).requires_grad_()
        scores = torch.rand(t,k,device="cuda",dtype=torch.float32,requires_grad=True)
        ids = torch.rand(t,e,device="cuda").topk(k,-1).indices.int()
        token_ids = torch.arange(t,device="cuda",dtype=torch.int32).repeat_interleave(k)
        do = torch.randn_like(x)
        params=[x,w1,w2,scores]
        shape=dict(t=t,d=d,i=i,e=e,k=k)
        modes={
            "sonic_bf16":lambda:moe_general_routing_inputs(
                x=x, router_scores=scores.flatten(), token_indices=token_ids, expert_indices=ids.flatten(),
                w1=w1.permute(1,2,0), b1=None, w2=w2.permute(1,2,0), b2=None, E=e,
                stream_id=torch.cuda.current_stream().cuda_stream, activation_type=ActivationType.SWIGLU)[0],
            "materialize_bf16":lambda:deepgemm_swiglu(x,w1,w2,scores,ids,precision="bf16",scatter=False),
            "materialize_fp8":lambda:deepgemm_swiglu(x,w1,w2,scores,ids,precision="fp8",scatter=False),
            "scatter_bf16":lambda:deepgemm_swiglu(x,w1,w2,scores,ids,precision="bf16",scatter=True),
            "scatter_fp8":lambda:deepgemm_swiglu(x,w1,w2,scores,ids,precision="fp8",scatter=True),
            "chunked_bf16":lambda:deepgemm_swiglu(x,w1,w2,scores,ids,precision="bf16"),
            "chunked_fp8":lambda:deepgemm_swiglu(x,w1,w2,scores,ids,precision="fp8"),
            "recompute_fp8":lambda:deepgemm_swiglu(x,w1,w2,scores,ids,precision="fp8",recompute_dh=True),
        }
        # FP32 output and gradient comparisons ran above at bounded T/D;
        # performance shapes never construct the much larger reference graph.
        for mode, fn in modes.items():
            print("BENCHMARK",shape,mode,flush=True)
            value = {"shape":shape,"mode":mode,"distinct_routes":True,
                     **time_training(fn,params,do,iterations)}
            report["benchmarks"].append(value)
            save()
            print(json.dumps(value),flush=True)
        del modes, fn, params, x, w1, w2, scores, ids, token_ids, do
    report["projection"] = projection_benchmark(iterations, debug_shapes, extended_shapes)
    report["passed"] = True
    save()
    print("ACCEPTANCE PASSED",flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--iterations",type=int,default=5)
    p.add_argument("--debug-shapes",action="store_true",help="append legacy K2 and E8/K8 performance controls")
    p.add_argument("--extended-shapes",action="store_true",help="append T32768,D4096,I1536,E64,K8")
    args=p.parse_args()
    if args.iterations < 1:
        p.error("--iterations must be positive")
    run(args.output,args.iterations,args.debug_shapes,args.extended_shapes)
