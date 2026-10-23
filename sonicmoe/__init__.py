# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

__version__ = "0.1.2.post1"

from .enums import KernelBackendMoE

__all__ = [
    "KernelBackendMoE",
    "MoE",
    "moe_general_routing_inputs",
    "moe_TC_softmax_topk_layer",
    "deepgemm_swiglu",
]


# Keep the DeepGEMM backend usable without importing the QuACK runtime. Existing
# public names are resolved lazily, with identical import syntax for callers.
def __getattr__(name):
    if name in ("moe_general_routing_inputs", "moe_TC_softmax_topk_layer"):
        from . import functional
        return getattr(functional, name)
    if name == "MoE":
        from .moe import MoE
        return MoE
    if name == "deepgemm_swiglu":
        from .deepgemm import deepgemm_swiglu
        return deepgemm_swiglu
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
