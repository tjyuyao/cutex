import torch
import numpy
import inspect
from torch import Tensor
from functools import lru_cache as cache

from .src_module import _CUDAContext, _Tensor, SourceModule

# https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype
# https://docs.nvidia.com/cuda/cuda-math-api/struct____nv__bfloat16.html#struct____nv__bfloat16
DTYPE_MAPPING = {
    torch.float16: "__half",
    torch.float32: "float",
    torch.float64: "double",
    torch.bfloat16: "__nv_bfloat16",
    torch.uint8: "uint8_t",
    torch.int8: "int8_t",
    torch.int16: "int16_t",
    torch.int32: "int32_t",
    torch.int64: "int64_t",
    torch.bool: "bool",
}

FLOAT_SCALAR = {
    torch.float16: ("__half", numpy.float16),
    torch.float32: ("float", numpy.float32),
    torch.float64: ("double", numpy.float64),
    torch.bfloat16: ("float", numpy.float32),  # only convert to float
    16: ("__half", numpy.float16),
    32: ("float", numpy.float32),
    64: ("double", numpy.float64),
}

INT_SCALAR = {
    16: ("int16_t", numpy.int16),
    32: ("int32_t", numpy.int32),
    64: ("int64_t", numpy.int64),
}


@cache
def _jit_inline_compile(signature, cuda_src, boundscheck=True):
    module = SourceModule(
        f"""
    __global__ void inline_kernel({signature}) {{
        {cuda_src}
    }}
    """,
        float_bits=None,
        int_bits=None,
        boundscheck=boundscheck,
    )
    module._jit_compile()
    func = module.mod.get_function("__wrapper_inline_kernel")
    return func


def inline(cuda_src, float_bits=None, int_bits=32, boundscheck=True) -> None:
    # capture all local vars from the prev frame
    args = inspect.currentframe().f_back.f_locals

    # determine float_bits automatically
    if float_bits is None:
        for v in args.values():
            if isinstance(v, Tensor) and v.dtype in FLOAT_SCALAR:
                float_bits = v.dtype
                break
        else:
            float_bits = 32
    float_ctype, np_float_type = FLOAT_SCALAR[float_bits]
    int_ctype, np_int_type = INT_SCALAR[int_bits]
    bool_ctype = "bool"

    with _CUDAContext():
        # get argument lists
        formal_args = []
        actual_args = []
        for k, v in args.items():
            if isinstance(v, Tensor):
                decl = f"Tensor<{DTYPE_MAPPING[v.dtype]}, {v.dim()}> {k}"
                arg = _Tensor(v.cuda())
            elif isinstance(v, int):
                decl = f"const {int_ctype} {k}"
                arg = np_int_type(v)
            elif isinstance(v, float):
                decl = f"const {float_ctype} {k}"
                arg = np_float_type(v)
            elif isinstance(v, bool):
                decl = f"const {bool_ctype} {k}"
                arg = v
            else:
                continue
            formal_args.append(decl)
            actual_args.append(arg)
        signature = ", ".join(formal_args)
        # get (cached) kernel function
        func = _jit_inline_compile(signature, cuda_src, boundscheck=boundscheck)
        # call the kernel
        func(*actual_args, block=args["blockDim"], grid=args["gridDim"])
