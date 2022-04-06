"""Integrates PyCUDA to PyTorch and ice."""
from __future__ import annotations

import os
import re
from functools import wraps

import numpy

# According to [this link](https://numpy.org/devdocs/release/1.20.0-notes.html#using-the-aliases-of-builtin-types-like-np-int-is-deprecated),
# `np.bool` is a deprecated alias for the builtin `bool`. This deprecation will cause `/site-packages/pycuda/compyte/dtypes.py:122`  with code
# `reg.get_or_register_dtype("bool", np.bool)` throwing a DeprecationWarning. We choose to use the old behavior.
numpy.bool = bool

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

import pycuda.driver as cuda

_int_regex = re.compile(r'\bint\b')
_float_regex = re.compile(r'\bfloat\b')
_extern_c_regex = re.compile(r'extern[ ]+"C"')
_kernel_cu_regex = re.compile(r'kernel.cu\((?P<lineno>[0-9]+)\).*')

# https://nw.tsuda.ac.jp/lec/cuda/doc_v9_0/pdf/CUDA_Math_API.pdf

_MEMCHECK_ENABLER = '#define _ICE_MEMCHECK_ 1 \n'
_EXTRA_HEADERS ='#include "PyCUDATensorAccessor.cuh" \n'


class _CUDAContext:
    
    def __init__(self):
        import torch

        import pycuda.driver as cuda
        self.device = cuda.Device(torch.cuda.current_device())
    
    def __enter__(self):
        self.context = self.device.retain_primary_context()
        self.context.push()
        return self.context
    
    def __exit__(self, type, value, traceback):
        self.context.pop()
        
class _Tensor(cuda.PointerHolderBase):
    def __init__(self, t: torch.Tensor):
        import torch
        if not t.is_cuda:
            raise ValueError('Cannot convert CPU tensor for pycuda (call `cuda()` on it)')
        if not t.layout is torch.strided:
            raise ValueError('Cannot convert sparse tensor for pycuda')
        super().__init__()
        self.t = t
        self.dims = t.dim()
        # memory allocation on cuda using pytorch interface
        self.struct = torch.empty((3,), dtype=torch.int64, device="cuda") # maps to Tensor struct defined in "PyCUDATensorAccessor.cuh"
        self.base = int(t.data_ptr())
        self.size = torch.tensor(tuple(t.size()), device="cuda", dtype=torch.int64)
        self.stride = torch.tensor(tuple(t.stride()), device="cuda", dtype=torch.int64)
        self.gpudata = self.struct.data_ptr()
        # initialize struct Tensor's fields.
        cuda.memcpy_htod(int(self.gpudata), memoryview(numpy.uintp(self.base)))
        cuda.memcpy_htod(int(self.gpudata) + 8, memoryview(numpy.uintp(int(self.size.data_ptr()))))
        cuda.memcpy_htod(int(self.gpudata) + 16, memoryview(numpy.uintp(int(self.stride.data_ptr()))))


class SourceModule(object):
    
    def __init__(self, source, float_bits, int_bits=32, include_dirs=[], boundscheck=True, **kwds):
        """Setup the parameters for compiling a CUDA source.

        Args:
            source (str): CUDA C++ source string.
            float_bits (int): bit width of float values used as tensor scalar.
            int_bits (int, optional): bit width of default int scalar. Defaults to 32.
            include_dirs (list, optional): paths of extra include dirs. Defaults to [].
            boundscheck (bool, optional): enable out of bound check for tensors. Defaults to True.
            **kwds: other keyword args you would like to pass to pycuda's ``SourceModule``.

        Note:
            Direct written `float` and `int` token in the source string will be substituted
            to ensure the default scalar data type matches the tensors. If you do not want 
            this to happen, use more specific CUDA typename such as `__half`, `double`, `int16_t`, etc.

        """
        import torch
        source = self._replace_data_type(source, int_bits, float_bits)
        if not self._find_extern_C(source):
            source = 'extern "C" {\n' + source +'\n}'
        if float_bits == 16:
            source = "# include <cuda_fp16.h>\n" + _EXTRA_HEADERS + source
        else:
            source = _EXTRA_HEADERS + source
        if boundscheck:
            source = _MEMCHECK_ENABLER + source
        include_dirs.append(os.path.join(os.path.dirname(__file__), "include"))
        kwds.update(dict(
            source=source,
            include_dirs=include_dirs,
            no_extern_c=1,
        ))
        self.mod = None
        self.compile_kwds = kwds
        self.numpy_int_type = {16: numpy.int16, 32: numpy.int32, 64: numpy.int64}[int_bits]
        self.numpy_float_type = {16: numpy.float32, 32: numpy.float32, 64: numpy.float64}[float_bits]
        self.torch_int_types = [torch.int16, torch.int32, torch.int64]
        self.torch_all_float_types = {16: torch.float16, 32: torch.float32, 64: torch.float64}
        self.torch_float_type = self.torch_all_float_types[float_bits]
        self.float_bits = float_bits
    
    def __getattr__(self, name):
        import torch
        import pycuda.driver as cuda
        @wraps(cuda.Function.__call__)
        def wrapper(*args, block=(1, 1, 1), grid=(1, 1), **kwds):
            casted_args = []
            with _CUDAContext():
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        if arg.dtype in self.torch_all_float_types and \
                            arg.dtype != self.torch_float_type:
                            arg = arg.to(dtype=self.torch_float_type)
                            # TODO: warning? autocast mode sensitive?
                            # raise TypeError(f"This CUDAModule expects {self.torch_float_type} but you passed in a {arg.dtype} Tensor.")
                        arg = _Tensor(arg)
                    elif isinstance(arg, int):
                        arg = self.numpy_int_type(arg)
                    elif isinstance(arg, float):
                        arg = self.numpy_float_type(arg)
                    casted_args.append(arg)
                if self.mod is None:
                    self._jit_compile()  # compile lazily
                func = self.mod.get_function(name)
                func(*casted_args, block=block, grid=grid, **kwds)
        return wrapper

    def _jit_compile(self):
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        try:
            self.mod = SourceModule(**self.compile_kwds)
        except cuda.CompileError as e:
            # add source code that triggers the error to print
            source_lines = self.compile_kwds["source"].split("\n")
            def repl(x):
                lineno = int(x.group("lineno")) - 1
                return f"{x.group(0)}\n{source_lines[lineno]}"
            e.stderr = _kernel_cu_regex.sub(repl, e.stderr)
            raise e
    
    @staticmethod
    def _find_extern_C(source):
        for _ in _extern_c_regex.finditer(source):
            return True
        return False

    @staticmethod
    def _replace_data_type(string, int_bits, float_bits):
        int_t = f"int{int_bits}_t"
        float_t = {16:"__half", 32:"float", 64:"double"}[float_bits]
        string = _int_regex.sub(int_t, string)
        string = _float_regex.sub(float_t, string)
        return string
