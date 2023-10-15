
<p align="center"><img src="https://github.com/tjyuyao/cutex/raw/main/logo.png" alt="Logo"></p>

<h3 align="center" style="font-weight:bold"> PyCUDA based PyTorch Extension Made Easy </h3>

---

In a word, `cutex` bridges PyCUDA's just-in-time compilation with PyTorch's Tensor type.

``cutex.SourceModule`` extends pycuda's ``SourceModule`` in following ways:

- Designed to work seemlessly with pytorch `Tensor` type, Data-Distributed Parallel (DDP), and `autograd.Function` API.
- Support efficient **multi-dimensional `torch.Tensor` access with (efficient & optional) out-of-boundary check**.
- Enhanced automatic type conversion and error messages.

``cutex.SourceModule`` works differently compared to [PyTorch's official cuda extension guide](https://pytorch.org/tutorials/advanced/cpp_extension.html) in following ways:

- **It compiles lightning fast!** Especially suitable for rapidly developing your favorite new algorithm.
- Without boilerplate cpp wrappers, **every user code goes within one python file**.
- It use raw CUDA syntax so that PyTorch's c++ API is _not_ available, it is recommended to use either raw CUDA with cutex or python API with pytorch.

## Example (inline CUDA API)

This is a new high level API for writing custom kernels since v0.3.0. You omit the signature of the kernel function and `cutex.inline()` will compile and run it according to the context on the cuda device. This facillitates a fluent switching between pytorch and pycuda.

```py
import cutex
import torch
from torch import Tensor


def matmul(A:Tensor, B:Tensor):
    M, J = A.size()
    K, N = B.size()
    assert J == K
    gridDim = (cutex.ceildiv(N, 16), cutex.ceildiv(M, 16), 1)
    blockDim = (16, 16, 1)
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    cutex.inline("""
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.f;
    if (m >= M || n >= N) return;
    for (int k = 0; k < K; ++k) {
        v += A[m][k] * B[k][n];
    }
    C[m][n] = v;
    """, boundscheck=False)  # all local vars are captured into the kernel except for those with unknown types.
    return C


def test():
    M, N, K = 4, 4, 1
    A = torch.rand((M, K), dtype=torch.float32).cuda()
    B = torch.rand((K, N), dtype=torch.float32).cuda()
    torch.testing.assert_close(matmul(A, B), torch.mm(A, B))
    print(matmul(A, B)) 


test()
```

Local variables of Tensor and common scalar types (`int`, `float`, etc.) and special ones `gridDim` and `blockDim` are captured into the inline execution, as if they were in the same scope. The order of defining them does not matter, only have to be assigned before the inline execution. 
Multiple inline execution in the same python function is legal. When doing so, make `gridDim` and `blockDim` update their value before the next execution.

The tensors can be acccessed element-wise using multi-dimensional squared brackets `[]` as illustrated in the above example. It can be read and write, and the modifications would reflected directly to the pytorch tensor on cuda devices. By default, with the `boundscheck` option on, these brackets will check for out of bound error. While this is very useful for debugging novel algorithms, it will make use of more registers in the SM, so if you want to make full use of the SM register resources, e.g. using maximum block threads, you need to turn boundscheck off for best performance.

Unless explicitly specified, the `float` type in the CUDA part will be automatically replaced to the same type as the first local Tensor variable with float dtype, in the above example, it would be aligned with `A`.

## Example (lower level SourceModule API)

The following example demonstrates a vanilla matrix multiplication implementation for pytorch tensor but written in pure cuda.
As you may happily notice, pytorch is responsible for allocation of new Tensors instead of in the cuda code, and the elements of tensors can be read and modified inside the kernel function. 

```python
import torch
import cutex

M, N, K = 4, 4, 1
a = torch.rand((M, K), dtype=torch.float32).cuda()
b = torch.rand((K, N), dtype=torch.float32).cuda()
c = torch.empty((M, N), dtype=torch.float32).cuda()

kernels = cutex.SourceModule("""
//cuda
__global__ void matmul(Tensor<float, 2> a, Tensor<float, 2> b, Tensor<float, 2> c, int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.f;
    if (m >= M || n >= N) return; // you can also write `a.size(0)` instead of `M`, `b.size(1)` instead of `N`
    for (int k = 0; k < K; ++k) { // you can also write `a.size(1)` instead of `K`
        v += a[m][k] * b[k][n]; // you can access tensor elements just like operating a multi-level array, with optional out-of-bound check.
    }
    c[m][n] = v; // the modification will be reflected in the torch tensor in place, no redundant data copying.
}
//!cuda
""",
    float_bits=32,  # change to 16 to use half precision as `float` type in the above source code.
    boundscheck=True, # turning on for debug and off for performance (to use full threads of a block), default is on.
    )

kernels.matmul(  # automatically discover the kernel function by its name (e.g. 'matmul'), just like a normal python module.
    a, b, c, M, N, K,  # directly pass tensors and scalars as arguments
    grid=(N // 16 + 1, M // 16 + 1),  # grid size (number of blocks to be executed)
    block=(16, 16, 1),  # block size (number of threads in each block)
)

assert torch.allclose(c, torch.mm(a, b))
```

## Installation

```bash
pip install -U cutex --index-url "https://pypi.org/simple/"
```

**Note:**

- You should install pytorch and nvcc manually, which are not automatically managed dependencies.
- The `//cuda` and `//!cuda` comments are not mandatory, it works together with the VSCode [extension](https://marketplace.visualstudio.com/items?itemName=huangyuyao.pycuda-highlighter) for highlighting CUDA source in python docstring.

## Change Log

```
# format: {pypi-version}+{git-commit-hash} - ["[CUDA]"] {description}
# "[CUDA]" means changes related to the cuda side Tensor API.

v0.3.8+HEAD - add boundscheck option to inline execution
v0.3.7+e48537 - bugfix: passing python float to a kernel that accept a __half type now works.
v0.3.6+4e9b41 - bugfix: v0.3.5 uses regex to replace bool, this may be confused with Tensor with bool dtype, this version revert v0.3.5 and use the wrapper to convert scalar type.
v0.3.5+8bdfbc - bugfix: bool scalar type automatically converted into int32_t.
v0.3.4+07b6af - bugfix: error report in jupyter cell.
v0.3.3+0dc015 - bugfix: error report should find in the whole file.
v0.3.2+bc47ee - enhanced the error report, accurate lineno in the python file; ensure gridDim and blockDim to be integers.
v0.3.1+b46561 - automatically send tensor to cuda in inline execution; scalars are const;
v0.3.0+b93dc6 - !NEW FEATURE! inline execution of CUDA code
v0.2.2+025fb1 - multiple enhancements.
    - [CUDA] fatal bug fixed checking OOB in `Tensor<Any,1>.size(dim:int)->int` function;
    - !NEW FEATURE! add `ceildiv(int, int)->int` API as a util function.
v0.2.1+dc4373 - [CUDA] add `Tensor.size(dim:int)->int` API.
v0.2.0+03c3c5 - [CUDA] !NEW FEATURE! declare Tensor type argument instead of its pointer.
v0.1.1+d088de - core features
    - basic automatic cuda context management;
    - basic automatic tensor type argument via `pycuda.driver.PointerHolderBase`;
    - basic out-of-boundary check;
    - easy to use `SourceModule` API.
```
