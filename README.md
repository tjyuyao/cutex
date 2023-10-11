
<p align="center"><img src="https://github.com/tjyuyao/cutex/raw/main/logo.png" alt="Logo"></p>

<h3 align="center" style="font-weight:bold"> PyCUDA based PyTorch Extension Made Easy </h3>

---

In a word, `cutex` bridges PyCUDA's just-in-time compilation with PyTorch's Tensor type.

``cutex.SourceModule`` works differently compared to [PyTorch's official cuda extension guide](https://pytorch.org/tutorials/advanced/cpp_extension.html) in following ways:

- **It compiles lightning fast!** Especially suitable for rapidly developing your favoritenew algorithm.
- Without boilerplate cpp wrappers, **every user code goes within one python file**.
- It use raw CUDA syntax so that PyTorch's c++ API is _not_ available.

``cutex.SourceModule`` extends pycuda's ``SourceModule`` in following ways:

- Support efficient **multi-dimensional `torch.Tensor` access with (efficient & optional) out-of-boundary check**.
- Enhanced automatic type conversion and error messages.

## Example (inline CUDA API)

```py
import cutex
import torch


def matmul(A, B):
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
    """)  # all local vars are captured into the kernel except for those with unknown types.
    return C


def test():
    M, N, K = 4, 4, 1
    A = torch.rand((M, K), dtype=torch.float32).cuda()
    B = torch.rand((K, N), dtype=torch.float32).cuda()
    torch.testing.assert_close(matmul(A, B), torch.mm(A, B))
    print(matmul(A, B)) 


test()
```

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
    boundscheck=True, # turning off checking makes the program to run faster, default is on.
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

v0.3.5+HEAD - bugfix: bool scalar type automatically converted into int32_t.
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
