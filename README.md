
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

## Example

The following example demonstrates a vanilla matrix multiplication implementation for pytorch tensor but written in pure cuda.
As you may happily notice, pytorch is responsible for allocation of new Tensors instead of in the cuda code, and the elements of tensors can be read and modified inside the kernel function. 

```python
import torch
import cutex

M, N, K = 4, 4, 1
a = torch.rand((M, K), dtype=torch.float32).cuda()
b = torch.rand((K, N), dtype=torch.float32).cuda()
c = torch.empty((M, N), dtype=torch.float32).cuda()

kernels = cutex.SourceModule(r"""
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
pip install cutex
```

**Note:**

- You should install pytorch and nvcc manually, which are not automatically managed dependencies.
- The `//cuda` and `//!cuda` comments are not mandatory, it works together with the VSCode [extension](https://marketplace.visualstudio.com/items?itemName=huangyuyao.pycuda-highlighter) for highlighting CUDA source in python docstring.
