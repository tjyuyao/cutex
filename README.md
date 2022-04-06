
<p align="center"><img src="https://github.com/tjyuyao/cutex/raw/main/logo.png" alt="Logo"></p>

<h3 align="center" style="font-weight:bold"> PyCUDA based PyTorch Extension Made Easy </h3>

---

In a word, `cutex` bridges PyCUDA's just-in-time compilation with PyTorch's Tensor type.

``cutex.SourceModule`` works differently compared to [PyTorch's official cuda extension guide](https://pytorch.org/tutorials/advanced/cpp_extension.html) in following ways:

- **It compiles lightning fast!** Especially suitable for rapidly developing new algorithms with a jupyter kernel, so that you don't wait for importing pytorch repeatedly.
- Without boilerplate cpp wrappers, **every user code goes within one python file**.
- It use raw CUDA syntax so that PyTorch's c++ API is _not_ available.

``cutex.SourceModule`` works differently compared to pycuda's ``SourceModule`` in following ways:

- Support efficient **multi-dimensional `torch.Tensor` access with (efficient & optional) out-of-boundary check**.
- Enhanced automatic type conversion and error messages.

## Example

The following example demonstrates a vanilla matrix multiplication implementation for pytorch tensor but written in pure cuda.

```python
import torch
import cutex

M, N, K = 4, 4, 1
a = torch.rand((M, K), dtype=torch.float32).cuda()
b = torch.rand((K, N), dtype=torch.float32).cuda()
c = torch.empty((M, N), dtype=torch.float32).cuda()

kernels = cutex.SourceModule(r"""
__global__ void matmul(Tensor<float, 2> *a, Tensor<float, 2> *b, Tensor<float, 2> *c, int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.f;
    if (m >= M || n >= N) return;
    for (int k = 0; k < K; ++k) {
        v += (*a)[m][k] * (*b)[k][n];
    }
    (*c)[m][n] = v;
}
""", float_bits=32)

kernels.matmul(a, b, c, M, N, K, grid=(N // 32 + 1, M // 32 + 1), block=(32, 32, 1))

torch.cuda.synchronize()
assert torch.allclose(c, torch.mm(a, b))
```

## Installation

```bash
pip install cutex
```

**Note:**

- You should install pytorch seperately.
- If you use vscode, there is a recommended [extension](https://marketplace.visualstudio.com/items?itemName=huangyuyao.pycuda-highlighter) for highlighting CUDA source in python docstring.
