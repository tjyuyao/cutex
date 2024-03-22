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

kernels.matmul(a, b, c, M, N, K, grid=(N // 16 + 1, M // 16 + 1), block=(16, 16, 1))

torch.cuda.synchronize()
assert torch.allclose(c, torch.mm(a, b))
