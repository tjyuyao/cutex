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
