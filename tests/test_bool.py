import torch
from cutex import SourceModule

booldata = torch.randint(0, 2, (3, 3), dtype=torch.bool).cuda()

kernel = SourceModule(
"""
__global__ void print(Tensor<bool, 2> booldata) {
    int m = threadIdx.x;
    int n = threadIdx.y;
    if (m >= booldata.size(0) || n >= booldata.size(1)) return;

    booldata[m][n] = !booldata[m][n];

    printf("booldata[%d][%d]=%s\\n", m, n, (booldata[m][n] ? "True" : "False"));
}
""", 32)

print(booldata)
kernel.print(booldata, block=(3, 3, 1))


kernel = SourceModule(
"""
__global__ void print(bool x) {
    if (x) {
        printf("True\\n");
    } else {
        printf("False\\n");
    }
}
""", 32)

kernel.print(True)
