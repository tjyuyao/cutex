SOURCE = r"""
__global__      void     matmul(Tensor<float, 2> a, Tensor<float, 2> b, Tensor<float, 2> c, int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.f;
    if (m >= M || n >= N) return;
    for (int k = 0; k < K; ++k) {
        v += a[m][k] * b[k][n];
    }
    c[m][n] = v;
}

__global__ 
void
matmul2(Tensor<float, 2> a, Tensor<float, 2> *b, Tensor<float, 2> c, int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.f;
    if (m >= M || n >= N) return;
    for (int k = 0; k < K; ++k) {
        v += a[m][k] * b[k][n];
    }
    c[m][n] = v;
}
"""

import CppHeaderParser

parser = CppHeaderParser.CppHeader(SOURCE, argType="string")

# from pprint import pprint
# pprint(parser.functions)

for f in parser.functions:
    # print(f['rtnType'], f['name'])
    if f['rtnType'] != "__global__ void": continue # ignore non-kernel function
    new_parameters = []
    call_parameters = []
    for p in f['parameters']:
        star = "*" if p['type'].startswith('Tensor<') and not p['type'].endswith('*') else ""
        new_parameters.append(' '.join([p['type'], star+p['name']]))
        call_parameters.append(star+p['name'])
    wrapper_func = f"""
__global__ void __wrapper_{f['name']}({', '.join(new_parameters)}) {{
    {f['name']}({', '.join(call_parameters)});
}}
"""
    print(wrapper_func)
