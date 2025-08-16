#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_WIDTH 16

#define A2(i,j) A[(i)*K + (j)]   // A is M×K
#define B2(i,j) B[(i)*N + (j)]   // B is K×N
#define C2(i,j) C[(i)*N + (j)]   // C is M×N

__global__
void naive_matmul_k(const int M, const int K, const int N, const float *A, const float *B, float *C)
{
    int row = blockIdx.y * BLOCK_WIDTH + threadIdx.y;
    int col = blockIdx.x * BLOCK_WIDTH + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        // Dot‐product of A’s row “row” and B’s column “col”
        for (int k = 0; k < K; ++k) {
            sum += A2(row, k) * B2(k, col);
        }
        C2(row, col) = sum;
    }
}

#undef A2
#undef B2
#undef C2

torch::Tensor naive_matmul(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(A.is_contiguous(), "B must be contiguous");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 block(BLOCK_WIDTH, BLOCK_WIDTH);

    dim3 grid((N + BLOCK_WIDTH - 1) / BLOCK_WIDTH, (M + BLOCK_WIDTH - 1) / BLOCK_WIDTH);

    naive_matmul_k<<<grid, block>>>(
        (int)M, 
        (int)K, 
        (int)N,
        A.data_ptr<float>(), // equiv to ctypes
        B.data_ptr<float>(),
        C.data_ptr<float>()
    );

    cudaDeviceSynchronize(); // wait till the completion is done

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_matmul", &naive_matmul, "CUDA naive");
}