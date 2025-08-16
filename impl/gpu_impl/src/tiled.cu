#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_WIDTH 16

#define A2(i,j) A[(i)*K + (j)]   // A is M×K
#define B2(i,j) B[(i)*N + (j)]   // B is K×N
#define C2(i,j) C[(i)*N + (j)]   // C is M×N

#define sA2(i,j) sA[(i)][(j)]    // Shared A TILE
#define sB2(i,j) sB[(i)][(j)]    // Shared B TILE

__global__ 
void tiled_matmul_k(int M, int K, int N, const float *__restrict__ A, const float *__restrict__ B, float *C) {

    // Think same of TILE and BLOCKDIM onwards:

    // Shared-memory tiles for A and B
    __shared__ float sA[BLOCK_WIDTH][BLOCK_WIDTH];
    __shared__ float sB[BLOCK_WIDTH][BLOCK_WIDTH];
    
    int tid_y = threadIdx.x / BLOCK_WIDTH;
    int tid_x = threadIdx.x % BLOCK_WIDTH;

    int row = blockIdx.y * BLOCK_WIDTH + tid_y;
    int col = blockIdx.x * BLOCK_WIDTH + tid_x;


    float sum = 0.0f; // private to threads

    for (int t = 0; t < K; t += BLOCK_WIDTH) {

        int Brow = t + tid_y;
        int Acol = t + tid_x;

        sA[tid_y][tid_x] = (row < M && Acol < K) ? A2(row, Acol) : 0.0f;
        sB[tid_y][tid_x] = (Brow < K && col < N) ? B2(Brow, col) : 0.0f;

        __syncthreads();  // make sure the tile is loaded

        // Multiply the two tiles
        #pragma unroll
        for (int i = 0; i < BLOCK_WIDTH; ++i) {
            sum += sA2(tid_y , i) * sB2(i, tid_x);
        }
        __syncthreads();
    }

    // Write the result
    if (row < M && col < N) {
        C2(row, col) = sum;
    }
}

#undef A2
#undef B2
#undef C2
#undef sA2
#undef sB2

torch::Tensor tiled_matmul(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 block(BLOCK_WIDTH * BLOCK_WIDTH); // Matched to TILE SIZE
    dim3 grid((N + BLOCK_WIDTH - 1) / BLOCK_WIDTH, (M + BLOCK_WIDTH - 1) / BLOCK_WIDTH);

    tiled_matmul_k<<<grid, block>>>(
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
    m.def("tiled_matmul", &tiled_matmul, "CUDA tiled");
}