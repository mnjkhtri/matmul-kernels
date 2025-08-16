#include <torch/extension.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/execution_policy.h>


torch::Tensor thrust_matmul(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);

    // Allocate output
    auto C = torch::zeros({M, N}, A.options());

    // Total number of elements in C
    int64_t total = M * N;

    // Wrap raw pointers with thrust device_ptr
    float* a_ptr = A.data_ptr<float>();
    float* b_ptr = B.data_ptr<float>();
    float* c_ptr = C.data_ptr<float>();

    // Wrap raw pointer in a Thrust device_ptr
    thrust::device_ptr<float> c_dev_ptr = thrust::device_pointer_cast(c_ptr);

    // Create counting iterators [0, total)
    auto begin = thrust::make_counting_iterator<int64_t>(0);
    auto end   = thrust::make_counting_iterator<int64_t>(total);

    // Compute each C[row, col] in parallel
    thrust::transform(
        thrust::cuda::par,      // correct CUDA policy
        begin, end,             // input range: 0 .. total
        c_dev_ptr,              // output iterator
        [=] __device__(int64_t idx) -> float {
            const int row = idx / N;
            const int col = idx % N;
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += a_ptr[row * K + k] * b_ptr[k * N + col];
            }
            return sum;
        }
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("thrust_matmul", &thrust_matmul, "CUDA matmul via Thrust");
}
