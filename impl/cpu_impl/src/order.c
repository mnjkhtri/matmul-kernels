void order_matmul(
    int M,
    int K,
    int N,
    float* A,
    float* B,
    float* C
) {
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}