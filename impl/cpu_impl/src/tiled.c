#include <omp.h>

#define ROW_TILE  32
#define COL_TILE  32
#define K_TILE     8

static inline int min_size(int a, int b) {
    return (a < b) ? a : b;
}

void tiled_matmul(
    int M,
    int K,
    int N,
    float* A,   // [M][K]
    float* B,   // [K][N]
    float* C    // [M][N], assumed zero-initialized
) {
    for (int rowTile = 0; rowTile < M; rowTile += ROW_TILE) {
        for (int colTile = 0; colTile < N; colTile += COL_TILE) {
            for (int kTile = 0; kTile < K; kTile += K_TILE) {
                int rowEnd = min_size(rowTile + ROW_TILE, M);
                int colEnd = min_size(colTile + COL_TILE, N);
                int kEnd = min_size(kTile + K_TILE, K);
                // inner three loops: row → k → col
                for (int i = rowTile; i < rowEnd; ++i) {
                    for (int k = kTile; k < kEnd; ++k) {
                        float a_val = A[i*K + k];
                        for (int j = colTile; j < colEnd; ++j) {
                            C[i*N + j] += a_val * B[k*N + j];
                        }
                    }
                }

            }
        }
    }
}