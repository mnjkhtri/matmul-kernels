import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def numba_matmul(A: np.ndarray, B: np.ndarray):
    M, K = A.shape
    N = B.shape[1]
    C = np.zeros((M, N), dtype=A.dtype)
    for i in prange(M):
        for j in range(N):
            tmp = 0.0
            for k in range(K):
                tmp += A[i, k] * B[k, j]
            C[i, j] = tmp
    return C