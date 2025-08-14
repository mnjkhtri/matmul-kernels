import numpy as np
import cffi
import os

ffi = cffi.FFI()

ffi.cdef("""
void tiled_matmul(size_t M, size_t K, size_t N, float *A, float *B, float *C);
""")

def tiled_matmul(A: np.ndarray, B: np.ndarray):

    _LIB_PATH = "build/tiled.so"
    _LIB = ffi.dlopen(os.path.join(os.path.dirname(__file__), _LIB_PATH))

    M, K = A.shape
    _, N = B.shape

    C_res = np.zeros((M, N), dtype=np.float32)
    _LIB.tiled_matmul(M, K, N,
        ffi.cast("float *", A.ctypes.data), 
        ffi.cast("float *", B.ctypes.data), 
        ffi.cast("float *", C_res.ctypes.data)
    )
    return C_res