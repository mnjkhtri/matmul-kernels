import numpy as np
import cffi
import os

ffi = cffi.FFI()

ffi.cdef("""
void order_matmul(size_t M, size_t K, size_t N, float *A, float *B, float *C);
""")

def order_matmul(A: np.ndarray, B: np.ndarray, o3: bool = False):
    
    _LIB = ffi.dlopen(os.path.join(os.path.dirname(__file__), "build/order.so"))

    M, K = A.shape
    _, N = B.shape

    C_res = np.zeros((M, N), dtype=np.float32)
    _LIB.order_matmul(M, K, N,
        ffi.cast("float *", A.ctypes.data), 
        ffi.cast("float *", B.ctypes.data), 
        ffi.cast("float *", C_res.ctypes.data)
    )

    return C_res