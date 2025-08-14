import time
import pandas as pd
import numpy as np
from impl.cpu_impl import np_matmul, numba_matmul, naive_matmul, order_matmul, tiled_matmul
from config import MATRIX

def check_correctness(output, ref, name, atol=1e-6, rtol=1e-5):
    if np.allclose(output, ref, atol=atol, rtol=rtol):
        return True
    diff = np.abs(output - ref)
    max_err = diff.max()
    raise ValueError(f"[{name}] result incorrect (max abs error = {max_err:.3e})")

def benchmark(fn, *args, warmup=1, iters=5, **kwargs):
    for _ in range(warmup):
        fn(*args, **kwargs)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)

for M, K, N in MATRIX:
    print(f"\nBenchmarking for matrices of size ({M}, {K}) * ({K}, {N}):")
    
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    ref = np.matmul(A, B)

    backends = [
        {"name": "NumPy", "fn": np_matmul, "args": (A, B)},
        {"name": "Numba", "fn": numba_matmul, "args": (A, B)},
        {"name": "3 Loop Impl, in C", "fn": naive_matmul, "args": (A, B)},
        {"name": "Inner Reordering, in C", "fn": order_matmul, "args": (A, B)},
        {"name": "Block Tiling, in C", "fn": tiled_matmul, "args": (A, B)}
    ]
    
    current_results = []
    for be in backends:
        name = be["name"]
        fn = be["fn"]
        args = be.get("args", ())
        kwargs = be.get("kwargs", {})

        out = fn(*args, **kwargs)
        check_correctness(out, ref, name)

        avg_sec = benchmark(fn, *args, warmup=1, iters=5, **kwargs)
        gflops = (2 * M * N * K) / avg_sec / 1e9
        
        current_results.append({
            "Matrix Size (M,K,N)": f"({M},{K},{N})",
            "Backend": name, 
            "Avg Time (ms)": avg_sec * 1e3, 
            "GFLOPs": gflops
        })
        
    df = pd.DataFrame(current_results)
    print(df.to_string(index=False))