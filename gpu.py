import os
os.environ['TORCH_CUDA_ARCH_LIST'] = "6.1"
os.environ['MAX_JOBS'] = '1'

import time
import pandas as pd
import torch
from impl.gpu_impl import torch_matmul, triton_matmul, cuda_naive_matmul, cuda_coalescing_matmul, cuda_tiled_matmul, cuda_thrust_matmul
from config import MATRIX

def check_correctness(output, ref, name, atol=1e-6, rtol=1e-5):
    if not torch.allclose(output, ref, atol=atol, rtol=rtol):
        diff = (output - ref).abs()
        max_err = diff.max().item()
        raise ValueError(f"[{name}] result incorrect (max abs error = {max_err:.3e})")
    return True

def benchmark(fn, *args, warmup=1, iters=5, **kwargs):
    # warm-up
    for _ in range(warmup):
        fn(*args, **kwargs)
    # timing
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)

for M, K, N in MATRIX:
    print(f"\nBenchmarking for matrices of size ({M}, {K}) * ({K}, {N}):")
    
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(K, N, dtype=torch.float32, device="cuda")
    ref = torch.matmul(A, B)

    backends = [
        {"name": "Torch (GPU)", "fn": torch_matmul, "args": (A, B)},
        {"name": "Triton", "fn": triton_matmul, "args": (A, B)},
        {"name": "CUDA 3 loops", "fn": cuda_naive_matmul, "args": (A, B)},
        {"name": "CUDA (coalescing)", "fn": cuda_coalescing_matmul, "args": (A, B)},
        {"name": "CUDA with shared memory tiling", "fn": cuda_tiled_matmul, "args": (A, B)},
        {"name": "CUDA through Thrust", "fn": cuda_thrust_matmul, "args": (A, B)}
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