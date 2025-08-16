# matmul kernels

specifically written matmul kernels to learn hardware aware optimiziation

CPU := "Intel(R) Core(TM) i5-8300H CPU @ 2.30GHz"

GPU := "NVIDIA 1080 Mobile, CUDA Compute 6.1, CUDA Version 11.8"

cpu variations (numpy, numba, C):
1. numpy @
2. numba jit
3. naive in C
4. inner reordering
5. block tiling
6. (more ...)

gpu variations (cuda, thrust, triton, torch):
1. torch @
2. triton jit
3. cuda all threads
4. cuda tiling
5. coalescing
6. cccl
7. (more ...)


dependencies:

```
apt: build-essential (gcc), nvidia-cuda-toolkit (nvcc)
pip: pandas, cffi, ninja, numpy, numba
Torch ecosystem
```


# run 
1. put matrix sizes in config.py:

2. compile c kernels through make:
```
make
python3 cpu.py
```

3. gpu kernels are dynamically binded (so initial run is slow)
```
mkdir impl/gpu_impl/build
python3 gpu.py
```

# bechmarks

cpu:

| Matrix Size (M,K,N) | Backend | Avg Time (ms) | GFLOPs |
|---|---|---|---|
| (1024,1024,1024) | NumPy | 7.161171 | 299.878839 |
| (1024,1024,1024) | Numba | 374.500672 | 5.734258 |
| (1024,1024,1024) | 3 Loop Impl, in C | 1622.727298 | 1.323379 |
| (1024,1024,1024) | Inner Reordering, in C | 121.087351 | 17.734996 |
| (1024,1024,1024) | Block Tiling, in C | 212.803572 | 10.091389 |

gpu:

| Matrix Size (M,K,N) | Backend | Avg Time (ms) | GFLOPs |
|---|---|---|---|
| (1024,1024,1024) | Torch (GPU) | 1.555540 | 1380.538653 |
| (1024,1024,1024) | Triton | 1.884994 | 1139.252246 |
| (1024,1024,1024) | CUDA 3 loops | 22.093896 | 97.198052 |
| (1024,1024,1024) | CUDA (coalescing) | 20.897595 | 102.762239 |
| (1024,1024,1024) | CUDA with shared memory tiling | 8.344176 | 257.363171 |
| (1024,1024,1024) | CUDA through Thrust | 29.101638 | 73.792535 |

# resources:
- https://siboehm.com/articles/22/CUDA-MMM
- https://www.youtube.com/channel/UCJgIbYl6C5no72a0NUAPcTA
- https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf
- https://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf
- https://www.youtube.com/watch?v=VgSQ1GOC86s