import torch
import triton
import triton.language as tl

@triton.jit
def triton_k(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Row and column block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # Offsets for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Pointers to A and B blocks
    A_block_ptr = A_ptr + offs_m[:, None] * stride_am + 0 * stride_ak
    B_block_ptr = B_ptr + 0 * stride_bk + offs_n[None, :] * stride_bn

    # Loop over K dimension in chunks
    for k_start in range(0, K, BLOCK_K):
        # Load A tile (BLOCK_M x BLOCK_K)
        a = tl.load(
            A_block_ptr + tl.arange(0, BLOCK_K)[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k_start + tl.arange(0, BLOCK_K)[None, :] < K),
            other=0.0
        )
        # Load B tile (BLOCK_K x BLOCK_N)
        b = tl.load(
            B_block_ptr + tl.arange(0, BLOCK_K)[:, None] * stride_bk,
            mask=(k_start + tl.arange(0, BLOCK_K)[:, None] < K) & (offs_n[None, :] < N),
            other=0.0
        )
        # Matrix multiply-accumulate
        acc += tl.dot(a, b)
        # Advance pointers for next tile
        A_block_ptr += BLOCK_K * stride_ak
        B_block_ptr += BLOCK_K * stride_bk

    # Write result back to C
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        C_ptrs, acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )

def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    # Ensure inputs are CUDA float32 tensors
    assert A.is_cuda and B.is_cuda and A.dtype == torch.float32 and B.dtype == torch.float32, \
        "Inputs must be CUDA float32 tensors"
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Inner dimensions must match: {K} != {K2}"

    # Allocate output tensor
    C = torch.empty((M, N), device='cuda', dtype=torch.float32)

    # Extract strides
    stride_am, stride_ak = A.stride(0), A.stride(1)
    stride_bk, stride_bn = B.stride(0), B.stride(1)
    stride_cm, stride_cn = C.stride(0), C.stride(1)

    # Tiling parameters (tune for your GPU)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch Triton kernel
    triton_k[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    torch.cuda.synchronize()
    return C

