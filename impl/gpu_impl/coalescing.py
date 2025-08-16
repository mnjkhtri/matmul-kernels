import os
import torch
from torch.utils.cpp_extension import load

_SRC_CU_PATH = os.path.join(os.path.dirname(__file__), "src/coalescing.cu")

# if nvcc is used directly it needs to know path to Torch.h
_LIB = load(
    name="cuda_coalescing_matmul",
    sources=[_SRC_CU_PATH],
    build_directory=os.path.join(os.path.dirname(__file__), "build"),
    verbose=True
)

def cuda_coalescing_matmul(A: torch.Tensor, B: torch.Tensor):
    return _LIB.coalescing_matmul(A, B)