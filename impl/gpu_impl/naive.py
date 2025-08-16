import os
import torch
from torch.utils.cpp_extension import load

_SRC_CU_PATH = os.path.join(os.path.dirname(__file__), "src/naive.cu")

# if nvcc is used directly it needs to know path to Torch.h
_LIB = load(
    name="cuda_naive_matmul",
    sources=[_SRC_CU_PATH],
    build_directory=os.path.join(os.path.dirname(__file__), "build"),
    verbose=True
)

def cuda_naive_matmul(A: torch.Tensor, B: torch.Tensor):
    return _LIB.naive_matmul(A, B)