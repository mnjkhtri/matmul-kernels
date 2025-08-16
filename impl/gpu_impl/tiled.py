import os
import torch
from torch.utils.cpp_extension import load

_SRC_CU_PATH = os.path.join(os.path.dirname(__file__), "src/tiled.cu")

_LIB = load(
    name="cuda_tiled_matmul",
    sources=[_SRC_CU_PATH],
    verbose=True,
    build_directory=os.path.join(os.path.dirname(__file__), "build")
)

def cuda_tiled_matmul(A: torch.Tensor, B: torch.Tensor):
    return _LIB.tiled_matmul(A, B)