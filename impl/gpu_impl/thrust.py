import os
import torch
from torch.utils.cpp_extension import load

_SRC_CU_PATH = os.path.join(os.path.dirname(__file__), "src/thrust.cu")

# if nvcc is used directly it needs to know path to Torch.h
_LIB = load(
    name="cuda_thrust_matmul",
    sources=[_SRC_CU_PATH],
    verbose=True,
    build_directory=os.path.join(os.path.dirname(__file__), "build"),
    extra_cuda_cflags=[
        "--expt-extended-lambda"
    ],

)

def cuda_thrust_matmul(A: torch.Tensor, B: torch.Tensor):
    return _LIB.thrust_matmul(A, B)