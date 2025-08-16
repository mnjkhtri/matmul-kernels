import torch


def torch_matmul(A: torch.Tensor, B: torch.Tensor):
    C = A @ B    
    torch.cuda.synchronize()
    return C