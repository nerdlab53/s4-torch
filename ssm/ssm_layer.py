import torch 
from torch import nn
import numpy as np

class SSM(nn.Module):
    """Defines a SSM layer
       x̄(t) = Ax(t) + Bu(t) -> Change in x(t) over time
       y(t) = Cx(t) + Du(t) -> We assume D=0 in this implementation

    Args:
        
        N : size of the parameter matrices

        The elements of each matrix are drawn from a uniform distribution between 0 and 1. 

    Returns : 

        A, B, C, D : Parameters to be learned by gradient descent

        A : Size = NxN

        B : Size = Nx1

        C : Size = 1xN
    """
    def __init__(self, N : int):
        super().__init__()
        self.A = torch.rand(N, N)
        self.B = torch.rand(N, 1)
        self.C = torch.rand(1, N)

    def discretize_signal(self, A, B, C, step):
        I = np.eye(A.shape[0])
        BL = torch.linalg.inv(I - (step / 2.0) * A)