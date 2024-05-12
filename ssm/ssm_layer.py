import torch 
from torch import nn
import numpy as np
from torch_scan import torch_scan

class SSM(nn.Module):
    """Defines a SSM layer
       x̄(t) = Ax(t) + Bu(t) -> Change in x(t) over time
       y(t) = Cx(t) + Du(t) -> We assume D=0 in this implementation

    Args:
        
        N : size of the parameter matrices

        The elements of each matrix are drawn from a uniform distribution between 0 and 1. 
    """
    
    def __init__(self, A, B, C):
        super().__init__()
        self.A = A
        self.B = B
        self.C = C

    def discretize_signal(self, A, B, C, step):
        """

        > To be applied on a discrete input sequence (u0, u1, ...) instead of continuous function
        ut), the SSM must be discretized by a step size 'd' that represents the resolution of the
        input. 

        > Conceptually, the inputs Uk can be viewed as sampling an implicit underlying
        continuous signal u(t), where uk = u(kA).

        > To discretize the continuous-time SSM, we use the bilinear method, which converts the
        state matrix A into an approximation A. The discrete SSM is:

            Ab = (I - d/2 * A)^-1 @ (I + d/2 * A)
            Bb = (I - d/2 * A)^-1 @ (dB)
            C remains the same

        """
        I = torch.eye(A.shape[0])
        BL = torch.linalg.inv(I - (step / 2.0) * A)
        BL2 = (I + (step / 2) * A)
        Ab = BL @ BL2
        Bb = (BL * step) @ B
        return Ab, Bb, C
    
    def scan_SSM(self, Ab, Bb, Cb, u, x0):
        if isinstance(Ab, np.ndarray):
            Ab = torch.tensor(Ab, dtype=torch.float32)
        if isinstance(Bb, np.ndarray):
            Bb = torch.tensor(Bb, dtype=torch.float32)
        if isinstance(Cb, np.ndarray):
            Cb = torch.tensor(Cb, dtype=torch.float32)
        if isinstance(u, np.ndarray):
            u = torch.tensor(u, dtype=torch.float32)
        if isinstance(x0, np.ndarray):
            x0 = torch.tensor(x0, dtype=torch.float32)

        def step(x_k_1, u_k):
            x_k = Ab @ x_k_1 + Bb @ u_k
            y_k = Cb @ x_k
            return x_k, y_k
        
        return torch_scan(step, x0, u)
    
    def run_SSM(self, A, B, C, u):
        L = u.shape[0]
        N = A.shape[0]
        Ab, Bb, Cb = self.discretize_signal(A, B, C, step = 1./L)
        return self.scan_SSM(Ab, Bb, Cb, u[:, np.newaxis], np.zeros((N,)))[1]
            
