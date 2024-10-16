from torch.nn import Module, Linear, ReLU
from torch import Tensor
import torch

def build_koopman_operator(num_real: int, num_complex_conjugate_pairs: int, real_lambda: Tensor, mu: Tensor, omega: Tensor) -> Tensor:
    
    assert real_lambda.shape == (num_real,)
    assert mu.shape == (num_complex_conjugate_pairs,)
    assert omega.shape == (num_complex_conjugate_pairs,)
    device = real_lambda.device if num_real > 0 else mu.device

    n = num_real + 2 * num_complex_conjugate_pairs
    koopman_operator = torch.zeros(n, n)
    for i in range(num_real):
        koopman_operator[i, i] = real_lambda[i]
    base = num_real
    for i in range(num_complex_conjugate_pairs):
        block = torch.exp(mu[i]) * torch.tensor(
            [[torch.cos(omega[i]), -torch.sin(omega[i])], 
             [torch.sin(omega[i]), torch.cos(omega[i])]]
            )
        koopman_operator[base:base+2, base:base+2] = block
        base += 2
    return koopman_operator.to(device)


