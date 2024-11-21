from torch.nn import Module, Linear, ReLU, Sequential, ModuleList
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


class AuxillaryNetwork(Module):

    def __init__(self, num_real: int, num_complex_conjugate_pairs: int, num_layers: int, hidden_dim: int):
        super(AuxillaryNetwork, self).__init__()

        self.num_real = num_real
        self.num_complex_conjugate_pairs = num_complex_conjugate_pairs

        self.layers = ModuleList()
        for _ in range(num_real):
            mlp = []
            mlp.append(Linear(1, hidden_dim))
            mlp.append(ReLU())
            for _ in range(num_layers):
                mlp.append(Linear(hidden_dim, hidden_dim))
                mlp.append(ReLU())
            mlp.append(Linear(hidden_dim, 1))
            self.layers.append(Sequential(*mlp))
        for _ in range(num_complex_conjugate_pairs):
            mlp = []
            mlp.append(Linear(1, hidden_dim))
            mlp.append(ReLU())
            for _ in range(num_layers):
                mlp.append(Linear(hidden_dim, hidden_dim))
                mlp.append(ReLU())
            mlp.append(Linear(hidden_dim, 2))
            self.layers.append(Sequential(*mlp))

    def forward(self, z: Tensor) -> Tensor:
        
        assert z.shape[-1] == self.num_real + 2 * self.num_complex_conjugate_pairs
        real_lambda = []
        for i in range(self.num_real):
            real_lambda.append(self.layers[i](z[..., i:i+1]))
        if self.num_real > 0:
            real_lambda = torch.stack(real_lambda, dim=-2).squeeze(-1)
        else:
            real_lambda = None
        base = self.num_real
        mu_omega = []
        for i in range(self.num_complex_conjugate_pairs):
            z_mag = z[..., base:base+1]**2 + z[..., base+1:base+2]**2
            mu_omega.append(self.layers[self.num_real+i](z_mag))
            base += 2
        if self.num_complex_conjugate_pairs > 0:
            mu_omega = torch.stack(mu_omega, dim=-2)
            mu = mu_omega[..., 0]
            omega = mu_omega[..., 1]
        else:
            mu = None
            omega = None
        return real_lambda, mu, omega


class KoopmanOperator(Module):

    def __init__(self, auxillary_network: AuxillaryNetwork):
        super(KoopmanOperator, self).__init__()

        self.auxillary_network = auxillary_network
        self.num_real = auxillary_network.num_real
        self.num_complex_conjugate_pairs = auxillary_network.num_complex_conjugate_pairs


    def forward(self, z: Tensor) -> Tensor:
        
        batch_size, seq_len, c = z.shape
        assert c == self.num_real + 2 * self.num_complex_conjugate_pairs

        z = z.reshape(-1, c)

        real_lambda, mu, omega = self.auxillary_network(z)
        z_pred = torch.zeros_like(z)
        for i in range(len(z_pred)):

            if self.num_real > 0 and self.num_complex_conjugate_pairs > 0:
                l, m, o = real_lambda[i], mu[i], omega[i]
            elif self.num_real > 0:
                l = real_lambda[i]
                m = torch.zeros(0).to(z.device)
                o = torch.zeros(0).to(z.device)
            else:
                l = torch.zeros(0).to(z.device)
                m, o = mu[i], omega[i]

            koopman_mat = build_koopman_operator(self.num_real, self.num_complex_conjugate_pairs, l, m, o)
            z_pred[i] = z[i] @ koopman_mat
        z_pred = z_pred.reshape(batch_size, seq_len, c)
        return z_pred
