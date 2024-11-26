from torch.nn import Module, Linear, ReLU, Sequential, ModuleList
from torch import Tensor
import torch

def build_koopman_operator(num_real: int, num_complex_conjugate_pairs: int, real_lambda: Tensor, mu: Tensor, omega: Tensor) -> Tensor:
    
    if num_real > 0:
        batch_size =  real_lambda.shape[0]
        assert real_lambda.shape == (batch_size, num_real)
    if num_complex_conjugate_pairs > 0:
        batch_size =  mu.shape[0]
        assert mu.shape == (batch_size, num_complex_conjugate_pairs)
        assert omega.shape == (batch_size, num_complex_conjugate_pairs)
    device = real_lambda.device if num_real > 0 else mu.device

    n = num_real + 2 * num_complex_conjugate_pairs
    koopman_operator = torch.zeros(batch_size, n, n, device=device)

    if num_real > 0:
        koopman_operator[:, :num_real, :num_real] = torch.diag_embed(real_lambda)

    if num_complex_conjugate_pairs > 0:
        base = num_real
        mu_exp = torch.exp(mu)
        cos_omega = torch.cos(omega)
        sin_omega = torch.sin(omega)

        indices = torch.arange(num_complex_conjugate_pairs, device=device)
        base_indices = base + 2 * indices

        koopman_operator[:, base_indices, base_indices] = mu_exp * cos_omega
        koopman_operator[:, base_indices, base_indices + 1] = -mu_exp * sin_omega
        koopman_operator[:, base_indices + 1, base_indices] = mu_exp * sin_omega
        koopman_operator[:, base_indices + 1, base_indices + 1] = mu_exp * cos_omega

    return koopman_operator


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
        n = self.num_real + 2 * self.num_complex_conjugate_pairs


    def forward(self, z: Tensor) -> Tensor:
        
        batch_size, seq_len, c = z.shape
        assert c == self.num_real + 2 * self.num_complex_conjugate_pairs

        z = z.reshape(-1, c)

        real_lambda, mu, omega = self.auxillary_network(z)
        z_pred = torch.zeros_like(z)
        koopman_mat = build_koopman_operator(self.num_real, self.num_complex_conjugate_pairs, real_lambda, mu, omega)
        z_pred = torch.bmm(z.unsqueeze(1), koopman_mat).squeeze(1)

        z_pred = z_pred.reshape(batch_size, seq_len, c)
        return z_pred
