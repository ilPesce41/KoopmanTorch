from koopman.models.operator import build_koopman_operator
import torch

def test_build_koopman_operator():
    
    num_real = 1
    num_complex = 0

    real_lambda = torch.tensor([1.0])
    mu = torch.tensor([])
    omega = torch.tensor([])

    koopman_operator = build_koopman_operator(num_real, num_complex, real_lambda, mu, omega)
    assert koopman_operator.shape == (1, 1)
    assert torch.allclose(koopman_operator, real_lambda)

    num_real = 0
    num_complex = 1
    real_lambda = torch.tensor([])
    mu = torch.tensor([0.0])
    omega = torch.tensor([0.0])

    koopman_operator = build_koopman_operator(num_real, num_complex, real_lambda, mu, omega)
    assert koopman_operator.shape == (2, 2)
    assert torch.allclose(koopman_operator, torch.eye(2))

    omega = torch.tensor([torch.pi / 2])
    koopman_operator = build_koopman_operator(num_real, num_complex, real_lambda, mu, omega)
    assert koopman_operator.shape == (2, 2)
    assert torch.allclose(koopman_operator, torch.tensor([[0.0, -1.0], [1.0, 0.0]]), atol=1e-6)

    num_real = 1
    num_complex = 1
    real_lambda = torch.tensor([0.5])
    mu = torch.tensor([0.0])
    omega = torch.tensor([0.0])

    koopman_operator = build_koopman_operator(num_real, num_complex, real_lambda, mu, omega)
    assert koopman_operator.shape == (3, 3)
    assert torch.allclose(koopman_operator, torch.tensor([[0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), atol=1e-6)

