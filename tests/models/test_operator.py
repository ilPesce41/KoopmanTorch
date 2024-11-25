from koopman.models.operator import build_koopman_operator, AuxillaryNetwork
import torch

def test_build_koopman_operator():
    
    num_real = 1
    num_complex = 0

    real_lambda = torch.tensor([1.0]).unsqueeze(0)
    mu = torch.tensor([]).unsqueeze(0)
    omega = torch.tensor([]).unsqueeze(0)
    koopman_operator = build_koopman_operator(num_real, num_complex, real_lambda, mu, omega)
    assert koopman_operator.shape == (1, 1, 1)
    assert torch.allclose(koopman_operator, real_lambda)

    num_real = 0
    num_complex = 1
    real_lambda = torch.tensor([]).unsqueeze(0)
    mu = torch.tensor([0.0]).unsqueeze(0)
    omega = torch.tensor([0.0]).unsqueeze(0)
    koopman_operator = build_koopman_operator(num_real, num_complex, real_lambda, mu, omega)
    assert koopman_operator.shape == (1, 2, 2)
    assert torch.allclose(koopman_operator, torch.eye(2).unsqueeze(0))

    omega = torch.tensor([torch.pi / 2]).unsqueeze(0)
    koopman_operator = build_koopman_operator(num_real, num_complex, real_lambda, mu, omega)
    assert koopman_operator.shape == (1, 2, 2)
    assert torch.allclose(koopman_operator, torch.tensor([[0.0, -1.0], [1.0, 0.0]]).unsqueeze(0), atol=1e-6)

    num_real = 1
    num_complex = 1
    real_lambda = torch.tensor([0.5]).unsqueeze(0)
    mu = torch.tensor([0.0]).unsqueeze(0)
    omega = torch.tensor([0.0]).unsqueeze(0)
    koopman_operator = build_koopman_operator(num_real, num_complex, real_lambda, mu, omega)
    assert koopman_operator.shape == (1, 3, 3)
    assert torch.allclose(koopman_operator, torch.tensor([[0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).unsqueeze(0), atol=1e-6)

def test_auxillary_network_all_real():
    
    num_real = 2
    num_complex_conjugate_pairs = 0
    num_layers = 3
    hidden_dim = 10

    auxillary_network = AuxillaryNetwork(num_real, 
                                         num_complex_conjugate_pairs, 
                                         num_layers=num_layers, 
                                         hidden_dim=hidden_dim)
    
    batch_size = 32
    seq_len = 10

    z = torch.randn(batch_size, seq_len, num_real + 2 * num_complex_conjugate_pairs)
    real_lambda, mu, omega = auxillary_network(z)
    assert real_lambda.shape == (batch_size, seq_len, num_real)
    assert mu is None
    assert omega is None

def test_auxillary_network_all_complex():

    num_real = 0
    num_complex_conjugate_pairs = 3
    num_layers = 3
    hidden_dim = 10

    auxillary_network = AuxillaryNetwork(num_real, 
                                         num_complex_conjugate_pairs, 
                                         num_layers=num_layers, 
                                         hidden_dim=hidden_dim)
    
    batch_size = 32
    seq_len = 10

    z = torch.randn(batch_size, seq_len, num_real + 2 * num_complex_conjugate_pairs)
    real_lambda, mu, omega = auxillary_network(z)
    assert real_lambda is None
    assert mu.shape == (batch_size, seq_len, num_complex_conjugate_pairs)
    assert omega.shape == (batch_size, seq_len, num_complex_conjugate_pairs)

def test_auxillary_network_both():

    num_real = 3
    num_complex_conjugate_pairs = 3
    num_layers = 3
    hidden_dim = 10

    auxillary_network = AuxillaryNetwork(num_real, 
                                         num_complex_conjugate_pairs, 
                                         num_layers=num_layers, 
                                         hidden_dim=hidden_dim)
    
    batch_size = 32
    seq_len = 10

    z = torch.randn(batch_size, seq_len, num_real + 2 * num_complex_conjugate_pairs)
    real_lambda, mu, omega = auxillary_network(z)
    assert real_lambda.shape == (batch_size, seq_len, num_real)
    assert mu.shape == (batch_size, seq_len, num_complex_conjugate_pairs)
    assert omega.shape == (batch_size, seq_len, num_complex_conjugate_pairs)



