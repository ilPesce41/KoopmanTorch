from koopman.models.base_model import BaseKoopmanAutoencoder
from koopman.models.operator import KoopmanOperator, AuxillaryNetwork
import torch.nn as nn
import torch

def test_base_model_forward():

    batch_size = 32
    seq_len=10
    latent_dim=5
    input_dim=10
    
    encoder = nn.Linear(input_dim, latent_dim)
    decoder = nn.Linear(latent_dim, input_dim)

    batched_input = torch.randn(batch_size, seq_len, input_dim)
    aux_net = AuxillaryNetwork(num_real=3, num_complex_conjugate_pairs=1, hidden_dim=latent_dim, num_layers=2)
    koopman_operator = KoopmanOperator(aux_net)
    model = BaseKoopmanAutoencoder(encoder, decoder, koopman_operator)

    z, x_hat = model(batched_input)

    assert z.shape == (batch_size, seq_len, latent_dim)
    assert x_hat.shape == (batch_size, seq_len, input_dim)

def test_base_model_predict():

    batch_size = 32
    seq_len=10
    latent_dim=5
    input_dim=10
    
    encoder = nn.Linear(input_dim, latent_dim)
    decoder = nn.Linear(latent_dim, input_dim)

    batched_input = torch.randn(batch_size, seq_len, input_dim)
    aux_net = AuxillaryNetwork(num_real=3, num_complex_conjugate_pairs=1, hidden_dim=latent_dim, num_layers=2)
    koopman_operator = KoopmanOperator(aux_net)
    model = BaseKoopmanAutoencoder(encoder, decoder, koopman_operator)

    z, x_hat = model(batched_input)
    z_pred, x_pred = model.predict(z)

    assert z.shape == (batch_size, seq_len, latent_dim)
    assert x_hat.shape == (batch_size, seq_len, input_dim)
    assert z_pred.shape == (batch_size, seq_len, latent_dim)
    assert x_pred.shape == (batch_size, seq_len, input_dim)

    batched_input = torch.randn(batch_size, seq_len, input_dim)
    aux_net = AuxillaryNetwork(num_real=1, num_complex_conjugate_pairs=2, hidden_dim=latent_dim, num_layers=2)
    koopman_operator = KoopmanOperator(aux_net)
    model = BaseKoopmanAutoencoder(encoder, decoder, koopman_operator)

    z, x_hat = model(batched_input)
    z_pred, x_pred = model.predict(z)

    assert z.shape == (batch_size, seq_len, latent_dim)
    assert x_hat.shape == (batch_size, seq_len, input_dim)
    assert z_pred.shape == (batch_size, seq_len, latent_dim)
    assert x_pred.shape == (batch_size, seq_len, input_dim)