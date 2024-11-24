from torch.nn import Module
from torch import Tensor
from typing import Tuple
from koopman.models.operator import KoopmanOperator

class BaseKoopmanAutoencoder(Module):

    def __init__(self, encoder: Module, decoder: Module, koopman_operator: KoopmanOperator):
        super(BaseKoopmanAutoencoder, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.koopman_operator = koopman_operator

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Pass the input through the encoder, Koopman operator, and decoder.
        The Koopman operator is used to predict 1 timestep forward into the future.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, seq_len, *)

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor, Tensor]
            Output tensors (z, z_pred, x_hat, x_pred)
            z : Tensor of shape (batch_size, seq_len, latent_dim)
                Encoded input tensor
            x_hat : Tensor of shape (batch_size, seq_len, *)
                Reconstructed input tensor
        """


        z = self.encoder(x)
        x_hat = self.decoder(z)

        return z, x_hat
    
    def predict(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Predict the next state given the encoded input tensor z.

        Parameters
        ----------
        z : Tensor
            Encoded input tensor of shape (batch_size, seq_len, latent_dim)

        Returns
        -------
        Tuple[Tensor, Tensor]
            Output tensors (z_pred, x_pred)
            z_pred : Tensor of shape (batch_size, seq_len, latent_dim)
                Predicted latent tensor
            x_pred : Tensor of shape (batch_size, seq_len, *)
                Predicted output tensor
        """

        z_pred = self.koopman_operator(z)
        x_pred = self.decoder(z_pred)

        return z_pred, x_pred