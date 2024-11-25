from lightning import LightningModule
import torch
from torch import nn
from torch.optim import AdamW
from koopman.models.base_model import BaseKoopmanAutoencoder
from koopman.models.operator import KoopmanOperator, AuxillaryNetwork
from koopman.losses.loss import ReconstructionLoss, LinearityLoss, PredictionLoss
from collections import defaultdict

class KoopmanLightningModel(LightningModule):
    def __init__(self, input_dim, latent_dim, num_hidden_layers, hidden_dim, num_real, num_complex_conjugate_pairs, num_pred=30, learning_rate=1e-3, pretrain=False):
        super(KoopmanLightningModel, self).__init__()

        encoder = []
        encoder.append(nn.Linear(input_dim, hidden_dim))
        for i in range(num_hidden_layers):
            encoder.append(nn.Linear(hidden_dim, hidden_dim))
            encoder.append(nn.ReLU())
        encoder.append(nn.Linear(hidden_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        decoder.append(nn.Linear(latent_dim, hidden_dim))
        for i in range(num_hidden_layers):
            decoder.append(nn.Linear(hidden_dim, hidden_dim))
            decoder.append(nn.ReLU())
        decoder.append(nn.Linear(hidden_dim, input_dim))
        self.decoder = nn.Sequential(*decoder)

        aux_net = AuxillaryNetwork(num_real, num_complex_conjugate_pairs, num_hidden_layers, hidden_dim)
        koopman_operator = KoopmanOperator(aux_net)

        self.model = BaseKoopmanAutoencoder(self.encoder, self.decoder, koopman_operator)
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()

        self.num_pred = num_pred
        self.reconstruction_loss = ReconstructionLoss()
        self.linearity_loss = LinearityLoss()
        self.prediction_loss = PredictionLoss()
        self.pretrain = pretrain

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        
        x = batch[0]
        z, x_hat = self.model(x)
        z_i = z
        losses = defaultdict(lambda : 0)
        losses['reconstruction_loss'] += self.reconstruction_loss(x, x_hat)
        n_pred_steps = min(self.num_pred+1, x.shape[1])
        if not self.pretrain:
            for i in range(1, n_pred_steps):
                z_i, x_hat_i = self.model.predict(z_i)
                losses['prediction_loss'] += self.prediction_loss(x_hat_i[:, :-i], x[:, i:]) / n_pred_steps
                losses['linearity_loss'] += self.linearity_loss(z_i[:, :-i], z[:, i:]) / n_pred_steps
        loss = 0.0
        for key, value in losses.items():
            self.log(key, value)
            loss += value
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        z, x_hat = self.model(x)
        z_i = z
        losses = defaultdict(lambda : 0)
        losses['reconstruction_loss'] += self.reconstruction_loss(x, x_hat)
        n_pred_steps = min(self.num_pred+1, x.shape[1])
        for i in range(1, n_pred_steps):
            z_i, x_hat_i = self.model.predict(z_i)
            losses['prediction_loss'] += self.prediction_loss(x_hat_i[:, :-i], x[:, i:]) / n_pred_steps
            losses['linearity_loss'] += self.linearity_loss(z_i[:, :-i], z[:, i:]) / n_pred_steps
        
        loss = 0.0
        for key, value in losses.items():
            self.log(key, value)
            loss += value
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch[0]
        z, x_hat = self.model(x)
        z_i = z
        losses = defaultdict(lambda : 0)
        losses['reconstruction_loss'] += self.reconstruction_loss(x, x_hat)
        n_pred_steps = min(self.num_pred+1, x.shape[1])
        for i in range(1, n_pred_steps):
            z_i, x_hat_i = self.model.predict(z_i)
            losses['prediction_loss'] += self.prediction_loss(x_hat_i[:, :-i], x[:, i:]) / n_pred_steps
            losses['linearity_loss'] += self.linearity_loss(z_i[:, :-i], z[:, i:]) / n_pred_steps
        loss = 0.0
        for key, value in losses.items():
            self.log(key, value)
            loss += value
        self.log('test_loss', loss)
        return loss
            

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer