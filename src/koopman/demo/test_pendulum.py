from koopman.models.lightning_model import KoopmanLightningModel
import torch
import matplotlib.pyplot as plt
import numpy as np
torch.set_grad_enabled(False)
import matplotlib.cm as cm
import sys

if __name__ == "__main__":
    checkpoint = sys.argv[1] # TODO: Set up proper argument parsing
    model = KoopmanLightningModel.load_from_checkpoint(checkpoint,
            input_dim=2,
            latent_dim=2,
            num_hidden_layers=3,
            hidden_dim=80,
            num_real=0,
            num_complex_conjugate_pairs=1,
            num_pred=30,
            learning_rate=1e-3,
            pretrain=False)

    koopman_model = model.model

    def get_init():
        while True:
            x1 = np.random.uniform(-3.1, 3.1)
            bound = np.sqrt(2*(np.cos(x1)+0.99))
            if not np.isnan(bound):
                break
        x2 = np.random.uniform(-bound, bound)
        return np.array([x1, x2])

    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    num_pred = 1000
    for _ in  np.linspace(-3.1, 3.1, 10):
        latent = []
        pred = []

        x0 = get_init()
        x0 = torch.tensor(x0).float().to('mps').unsqueeze(0).unsqueeze(0)
        z, x_hat = koopman_model(x0)
        pred.append(x_hat.squeeze().cpu().numpy())
        latent.append(z.squeeze().cpu().numpy())
        for i in range(1, num_pred):
            z, x_hat = koopman_model.predict(z)
            latent.append(z.squeeze().cpu().numpy())
            pred.append(x_hat.squeeze().cpu().numpy())
        
        pred = np.vstack(pred)
        latent = np.vstack(latent)
        ax.scatter(pred[:,0], pred[:,1], c=cm.jet(np.linspace(0,1,num_pred)))
        ax2.scatter(latent[:,0], latent[:,1], c=cm.jet(np.linspace(0,1,num_pred)))
        ax3.plot(np.linspace(0, 1, num_pred), pred[:,0])
        ax4.plot(np.linspace(0, 1, num_pred), pred[:,1])
    plt.show()