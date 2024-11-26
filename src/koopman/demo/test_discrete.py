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
            hidden_dim=30,
            num_real=2,
            num_complex_conjugate_pairs=0,
            num_pred=300,
            learning_rate=1e-3,
            pretrain=False)

    koopman_model = model.model

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    num_pred = 300
    for x in np.linspace(-.5,.5,10):
        for y in [-.5, .5]:
            latent = []
            pred = []

            x0 = np.array([x, y])
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
    plt.show()