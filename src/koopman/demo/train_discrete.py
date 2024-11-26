from koopman.models.lightning_model import KoopmanLightningModel
from torch.utils.data import TensorDataset
import torch
from lightning import Trainer
from lightning import seed_everything
import lightning
from lightning.pytorch.loggers import TensorBoardLogger
import sys

if __name__ == "__main__":

    # Define the model
    model = KoopmanLightningModel(
        input_dim=2,
        latent_dim=2,
        num_hidden_layers=3,
        hidden_dim=30,
        num_real=2,
        num_complex_conjugate_pairs=0,
        num_pred=30,
        learning_rate=1e-3,
        pretrain=False
    )

    data = torch.load(sys.argv[1]) # TODO: Set up proper argument parsing
    dataset = TensorDataset(data)

    # seed pytorch lighting
    seed_everything(42)

    # Split dataset into train/val/test 0.6, 0.2, 0.2
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    num_workers=1
    batch_size=256
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logger = TensorBoardLogger("discrete_logs", name="koopman_model")

    # Train the model
    trainer = Trainer(logger=logger)
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    # trainer.test(model, test_loader)