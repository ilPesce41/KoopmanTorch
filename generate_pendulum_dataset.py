from koopman.data.base import generate_dataset
from koopman.data.pendulum import generate_pendulum
import torch


time_stop = 1.0
n_points = 51
num_samples = 15000

hyper_params = {}
dataset = generate_dataset(hyper_params, time_stop, n_points, generate_pendulum, num_samples)
print(dataset.shape)

torch.save(dataset, 'pendulum.pt')