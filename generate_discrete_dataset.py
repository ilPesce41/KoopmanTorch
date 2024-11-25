from koopman.data.base import generate_dataset
from koopman.data.simple import generate_simple
import torch

mu = -0.05
lamb = -1.0

time_stop = 1.0
n_points = 51
num_samples = 5000

hyper_params = {'mu': mu, 'lamb': lamb}
dataset = generate_dataset(hyper_params, time_stop, n_points, generate_simple, num_samples)
print(dataset.shape)

torch.save(dataset, 'simple.pt')