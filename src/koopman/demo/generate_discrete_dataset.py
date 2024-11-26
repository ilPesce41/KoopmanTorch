from koopman.data.base import generate_dataset
from koopman.data.simple import generate_simple
import torch
import sys

if __name__ == "__main__":

    save_path = sys.argv[1] # TODO: Set up proper argument parsing
    mu = -0.05
    lamb = -1.0

    time_stop = 1.0
    n_points = 51
    num_samples = 5000

    hyper_params = {'mu': mu, 'lamb': lamb}
    dataset = generate_dataset(hyper_params, time_stop, n_points, generate_simple, num_samples)
    print(dataset.shape)

    torch.save(dataset, save_path)