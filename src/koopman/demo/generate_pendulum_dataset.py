from koopman.data.base import generate_dataset
from koopman.data.pendulum import generate_pendulum
import torch
import sys

if __name__ == "__main__":

    save_path = sys.argv[1] # TODO: Set up proper argument parsing
    time_stop = 1.0
    n_points = 51
    num_samples = 15000

    hyper_params = {}
    dataset = generate_dataset(hyper_params, time_stop, n_points, generate_pendulum, num_samples)
    print(dataset.shape)

    torch.save(dataset, save_path)