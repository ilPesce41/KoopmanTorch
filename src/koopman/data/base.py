from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import numpy as np

def generate_dataset(hyper_params, time_stop, n_points, gen_func, num_samples):

    init_cond = set()
    dataset = []
    pbar = tqdm(total=num_samples)
    while True:
        
        cond, data = gen_func(time_stop, n_points, **hyper_params)
        if cond not in init_cond:
            init_cond.add(cond)
            dataset.append(data)
            pbar.update(1)

        if len(init_cond) == num_samples:
            break
    pbar.close()

    return torch.from_numpy(np.concatenate(dataset)).float() # (n, seq_len, dim)