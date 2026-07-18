import random
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


SEED = 1234
_DATASET_CACHE = {}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_mnist_dataset(train):
    if train in _DATASET_CACHE:
        return _DATASET_CACHE[train]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(
        root="./data",
        train=train,
        download=True,
        transform=transform,
    )
    _DATASET_CACHE[train] = dataset
    return dataset


def split_indices(dataset_size, num_clients):
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(dataset_size)
    return np.array_split(indices, num_clients)


def client_size(num_clients, client_id):
    dataset = load_mnist_dataset(train=True)
    return len(split_indices(len(dataset), num_clients)[client_id])


def get_batch(client_id, num_clients, batch_index, batch_size, device):
    dataset = load_mnist_dataset(train=True)
    indices = split_indices(len(dataset), num_clients)[client_id]
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    for current_index, batch in enumerate(loader):
        if current_index == batch_index:
            x, y = batch
            return x.to(device), y.to(device)

    raise IndexError(f"batch_index {batch_index} is outside client {client_id} data")


def fedavg_state_dicts(state_dicts, sizes):
    total_size = sum(sizes)
    avg_state = OrderedDict()

    for key in state_dicts[0].keys():
        avg_state[key] = sum(
            state_dicts[idx][key] * (sizes[idx] / total_size)
            for idx in range(len(state_dicts))
        )

    return avg_state
