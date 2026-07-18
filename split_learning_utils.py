import copy
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


SEED = 1234
_DATASET_CACHE = {}


class ClientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return F.relu(self.fc1(x))


class ServerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, smashed_data):
        return self.fc2(smashed_data)


class FullNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.client_model = ClientNet()
        self.server_model = ServerNet()

    def forward(self, x):
        return self.server_model(self.client_model(x))


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


def client_subset(client_id, num_clients, train=True):
    dataset = load_mnist_dataset(train=train)
    indices = split_indices(len(dataset), num_clients)[client_id]
    return Subset(dataset, indices)


def get_batch(client_id, num_clients, batch_index, batch_size, device):
    loader = DataLoader(
        client_subset(client_id, num_clients, train=True),
        batch_size=batch_size,
        shuffle=False,
    )

    for current_index, batch in enumerate(loader):
        if current_index == batch_index:
            x, y = batch
            return x.to(device), y.to(device)

    raise IndexError(f"batch_index {batch_index} is outside client {client_id} data")


def load_data(client_id, num_clients, train_batch_size=32, test_batch_size=128):
    trainloader = DataLoader(
        client_subset(client_id, num_clients, train=True),
        batch_size=train_batch_size,
        shuffle=True,
    )
    testloader = DataLoader(
        load_mnist_dataset(train=False),
        batch_size=test_batch_size,
        shuffle=False,
    )
    return trainloader, testloader


def fedavg_state_dicts(state_dicts, sizes):
    total_size = sum(sizes)
    avg_state = OrderedDict()

    for key in state_dicts[0].keys():
        avg_state[key] = sum(
            state_dicts[idx][key] * (sizes[idx] / total_size)
            for idx in range(len(state_dicts))
        )

    return avg_state


def load_split_checkpoint(checkpoint_path, num_clients, device):
    if not checkpoint_path:
        return None, None

    path = Path(checkpoint_path)
    if not path.exists():
        return None, None

    checkpoint = torch.load(path, map_location=device)
    client_states = checkpoint["client_models"]
    if len(client_states) != num_clients:
        raise ValueError(
            f"Checkpoint has {len(client_states)} clients, "
            f"but current run has {num_clients} clients"
        )

    print(f"Loaded checkpoint: {path}")
    return client_states, checkpoint["server_model"]


def save_split_checkpoint(checkpoint_path, client_states, server_model):
    if not checkpoint_path:
        return

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "client_models": [copy.deepcopy(state) for state in client_states],
            "server_model": copy.deepcopy(server_model.state_dict()),
        },
        path,
    )
    print(f"Saved checkpoint: {path}")
