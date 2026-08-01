import copy
import math
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
_SPLIT_CACHE = {}
MIN_DIRICHLET_ALPHA = 1e-3


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


def validate_noniid_alpha(noniid_alpha):
    alpha = float(noniid_alpha)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("--noniid-alpha must be in the range [0, 1]")
    return alpha


def partition_config(num_clients, noniid_alpha):
    return {
        "num_clients": int(num_clients),
        "noniid_alpha": validate_noniid_alpha(noniid_alpha),
        "seed": SEED,
    }


def split_indices_iid(dataset_size, num_clients):
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(dataset_size)
    return np.array_split(indices, num_clients)


def split_indices_dirichlet(dataset, num_clients, noniid_alpha):
    alpha = max(validate_noniid_alpha(noniid_alpha), MIN_DIRICHLET_ALPHA)
    rng = np.random.default_rng(SEED)
    targets = np.asarray(dataset.targets)
    class_indices = []

    for class_id in np.unique(targets):
        indices = np.where(targets == class_id)[0]
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        split_points = (np.cumsum(proportions)[:-1] * len(indices)).astype(int)
        class_indices.extend(np.array_split(indices, split_points))

    client_indices = [[] for _ in range(num_clients)]
    for client_id, indices in enumerate(class_indices):
        client_indices[client_id % num_clients].extend(indices.tolist())

    for client_id in range(num_clients):
        rng.shuffle(client_indices[client_id])

    empty_client_ids = [
        client_id
        for client_id, indices in enumerate(client_indices)
        if not indices
    ]
    for empty_client_id in empty_client_ids:
        donor_client_id = max(
            range(num_clients),
            key=lambda client_id: len(client_indices[client_id]),
        )
        if len(client_indices[donor_client_id]) <= 1:
            break
        client_indices[empty_client_id].append(client_indices[donor_client_id].pop())

    return [np.asarray(indices, dtype=np.int64) for indices in client_indices]


def split_indices(dataset, num_clients, noniid_alpha=1.0):
    alpha = validate_noniid_alpha(noniid_alpha)
    cache_key = (id(dataset), num_clients, alpha)
    if cache_key not in _SPLIT_CACHE:
        if alpha >= 1.0:
            _SPLIT_CACHE[cache_key] = split_indices_iid(len(dataset), num_clients)
        else:
            _SPLIT_CACHE[cache_key] = split_indices_dirichlet(
                dataset,
                num_clients,
                alpha,
            )
    return _SPLIT_CACHE[cache_key]


def client_size(num_clients, client_id, noniid_alpha=1.0):
    dataset = load_mnist_dataset(train=True)
    return len(split_indices(dataset, num_clients, noniid_alpha)[client_id])


def client_subset(client_id, num_clients, noniid_alpha=1.0):
    dataset = load_mnist_dataset(train=True)
    indices = split_indices(dataset, num_clients, noniid_alpha)[client_id]
    return Subset(dataset, indices)


def get_batch(client_id, num_clients, batch_index, batch_size, device, noniid_alpha=1.0):
    loader = DataLoader(
        client_subset(client_id, num_clients, noniid_alpha=noniid_alpha),
        batch_size=batch_size,
        shuffle=False,
    )

    for current_index, batch in enumerate(loader):
        if current_index == batch_index:
            x, y = batch
            return x.to(device), y.to(device)

    raise IndexError(f"batch_index {batch_index} is outside client {client_id} data")


def load_data(
    client_id,
    num_clients,
    train_batch_size=32,
    test_batch_size=128,
    noniid_alpha=1.0,
):
    trainloader = DataLoader(
        client_subset(client_id, num_clients, noniid_alpha=noniid_alpha),
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


def load_split_checkpoint(checkpoint_path, num_clients, device, noniid_alpha=1.0):
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

    expected_partition = partition_config(num_clients, noniid_alpha)
    saved_partition = checkpoint.get("partition")
    if saved_partition is None:
        print(
            "Loaded legacy checkpoint without partition metadata; "
            "cannot verify non-IID alpha consistency."
        )
    elif saved_partition != expected_partition:
        raise ValueError(
            "Checkpoint partition config does not match current run: "
            f"checkpoint={saved_partition}, current={expected_partition}"
        )

    print(f"Loaded checkpoint: {path}")
    return client_states, checkpoint["server_model"]


def save_split_checkpoint(
    checkpoint_path,
    client_states,
    server_model,
    num_clients=None,
    noniid_alpha=1.0,
):
    if not checkpoint_path:
        return

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "client_models": [copy.deepcopy(state) for state in client_states],
            "server_model": copy.deepcopy(server_model.state_dict()),
            "partition": (
                partition_config(num_clients, noniid_alpha)
                if num_clients is not None
                else None
            ),
        },
        path,
    )
    print(f"Saved checkpoint: {path}")


def check_simulation_backend(
    num_clients,
    client_num_cpus,
    simulation_name="Flower simulation",
):
    print("Checking Flower simulation backend...", flush=True)
    total_num_cpus = max(1, math.ceil(num_clients * client_num_cpus))
    try:
        import ray
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"{simulation_name} requires the Ray backend, but `ray` is not "
            "installed for this Python environment. On Windows, Ray is not "
            "available for all Python versions; use a Python version with a Ray "
            "wheel, or run this in WSL2."
        ) from exc

    try:
        ray.init(
            num_cpus=total_num_cpus,
            include_dashboard=False,
        )
        ray.shutdown()
    except Exception as exc:
        raise RuntimeError(
            f"{simulation_name} could not start Ray. Training did not start, so "
            "no round output was produced. Run this in WSL2/Linux or use a "
            "Python/Ray version combination that starts Ray successfully on "
            "this machine."
        ) from exc

    return total_num_cpus


def build_ray_backend_config(total_num_cpus, client_num_cpus, client_num_gpus=0.0):
    return {
        "init_args": {
            "num_cpus": total_num_cpus,
            "include_dashboard": False,
        },
        "client_resources": {
            "num_cpus": client_num_cpus,
            "num_gpus": client_num_gpus,
        },
    }
