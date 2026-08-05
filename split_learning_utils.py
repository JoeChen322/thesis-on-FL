import copy
import math
import os
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, default_collate
from torchvision import datasets, transforms


SEED = 1234
_DATASET_CACHE = {}
_SPLIT_CACHE = {}
MIN_DIRICHLET_ALPHA = 1e-3
SIMULATION_TOTAL_CPUS_ENV = "SIMULATION_TOTAL_CPUS"


class ClientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return F.relu(self.fc1(x))


class CifarClientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        return F.relu(self.fc1(x))


class ServerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, smashed_data):
        return self.fc2(smashed_data)


class FullNet(nn.Module):
    def __init__(self, client_model_cls=ClientNet, server_model_cls=ServerNet):
        super().__init__()
        self.client_model = client_model_cls()
        self.server_model = server_model_cls()

    def forward(self, x):
        return self.server_model(self.client_model(x))


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_mnist_dataset(train):
    return load_dataset("mnist", train)


def normalize_dataset_name(dataset_name):
    normalized = dataset_name.lower().replace("-", "")
    if normalized in ("mnist", "cifar10"):
        return normalized
    raise ValueError("--dataset must be one of: mnist, cifar10")


def get_model_classes(dataset_name):
    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name == "cifar10":
        return CifarClientNet, ServerNet
    return ClientNet, ServerNet


def load_dataset(dataset_name, train):
    dataset_name = normalize_dataset_name(dataset_name)
    cache_key = (dataset_name, bool(train))
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    if dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        dataset = datasets.CIFAR10(
            root="./data",
            train=train,
            download=True,
            transform=transform,
        )
    else:
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
    _DATASET_CACHE[cache_key] = dataset
    return dataset


def validate_noniid_alpha(noniid_alpha):
    alpha = float(noniid_alpha)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("--noniid-alpha must be in the range [0, 1]")
    return alpha


def partition_config(num_clients, noniid_alpha, dataset_name="mnist"):
    return {
        "dataset": normalize_dataset_name(dataset_name),
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


def client_size(num_clients, client_id, noniid_alpha=1.0, dataset_name="mnist"):
    dataset = load_dataset(dataset_name, train=True)
    return len(split_indices(dataset, num_clients, noniid_alpha)[client_id])


def client_subset(client_id, num_clients, noniid_alpha=1.0, dataset_name="mnist"):
    dataset = load_dataset(dataset_name, train=True)
    indices = split_indices(dataset, num_clients, noniid_alpha)[client_id]
    return Subset(dataset, indices)


def get_batch(
    client_id,
    num_clients,
    batch_index,
    batch_size,
    device,
    noniid_alpha=1.0,
    dataset_name="mnist",
):
    dataset = load_dataset(dataset_name, train=True)
    indices = split_indices(dataset, num_clients, noniid_alpha)[client_id]
    start = batch_index * batch_size
    end = min(start + batch_size, len(indices))
    if start >= len(indices):
        raise IndexError(f"batch_index {batch_index} is outside client {client_id} data")

    x, y = default_collate([dataset[int(index)] for index in indices[start:end]])
    return x.to(device), y.to(device)


def client_label_boundary_score(
    client_id,
    num_clients,
    noniid_alpha=1.0,
    num_classes=10,
    dataset_name="mnist",
):
    dataset = load_dataset(dataset_name, train=True)
    indices = split_indices(dataset, num_clients, noniid_alpha)[client_id]
    if len(indices) == 0:
        return 1.0

    targets = np.asarray(dataset.targets)
    labels = targets[np.asarray(indices, dtype=np.int64)]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    distribution = counts / counts.sum()
    reference = np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
    mixture = 0.5 * (distribution + reference)
    return 0.5 * _kl_divergence_log2(distribution, mixture) + 0.5 * _kl_divergence_log2(reference, mixture)


def _kl_divergence_log2(left, right):
    mask = left > 0
    return float(np.sum(left[mask] * np.log2(left[mask] / right[mask])))


def load_data(
    client_id,
    num_clients,
    train_batch_size=32,
    test_batch_size=128,
    noniid_alpha=1.0,
    dataset_name="mnist",
):
    trainloader = DataLoader(
        client_subset(
            client_id,
            num_clients,
            noniid_alpha=noniid_alpha,
            dataset_name=dataset_name,
        ),
        batch_size=train_batch_size,
        shuffle=True,
    )
    testloader = DataLoader(
        load_dataset(dataset_name, train=False),
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


def load_split_checkpoint(
    checkpoint_path,
    num_clients,
    device,
    noniid_alpha=1.0,
    dataset_name="mnist",
):
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

    expected_partition = partition_config(num_clients, noniid_alpha, dataset_name)
    saved_partition = checkpoint.get("partition")
    if saved_partition is None:
        print(
            "Loaded legacy checkpoint without partition metadata; "
            "cannot verify non-IID alpha consistency."
        )
    elif "dataset" not in saved_partition:
        if normalize_dataset_name(dataset_name) != "mnist":
            raise ValueError(
                "Checkpoint has no dataset metadata and can only be reused with MNIST"
            )
        legacy_expected = {
            "num_clients": expected_partition["num_clients"],
            "noniid_alpha": expected_partition["noniid_alpha"],
            "seed": expected_partition["seed"],
        }
        if saved_partition != legacy_expected:
            raise ValueError(
                "Checkpoint partition config does not match current run: "
                f"checkpoint={saved_partition}, current={legacy_expected}"
            )
        print("Loaded legacy MNIST checkpoint without dataset metadata.")
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
    dataset_name="mnist",
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
                partition_config(num_clients, noniid_alpha, dataset_name)
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
    requested_cpus = max(1, math.ceil(num_clients * client_num_cpus))
    total_num_cpus = simulation_total_num_cpus()
    try:
        import ray  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"{simulation_name} requires the Ray backend, but `ray` is not "
            "installed for this Python environment. On Windows, Ray is not "
            "available for all Python versions; use a Python version with a Ray "
            "wheel, or run this in WSL2."
        ) from exc

    print(f"Ray total CPU slots: {total_num_cpus}", flush=True)
    print(f"Requested client CPU slots: {requested_cpus}", flush=True)
    if requested_cpus > total_num_cpus:
        print(
            f"Requested client CPU slots exceed Ray total CPU slots; "
            f"clients will run in waves.",
            flush=True,
        )

    return total_num_cpus


def simulation_total_num_cpus():
    configured_total_cpus = os.environ.get(SIMULATION_TOTAL_CPUS_ENV)
    if configured_total_cpus:
        total_num_cpus = int(configured_total_cpus)
        if total_num_cpus < 1:
            raise ValueError(f"{SIMULATION_TOTAL_CPUS_ENV} must be at least 1")
        return total_num_cpus

    return max(1, torch.get_num_threads())


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
