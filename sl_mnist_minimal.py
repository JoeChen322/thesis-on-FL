import argparse
from functools import partial
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from flower_split_message import run_message_simulation


SEED = 1234
BATCH_SIZE = 64
LR_CLIENT = 0.01
LR_SERVER = 0.01
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


def message_client_size(num_clients, client_id):
    dataset = load_mnist_dataset(train=True)
    return len(split_indices(len(dataset), num_clients)[client_id])


def message_get_batch(client_id, num_clients, batch_index, batch_size, device):
    dataset = load_mnist_dataset(train=True)
    indices = split_indices(len(dataset), num_clients)[client_id]
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    for current_index, batch in enumerate(loader):
        if current_index == batch_index:
            x, y = batch
            return x.to(device), y.to(device)

    raise IndexError(f"batch_index {batch_index} is outside client {client_id} data")


def message_fedavg(state_dicts, sizes):
    total_size = sum(sizes)
    avg_state = OrderedDict()

    for key in state_dicts[0].keys():
        avg_state[key] = sum(
            state_dicts[idx][key] * (sizes[idx] / total_size)
            for idx in range(len(state_dicts))
        )

    return avg_state


# -----------------------------
# Client-side model
# -----------------------------
class ClientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x


# -----------------------------
# Main server-side model
# -----------------------------
class ServerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, smashed_data):
        return self.fc2(smashed_data)


def load_message_checkpoint(
    checkpoint_path,
    num_clients,
    device,
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

    print(f"Loaded checkpoint: {path}")
    return client_states, checkpoint["server_model"]


def save_message_checkpoint(checkpoint_path, client_states, server_model):
    if not checkpoint_path:
        return

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "client_models": client_states,
            "server_model": server_model.state_dict(),
        },
        path,
    )
    print(f"Saved checkpoint: {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--client-num-cpus", type=float, default=1.0)
    parser.add_argument("--client-num-gpus", type=float, default=0.0)
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_clients < 1:
        raise ValueError("--num-clients must be at least 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initial_client_states, initial_server_state = load_message_checkpoint(
        args.checkpoint_path,
        args.num_clients,
        device,
    )

    on_finished_fn = None
    if args.checkpoint_path:
        on_finished_fn = partial(save_message_checkpoint, args.checkpoint_path)

    run_message_simulation(
        client_model_cls=ClientNet,
        server_model_cls=ServerNet,
        set_seed_fn=set_seed,
        client_size_fn=message_client_size,
        get_batch_fn=message_get_batch,
        fedavg_fn=message_fedavg,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=BATCH_SIZE,
        lr_client=LR_CLIENT,
        lr_server=LR_SERVER,
        use_client_fedavg=False,
        num_cpus=args.client_num_cpus,
        num_gpus=args.client_num_gpus,
        initial_client_states=initial_client_states,
        initial_server_state=initial_server_state,
        on_finished_fn=on_finished_fn,
        max_batches=args.max_batches or None,
    )


if __name__ == "__main__":
    main()
