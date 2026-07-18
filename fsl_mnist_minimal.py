import argparse
import copy
from functools import partial
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from flower_split_message import run_message_simulation


# -------------Basic settings---------------------
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
#learning rate
LR_CLIENT = 0.01
LR_SERVER = 0.01
_DATASET_CACHE = {}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

#------------ Dataset split-------------

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
        transform=transform
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

def split_dataset_iid(dataset, num_clients):
    """
    Split dataset indices equally and randomly among clients.
    IID splitting.
    """
    indices = np.random.permutation(len(dataset))
    split_size = len(dataset) // num_clients

    client_indices = []
    for i in range(num_clients):
        start = i * split_size
        end = (i + 1) * split_size if i != num_clients - 1 else len(dataset)
        client_indices.append(indices[start:end])

    return client_indices



# --------------Client-side model--------------

class ClientNet(nn.Module):
    """
    on the client side.
    Input: MNIST image [1, 28, 28]
    Output: feature tensor
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # [16, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [16, 14, 14]
        )

    def forward(self, x):
        return self.features(x)


# -------------Server-side model-------------

class ServerNet(nn.Module):
    """
    on the server side.
    Input: activation from ClientNet
    Output: class logits
    """

    def __init__(self):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # [32, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2),                              # [32, 7, 7]
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        return self.classifier(x)


#---------------- FedAvg for client-side models-----------------
def fedavg(state_dicts, sizes):
    total_size = sum(sizes)
    avg_state = copy.deepcopy(state_dicts[0])

    for key in avg_state.keys():
        avg_state[key] = sum(
            state_dicts[i][key] * (sizes[i] / total_size)
            for i in range(len(state_dicts))
        )

    return avg_state


# -------One client training under split learning----------
def train_one_client(
    client_model,
    server_model,
    dataloader,#client traning data
    optimizer_client,
    optimizer_server,
    criterion,
):
    client_model.train()
    server_model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer_client.zero_grad()
        optimizer_server.zero_grad()

        # Client-side forward
        activation = client_model(images)

        # In real FSL, activation is sent from client to server.
        # Here we simulate it in one process.
        smashed_data = activation.detach().requires_grad_(True)
        # Server-side forward
        outputs = server_model(smashed_data)
        loss = criterion(outputs, labels)
        # Server-side backward
        loss.backward()

        # Gradient w.r.t. activation.
        # In real FSL, this gradient is sent back to client.
        grad_to_client = smashed_data.grad.detach()
        optimizer_server.step()

        # Client-side backward
        activation.backward(grad_to_client)
        optimizer_client.step()

        # Statistics
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# --------------Evaluation-----------------------
def evaluate(client_model, server_model, dataloader, criterion):
    client_model.eval()
    server_model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            activation = client_model(images)
            outputs = server_model(activation)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)#find the highest possibility
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def load_checkpoint(checkpoint_path, client_model, server_model):
    if not checkpoint_path:
        return

    path = Path(checkpoint_path)
    if not path.exists():
        return

    checkpoint = torch.load(path, map_location=device)
    client_model.load_state_dict(checkpoint["global_client_model"])
    server_model.load_state_dict(checkpoint["server_model"])
    print(f"Loaded checkpoint: {path}")


def save_checkpoint(checkpoint_path, client_model, server_model):
    if not checkpoint_path:
        return

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "global_client_model": client_model.state_dict(),
            "server_model": server_model.state_dict(),
        },
        path,
    )
    print(f"Saved checkpoint: {path}")


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
    global_client_state = checkpoint["global_client_model"]
    client_states = [
        copy.deepcopy(global_client_state)
        for _ in range(num_clients)
    ]

    print(f"Loaded checkpoint: {path}")
    return client_states, checkpoint["server_model"]


def save_message_checkpoint(checkpoint_path, client_states, server_model):
    if not checkpoint_path:
        return

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "global_client_model": client_states[0],
            "server_model": server_model.state_dict(),
        },
        path,
    )
    print(f"Saved checkpoint: {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--client-num-cpus", type=float, default=1.0)
    parser.add_argument("--client-num-gpus", type=float, default=0.0)
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args()

# -----------------Main training process-------------
def main():
    args = parse_args()
    if args.num_clients < 1:
        raise ValueError("--num-clients must be at least 1")

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
        fedavg_fn=fedavg,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=BATCH_SIZE,
        lr_client=LR_CLIENT,
        lr_server=LR_SERVER,
        use_client_fedavg=True,
        num_cpus=args.client_num_cpus,
        num_gpus=args.client_num_gpus,
        initial_client_states=initial_client_states,
        initial_server_state=initial_server_state,
        on_finished_fn=on_finished_fn,
        max_batches=args.max_batches or None,
    )
    return

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    client_indices = split_dataset_iid(train_dataset, args.num_clients)

    client_loaders = []
    client_sizes = []

    for i in range(args.num_clients):
        subset = Subset(train_dataset, client_indices[i])
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
        client_loaders.append(loader)
        client_sizes.append(len(subset))

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Global client-side model
    global_client_model = ClientNet().to(device)
    # Shared server-side model
    server_model = ServerNet().to(device)

    criterion = nn.CrossEntropyLoss()
    load_checkpoint(args.checkpoint_path, global_client_model, server_model)

    print(f"Device: {device}")
    print(f"Number of clients: {args.num_clients}")
    print(f"Client data sizes: {client_sizes}")
    print("Start Federated Split Learning training")

    for round_id in range(args.num_rounds):
        print(f"\n========== Round {round_id + 1} ==========")

        local_client_states = []
        round_losses = []
        round_accs = []

        # One shared server optimizer.
        # In SFLV2, server-side model is shared and continuously updated.
        optimizer_server = torch.optim.SGD(server_model.parameters(), lr=LR_SERVER)

        for client_id in range(args.num_clients):
            # Each client starts from the current global client-side model
            local_client_model = copy.deepcopy(global_client_model).to(device)

            optimizer_client = torch.optim.SGD(
                local_client_model.parameters(),
                lr=LR_CLIENT
            )

            for _ in range(args.local_epochs):
                train_loss, train_acc = train_one_client(
                    client_model=local_client_model,
                    server_model=server_model,
                    dataloader=client_loaders[client_id],
                    optimizer_client=optimizer_client,
                    optimizer_server=optimizer_server,
                    criterion=criterion,
                )

            local_client_states.append(copy.deepcopy(local_client_model.state_dict()))
            round_losses.append(train_loss)
            round_accs.append(train_acc)

            print(
                f"Client {client_id}: "
                f"train loss = {train_loss:.4f}, "
                f"train acc = {train_acc * 100:.2f}%"
            )

        # Federated averaging on client-side models
        new_global_client_state = fedavg(local_client_states, client_sizes)
        global_client_model.load_state_dict(new_global_client_state)

        # Evaluation using global client-side model + shared server-side model
        test_loss, test_acc = evaluate(
            global_client_model,
            server_model,
            test_loader,
            criterion
        )

        print("--------------------------------")
        print(f"Round {round_id + 1} summary:")
        print(f"Average train loss: {np.mean(round_losses):.4f}")
        print(f"Average train acc:  {np.mean(round_accs) * 100:.2f}%")
        print(f"Test loss:          {test_loss:.4f}")
        print(f"Test acc:           {test_acc * 100:.2f}%")

    print("\nTraining finished.")
    save_checkpoint(args.checkpoint_path, global_client_model, server_model)


if __name__ == "__main__":
    main()
