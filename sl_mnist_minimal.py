import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from flower_split_message import run_message_local, run_message_simulation


SEED = 1234
BATCH_SIZE = 64
LR_CLIENT = 0.01
LR_SERVER = 0.01


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------
# Dataset split
# -----------------------------
def split_dataset_iid(dataset, num_clients):
    indices = np.random.permutation(len(dataset))
    return np.array_split(indices, num_clients)


def load_data(num_clients):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    client_indices = split_dataset_iid(train_dataset, num_clients)
    client_loaders = []
    client_sizes = []

    for indices in client_indices:
        subset = Subset(train_dataset, indices)
        client_loaders.append(DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True))
        client_sizes.append(len(subset))

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    return client_loaders, client_sizes, test_loader


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


def train_one_client(
    client_model,
    server_model,
    trainloader,
    client_optimizer,
    server_optimizer,
    criterion,
    device,
):
    client_model.train()
    server_model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in trainloader:
        x, y = x.to(device), y.to(device)

        client_optimizer.zero_grad()
        server_optimizer.zero_grad()

        # Client forward: the client keeps raw data and its model locally.
        smashed_data = client_model(x)

        # Simulate sending smashed data from this client to the main server.
        smashed_data_for_server = smashed_data.detach().requires_grad_()

        # Main server forward/backward on the shared server-side model.
        output = server_model(smashed_data_for_server)
        loss = criterion(output, y)
        loss.backward()
        server_optimizer.step()

        # Simulate sending gradient back from main server to this client.
        grad_from_server = smashed_data_for_server.grad.detach()
        smashed_data.backward(grad_from_server)
        client_optimizer.step()

        total_loss += loss.item() * y.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def evaluate_client(client_model, server_model, testloader, criterion, device):
    client_model.eval()
    server_model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            smashed_data = client_model(x)
            output = server_model(smashed_data)
            loss = criterion(output, y)

            total_loss += loss.item() * y.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total


def load_checkpoint(checkpoint_path, client_models, server_model, device):
    if not checkpoint_path:
        return

    path = Path(checkpoint_path)
    if not path.exists():
        return

    checkpoint = torch.load(path, map_location=device)
    client_states = checkpoint["client_models"]
    if len(client_states) != len(client_models):
        raise ValueError(
            f"Checkpoint has {len(client_states)} clients, "
            f"but current run has {len(client_models)} clients"
        )

    for client_model, state_dict in zip(client_models, client_states):
        client_model.load_state_dict(state_dict)
    server_model.load_state_dict(checkpoint["server_model"])
    print(f"Loaded checkpoint: {path}")


def save_checkpoint(checkpoint_path, client_models, server_model):
    if not checkpoint_path:
        return

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "client_models": [
                client_model.state_dict()
                for client_model in client_models
            ],
            "server_model": server_model.state_dict(),
        },
        path,
    )
    print(f"Saved checkpoint: {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("local", "message-local", "message-simulation"),
        default="local",
        help="Run the original loop, local Flower Message API, or Ray simulation.",
    )
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

    if args.mode in ("message-local", "message-simulation"):
        runner = (
            run_message_local
            if args.mode == "message-local"
            else run_message_simulation
        )
        runner(
            client_model_cls=ClientNet,
            server_model_cls=ServerNet,
            num_clients=args.num_clients,
            num_rounds=args.num_rounds,
            local_epochs=args.local_epochs,
            batch_size=BATCH_SIZE,
            lr_client=LR_CLIENT,
            lr_server=LR_SERVER,
            use_client_fedavg=False,
            num_cpus=args.client_num_cpus,
            num_gpus=args.client_num_gpus,
            max_batches=args.max_batches or None,
        )
        return

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client_loaders, client_sizes, test_loader = load_data(args.num_clients)

    client_models = [ClientNet().to(device) for _ in range(args.num_clients)]
    client_optimizers = [
        optim.SGD(client_model.parameters(), lr=LR_CLIENT)
        for client_model in client_models
    ]

    main_server_model = ServerNet().to(device)
    server_optimizer = optim.SGD(main_server_model.parameters(), lr=LR_SERVER)
    criterion = nn.CrossEntropyLoss()
    load_checkpoint(args.checkpoint_path, client_models, main_server_model, device)

    print(f"Device: {device}")
    print(f"Number of clients: {args.num_clients}")
    print(f"Client data sizes: {client_sizes}")
    print("Start multi-client Split Learning training")

    for round_idx in range(1, args.num_rounds + 1):
        print(f"\n========== Round {round_idx} ==========")
        round_losses = []
        round_accs = []

        for client_id in range(args.num_clients):
            for _ in range(args.local_epochs):
                train_loss, train_acc = train_one_client(
                    client_model=client_models[client_id],
                    server_model=main_server_model,
                    trainloader=client_loaders[client_id],
                    client_optimizer=client_optimizers[client_id],
                    server_optimizer=server_optimizer,
                    criterion=criterion,
                    device=device,
                )

            round_losses.append(train_loss)
            round_accs.append(train_acc)
            print(
                f"Client {client_id} -> main server: "
                f"train loss = {train_loss:.4f}, "
                f"train acc = {train_acc * 100:.2f}%"
            )

        eval_losses = []
        eval_accs = []
        for client_id, client_model in enumerate(client_models):
            test_loss, test_acc = evaluate_client(
                client_model,
                main_server_model,
                test_loader,
                criterion,
                device,
            )
            eval_losses.append(test_loss)
            eval_accs.append(test_acc)

        print("--------------------------------")
        print(f"Round {round_idx} summary:")
        print(f"Average train loss: {np.mean(round_losses):.4f}")
        print(f"Average train acc:  {np.mean(round_accs) * 100:.2f}%")
        print(f"Average test loss:  {np.mean(eval_losses):.4f}")
        print(f"Average test acc:   {np.mean(eval_accs) * 100:.2f}%")

    print("\nTraining finished.")
    save_checkpoint(args.checkpoint_path, client_models, main_server_model)


if __name__ == "__main__":
    main()
