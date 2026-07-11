"""Test for FL by using library PyTorch-flower
the result is the loss getting decrease after several epochs
AND the accurancy for each clients improved"""

import os
import argparse
from collections import OrderedDict
from pathlib import Path

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms


# -----------------------------
#  PyTorch model
# -----------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(4608, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))      # [batch, 16, 26, 26]
        x = F.relu(self.conv2(x))      # [batch, 32, 24, 24]
        x = F.max_pool2d(x, 2)         # [batch, 32, 12, 12]
        x = torch.flatten(x, 1)        # [batch, 4608]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -----------------------------
# Train
# -----------------------------
def train(model, trainloader, epochs, device):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for _ in range(epochs):
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()


def test(model, testloader, device):
    model.eval()
    loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)

            output = model(x)
            loss += F.cross_entropy(output, y, reduction="sum").item()

            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    loss /= total
    accuracy = correct / total
    return loss, accuracy


# -----------------------------
# Convert PyTorch parameters to NumPy
# -----------------------------
def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)


# -----------------------------
# Load and split MNIST dataset
# -----------------------------
def load_data(client_id, num_clients):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    testset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    # IID split: split MNIST equally among clients
    num_examples = len(trainset)
    indices = np.arange(num_examples)

    np.random.seed(42)
    np.random.shuffle(indices)

    client_indices = np.array_split(indices, num_clients)[client_id]
    client_trainset = Subset(trainset, client_indices)

    trainloader = DataLoader(client_trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=128, shuffle=False)

    return trainloader, testloader


# -----------------------------
# Flower client
# -----------------------------
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, num_clients, device):
        self.client_id = client_id
        self.device = device

        self.model = Net().to(device)
        self.trainloader, self.testloader = load_data(client_id, num_clients)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)

        train(
            model=self.model,
            trainloader=self.trainloader,
            epochs=1,
            device=self.device,
        )

        return (
            get_parameters(self.model),
            len(self.trainloader.dataset),
            {},
        )

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)

        loss, accuracy = test(
            model=self.model,
            testloader=self.testloader,
            device=self.device,
        )

        return (
            float(loss),
            len(self.testloader.dataset),
            {"accuracy": float(accuracy)},
        )


# -----------------------------
# Client
# -----------------------------
def client_fn(cid, num_clients=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return FlowerClient(
        client_id=int(cid),
        num_clients=num_clients,
        device=device,
    ).to_client()


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def aggregate_parameters(results):
    total_examples = sum(num_examples for _, num_examples in results)
    aggregated = []

    for param_values in zip(*(parameters for parameters, _ in results)):
        weighted_param = sum(
            param * num_examples / total_examples
            for param, (_, num_examples) in zip(param_values, results)
        )
        aggregated.append(weighted_param)

    return aggregated


def load_checkpoint(checkpoint_path, model, device):
    if not checkpoint_path:
        return

    path = Path(checkpoint_path)
    if not path.exists():
        return

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded checkpoint: {path}")


def save_checkpoint(checkpoint_path, model):
    if not checkpoint_path:
        return

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, path)
    print(f"Saved checkpoint: {path}")


# -----------------------------
# Start Flower server/client
# -----------------------------
def start_server(num_clients, num_rounds, server_address):
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


def start_client(client_id, num_clients, server_address):
    fl.client.start_client(
        server_address=server_address,
        client=client_fn(str(client_id), num_clients),
        insecure=True,
    )


def start_simulation(num_clients, num_rounds, checkpoint_path=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = Net().to(device)
    load_checkpoint(checkpoint_path, global_model, device)
    global_parameters = get_parameters(global_model)

    print(f"Device: {device}")
    print(f"Number of clients: {num_clients}")
    print("Start local FL simulation")

    for round_idx in range(1, num_rounds + 1):
        print(f"\n========== Round {round_idx} ==========")

        fit_results = []
        eval_results = []

        for client_id in range(num_clients):
            client = FlowerClient(client_id, num_clients, device)
            parameters, num_examples, _ = client.fit(global_parameters, {})
            fit_results.append((parameters, num_examples))

            loss, test_examples, metrics = client.evaluate(parameters, {})
            eval_results.append((test_examples, {"loss": loss, **metrics}))

            print(
                f"Client {client_id}: "
                f"test loss = {loss:.4f}, "
                f"test acc = {metrics['accuracy'] * 100:.2f}%"
            )

        global_parameters = aggregate_parameters(fit_results)
        set_parameters(global_model, global_parameters)
        avg_loss = sum(num_examples * m["loss"] for num_examples, m in eval_results)
        avg_loss /= sum(num_examples for num_examples, _ in eval_results)
        avg_accuracy = weighted_average(
            [
                (num_examples, {"accuracy": metrics["accuracy"]})
                for num_examples, metrics in eval_results
            ]
        )["accuracy"]

        print("--------------------------------")
        print(f"Round {round_idx} summary:")
        print(f"Average test loss: {avg_loss:.4f}")
        print(f"Average test acc:  {avg_accuracy * 100:.2f}%")

    print("\nTraining finished.")
    save_checkpoint(checkpoint_path, global_model)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("server", "client", "simulation"),
        default="server",
        help="Run Flower as a server, a real client, or a Ray-based simulation.",
    )
    parser.add_argument("--client-id", type=int, default=0)
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--checkpoint-path", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "server":
        start_server(args.num_clients, args.num_rounds, args.server_address)
    elif args.mode == "client":
        start_client(args.client_id, args.num_clients, args.server_address)
    else:
        start_simulation(args.num_clients, args.num_rounds, args.checkpoint_path)
