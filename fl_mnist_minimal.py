"""Test for FL by using library PyTorch-flower
the result is the loss getting decrease after several epochs
AND the accurancy for each client improved"""

import argparse
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn.functional as F

from split_learning_utils import (
    FullNet,
    client_size,
    fedavg_state_dicts,
    load_data,
    load_split_checkpoint,
    save_split_checkpoint,
    set_seed,
)


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
# Flower client
# -----------------------------
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, num_clients, device):
        self.client_id = client_id
        self.device = device

        self.model = FullNet().to(device)
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


def load_checkpoint(checkpoint_path, model, num_clients, device):
    client_states, server_state = load_split_checkpoint(
        checkpoint_path,
        num_clients,
        device,
    )
    if client_states is None:
        return

    sizes = [client_size(num_clients, client_id) for client_id in range(num_clients)]
    model.client_model.load_state_dict(fedavg_state_dicts(client_states, sizes))
    model.server_model.load_state_dict(server_state)


def save_checkpoint(checkpoint_path, model, num_clients):
    client_states = [
        model.client_model.state_dict()
        for _ in range(num_clients)
    ]
    save_split_checkpoint(checkpoint_path, client_states, model.server_model)


def start_simulation(num_clients, num_rounds, checkpoint_path=""):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = FullNet().to(device)
    load_checkpoint(checkpoint_path, global_model, num_clients, device)
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
    save_checkpoint(checkpoint_path, global_model, num_clients)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--checkpoint-path", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_simulation(args.num_clients, args.num_rounds, args.checkpoint_path)
