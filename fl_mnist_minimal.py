"""Test for FL by using library PyTorch-flower
the result is the loss getting decrease after several epochs
AND the accurancy for each client improved"""

import argparse
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn.functional as F
from flwr.clientapp import ClientApp
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerAppComponents, ServerConfig
from flwr.serverapp import ServerApp
from flwr.simulation import run_simulation

from mnist_evaluation import evaluate_model, print_test_metrics
from split_learning_utils import (
    FullNet,
    check_simulation_backend,
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
    def __init__(self, client_id, num_clients, device, noniid_alpha):
        self.client_id = client_id
        self.device = device

        self.model = FullNet().to(device)
        self.trainloader, self.testloader = load_data(
            client_id,
            num_clients,
            noniid_alpha=noniid_alpha,
        )

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)

        train(
            model=self.model,
            trainloader=self.trainloader,
            epochs=int(config.get("local_epochs", 1)),
            device=self.device,
        )

        return (
            get_parameters(self.model),
            len(self.trainloader.dataset),
            {},
        )

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)

        loss, accuracy = evaluate_model(
            model=self.model,
            testloader=self.testloader,
            device=self.device,
        )

        return (
            float(loss),
            len(self.testloader.dataset),
            {"accuracy": float(accuracy), "client_id": self.client_id},
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


def load_checkpoint(checkpoint_path, model, num_clients, device, noniid_alpha):
    client_states, server_state = load_split_checkpoint(
        checkpoint_path,
        num_clients,
        device,
        noniid_alpha,
    )
    if client_states is None:
        return

    sizes = [
        client_size(num_clients, client_id, noniid_alpha)
        for client_id in range(num_clients)
    ]
    model.client_model.load_state_dict(fedavg_state_dicts(client_states, sizes))
    model.server_model.load_state_dict(server_state)


def save_checkpoint(checkpoint_path, model, num_clients, noniid_alpha):
    client_states = [
        model.client_model.state_dict()
        for _ in range(num_clients)
    ]
    save_split_checkpoint(
        checkpoint_path,
        client_states,
        model.server_model,
        num_clients,
        noniid_alpha,
    )


class ReportingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_parameters = kwargs.get("initial_parameters")

    def aggregate_fit(self, server_round, results, failures):
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if parameters is not None:
            self.current_parameters = parameters
        return parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        print(f"\n========== Round {server_round} ==========")
        for _, evaluate_res in sorted(
            results,
            key=lambda item: int(item[1].metrics.get("client_id", 0)),
        ):
            client_id = int(evaluate_res.metrics.get("client_id", -1))
            accuracy = float(evaluate_res.metrics["accuracy"])
            print(
                f"Client {client_id}: "
                f"test loss = {evaluate_res.loss:.4f}, "
                f"test acc = {accuracy * 100:.2f}%"
            )

        loss, metrics = super().aggregate_evaluate(
            server_round,
            results,
            failures,
        )
        if loss is not None and "accuracy" in metrics:
            print("--------------------------------")
            print(f"Round {server_round} summary:")
            print_test_metrics("Average", loss, float(metrics["accuracy"]))
        return loss, metrics


def make_client_app(num_clients, noniid_alpha):
    def client_fn(context):
        client_id = int(context.node_config["partition-id"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return FlowerClient(client_id, num_clients, device, noniid_alpha).to_client()

    return ClientApp(client_fn=client_fn)


def make_server_app(num_rounds, num_clients, strategy):
    def server_fn(context):
        return ServerAppComponents(
            config=ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )

    return ServerApp(server_fn=server_fn)


def start_simulation(
    num_clients,
    num_rounds,
    checkpoint_path="",
    local_epochs=1,
    client_num_cpus=1.0,
    client_num_gpus=0.0,
    noniid_alpha=1.0,
):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = FullNet().to(device)
    load_checkpoint(checkpoint_path, global_model, num_clients, device, noniid_alpha)
    initial_parameters = ndarrays_to_parameters(get_parameters(global_model))
    strategy = ReportingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=lambda _: {"local_epochs": local_epochs},
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    print(f"Device: {device}")
    print(f"Number of clients: {num_clients}")
    print(f"Non-IID alpha: {noniid_alpha}")
    print("Start Flower FL simulation")

    total_num_cpus = check_simulation_backend(num_clients, client_num_cpus)
    backend_config = {
        "init_args": {
            "num_cpus": total_num_cpus,
            "include_dashboard": False,
        },
        "client_resources": {
            "num_cpus": client_num_cpus,
            "num_gpus": client_num_gpus,
        }
    }
    run_simulation(
        server_app=make_server_app(num_rounds, num_clients, strategy),
        client_app=make_client_app(num_clients, noniid_alpha),
        num_supernodes=num_clients,
        backend_config=backend_config,
        verbose_logging=True,
    )

    print("\nTraining finished.")
    if strategy.current_parameters is not None:
        set_parameters(global_model, parameters_to_ndarrays(strategy.current_parameters))
    loss, accuracy = evaluate_model(global_model, device)
    print_test_metrics("Final", loss, accuracy)
    save_checkpoint(checkpoint_path, global_model, num_clients, noniid_alpha)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--client-num-cpus", type=float, default=1.0)
    parser.add_argument("--client-num-gpus", type=float, default=0.0)
    parser.add_argument(
        "--noniid-alpha",
        type=float,
        default=1.0,
        help="Shared Dirichlet non-IID degree in [0, 1]. 1.0 keeps IID splitting.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.num_clients < 1:
        raise ValueError("--num-clients must be at least 1")
    start_simulation(
        args.num_clients,
        args.num_rounds,
        args.checkpoint_path,
        args.local_epochs,
        args.client_num_cpus,
        args.client_num_gpus,
        args.noniid_alpha,
    )
