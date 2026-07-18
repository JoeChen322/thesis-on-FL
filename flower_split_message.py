import math
from typing import Type

import torch
from flwr.app.message import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.clientapp import ClientApp
from flwr.serverapp import ServerApp
from flwr.simulation import run_simulation
from torch import nn

from split_learning_utils import (
    client_size,
    fedavg_state_dicts,
    get_batch,
    set_seed,
)


def model_to_record(model):
    return ArrayRecord.from_torch_state_dict(model.state_dict())


def record_to_state_dict(record):
    return record.to_torch_state_dict()


def load_model_from_state(context, state_key, model_cls, device):
    model = model_cls().to(device)#initial a instance by random parameters weights
    #has this model trained before
    if state_key in context.state:
        model.load_state_dict(record_to_state_dict(context.state[state_key]))
    else:
        context.state[state_key] = model_to_record(model)
    return model

#parameter input， model instance output
def make_client_app(client_model_cls: Type[nn.Module]):
    app = ClientApp()

    @app.train("forward")
    def forward(message, context):
        config = message.content["config"]#get the content
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        client_id = int(config["client_id"])
        num_clients = int(config["num_clients"])
        batch_index = int(config["batch_index"])
        batch_size = int(config["batch_size"])

        client_model = load_model_from_state(
            context, "client_model", client_model_cls, device
        )
        client_model.train()
        x, y = get_batch(client_id, num_clients, batch_index, batch_size, device)

        with torch.no_grad():
            activation = client_model(x)

        content = RecordDict({
            "activation": ArrayRecord.from_numpy_ndarrays([
                activation.detach().cpu().numpy()
            ]),
            "labels": ArrayRecord.from_numpy_ndarrays([
                y.detach().cpu().numpy()
            ]),
            "metrics": MetricRecord({"num_examples": int(y.size(0))}),
        })
        return Message(content, reply_to=message)

    @app.train("backward")
    def backward(message, context):
        config = message.content["config"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        client_id = int(config["client_id"])
        num_clients = int(config["num_clients"])
        batch_index = int(config["batch_index"])
        batch_size = int(config["batch_size"])
        lr_client = float(config["lr_client"])

        client_model = load_model_from_state(
            context, "client_model", client_model_cls, device
        )
        client_model.train()
        optimizer = torch.optim.SGD(client_model.parameters(), lr=lr_client)
        x, _ = get_batch(client_id, num_clients, batch_index, batch_size, device)
        grad = torch.tensor(
            message.content["gradient"].to_numpy_ndarrays()[0],
            device=device,
        )

        optimizer.zero_grad()
        activation = client_model(x)
        activation.backward(grad)
        optimizer.step()
        context.state["client_model"] = model_to_record(client_model)

        return Message(RecordDict({
            "metrics": MetricRecord({"updated": 1})
        }), reply_to=message)

    @app.train("get_params")
    def get_params(message, context):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        client_model = load_model_from_state(
            context, "client_model", client_model_cls, device
        )
        return Message(RecordDict({
            "client_model": model_to_record(client_model)
        }), reply_to=message)

    @app.train("set_params")
    def set_params(message, context):
        context.state["client_model"] = message.content["client_model"]
        return Message(RecordDict({
            "metrics": MetricRecord({"updated": 1})
        }), reply_to=message)

    return app


def make_server_app(
    client_model_cls: Type[nn.Module],
    server_model_cls: Type[nn.Module],
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    lr_client: float,
    lr_server: float,
    use_client_fedavg: bool,
    max_batches: int | None,
):
    app = ServerApp()

    @app.main()
    def main(grid, context):
        set_seed()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        server_model = server_model_cls().to(device)
        server_optimizer = torch.optim.SGD(server_model.parameters(), lr=lr_server)#Stochastic Gradient Descent
        criterion = nn.CrossEntropyLoss()
        #which nodes are online，polling all clients
        node_ids = list(grid.get_node_ids())
        if len(node_ids) < num_clients:
            raise RuntimeError(
                f"Expected {num_clients} clients, but only {len(node_ids)} are available"
            )
        node_ids = node_ids[:num_clients]
        sizes = [client_size(num_clients, cid) for cid in range(num_clients)]

        print(f"Device: {device}")
        print(f"Number of clients: {num_clients}")
        print(f"Client data sizes: {sizes}")
        print("Start Flower Message API Split Learning simulation")

        for round_idx in range(1, num_rounds + 1):
            total_loss = 0.0
            total_correct = 0
            total_examples = 0
            print(f"\n========== Round {round_idx} ==========")

            for client_id, node_id in enumerate(node_ids):
                num_batches = math.ceil(sizes[client_id] / batch_size)
                if max_batches is not None:
                    num_batches = min(num_batches, max_batches)
                for _ in range(local_epochs):
                    for batch_index in range(num_batches):
                        config = ConfigRecord({
                            "client_id": client_id,
                            "num_clients": num_clients,
                            "batch_index": batch_index,
                            "batch_size": batch_size,
                        })
                        forward_msg = grid.create_message(
                            RecordDict({"config": config}),
                            message_type="train.forward",
                            dst_node_id=node_id,
                            group_id=f"{round_idx}-forward",
                        )
                        forward_reply = list(
                            grid.send_and_receive([forward_msg])
                        )[0]
                        #Sequentially processes clients, so only use[0]
                        activation = torch.tensor(
                            forward_reply.content["activation"].to_numpy_ndarrays()[0],
                            device=device,
                            requires_grad=True,
                        )
                        labels = torch.tensor(
                            forward_reply.content["labels"].to_numpy_ndarrays()[0],
                            device=device,
                            dtype=torch.long,
                        )

                        server_optimizer.zero_grad()
                        outputs = server_model(activation)
                        #compute the loss
                        loss = criterion(outputs, labels)
                        loss.backward()
                        server_optimizer.step()

                        grad = activation.grad.detach().cpu().numpy()
                        backward_config = ConfigRecord({
                            "client_id": client_id,
                            "num_clients": num_clients,
                            "batch_index": batch_index,
                            "batch_size": batch_size,
                            "lr_client": lr_client,
                        })
                        backward_msg = grid.create_message(
                            RecordDict({
                                "config": backward_config,
                                "gradient": ArrayRecord.from_numpy_ndarrays([grad]),
                            }),
                            message_type="train.backward",
                            dst_node_id=node_id,
                            group_id=f"{round_idx}-backward",
                        )
                        list(grid.send_and_receive([backward_msg]))

                        num_examples = labels.size(0)
                        total_loss += loss.item() * num_examples
                        total_correct += (
                            outputs.argmax(dim=1) == labels
                        ).sum().item()
                        total_examples += num_examples

                print(f"Client {client_id} -> server: finished {num_batches} batches")

            if use_client_fedavg:
                param_replies = []
                for node_id in node_ids:
                    msg = grid.create_message(
                        RecordDict({}),
                        message_type="train.get_params",
                        dst_node_id=node_id,
                        group_id=f"{round_idx}-get-client-params",
                    )
                    param_replies.append(list(grid.send_and_receive([msg]))[0])

                client_states = [
                    record_to_state_dict(reply.content["client_model"])
                    for reply in param_replies
                ]
                #get the average value
                avg_state = fedavg_state_dicts(client_states, sizes)
                #switch the dict form into message form
                avg_record = ArrayRecord.from_torch_state_dict(avg_state)

                for node_id in node_ids:
                    msg = grid.create_message(
                        RecordDict({"client_model": avg_record}),
                        message_type="train.set_params",
                        dst_node_id=node_id,
                        group_id=f"{round_idx}-set-client-params",
                    )
                    list(grid.send_and_receive([msg]))

                print("SFL client-side FedAvg: completed")

            print("--------------------------------")
            print(f"Round {round_idx} summary:")
            print(f"Average train loss: {total_loss / total_examples:.4f}")
            print(f"Average train acc:  {total_correct / total_examples * 100:.2f}%")

        print("\nTraining finished.")

    return app


def run_message_simulation(
    client_model_cls: Type[nn.Module],
    server_model_cls: Type[nn.Module],
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    lr_client: float,
    lr_server: float,
    use_client_fedavg: bool,
    num_cpus: float,
    num_gpus: float,
    max_batches: int | None = None,
):
    try:
        import ray  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Flower message-simulation requires the Ray backend, but `ray` is "
            "not installed for this Python environment. On Windows, Ray is not "
            "available for all Python versions; use a Python version with a Ray "
            "wheel, or run this in WSL2."
        ) from exc

    server_app = make_server_app(
        client_model_cls=client_model_cls,
        server_model_cls=server_model_cls,
        num_clients=num_clients,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        lr_client=lr_client,
        lr_server=lr_server,
        use_client_fedavg=use_client_fedavg,
        max_batches=max_batches,
    )
    client_app = make_client_app(client_model_cls)
    backend_config = {
        "client_resources": {
            "num_cpus": num_cpus,
            "num_gpus": num_gpus,
        }
    }
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=num_clients,
        backend_config=backend_config,
    )
