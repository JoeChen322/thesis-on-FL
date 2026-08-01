import math
from typing import Callable, Type

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
    build_ray_backend_config,
    check_simulation_backend,
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
def make_client_app(
    client_model_cls: Type[nn.Module],
    get_batch_fn: Callable,
):
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
        x, y = get_batch_fn(client_id, num_clients, batch_index, batch_size, device)

        with torch.no_grad():
            activation = client_model(x)

        content = RecordDict({
            "activation": ArrayRecord.from_numpy_ndarrays([
                activation.detach().cpu().numpy()
            ]),
            "labels": ArrayRecord.from_numpy_ndarrays([
                y.detach().cpu().numpy()
            ]),
            "metrics": MetricRecord({
                "client_id": client_id,
                "batch_index": batch_index,
                "num_examples": int(y.size(0)),
            }),
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
        x, _ = get_batch_fn(client_id, num_clients, batch_index, batch_size, device)
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
            "metrics": MetricRecord({
                "client_id": client_id,
                "batch_index": batch_index,
                "updated": 1,
            })
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
    set_seed_fn: Callable,
    client_size_fn: Callable,
    fedavg_fn: Callable,
    initial_client_states: list | None,
    initial_server_state: dict | None,
    on_finished_fn: Callable | None,
    evaluate_fn: Callable | None,
    print_metrics_fn: Callable | None,
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    lr_client: float,
    lr_server: float,
    use_client_fedavg: bool,
    max_batches: int | None,
    eval_every_round: bool,
):
    app = ServerApp()

    @app.main()
    def main(grid, context):
        set_seed_fn()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        server_model = server_model_cls().to(device)
        if initial_server_state is not None:
            server_model.load_state_dict(initial_server_state)
        server_optimizer = torch.optim.SGD(server_model.parameters(), lr=lr_server)#Stochastic Gradient Descent
        criterion = nn.CrossEntropyLoss()
        #which nodes are online，polling all clients
        node_ids = list(grid.get_node_ids())
        if len(node_ids) < num_clients:
            raise RuntimeError(
                f"Expected {num_clients} clients, but only {len(node_ids)} are available"
            )
        node_ids = node_ids[:num_clients]
        sizes = [client_size_fn(num_clients, cid) for cid in range(num_clients)]
        num_batches_by_client = [
            math.ceil(size / batch_size)
            for size in sizes
        ]
        if max_batches is not None:
            num_batches_by_client = [
                min(num_batches, max_batches)
                for num_batches in num_batches_by_client
            ]

        def get_client_states(group_id):
            msgs = [
                grid.create_message(
                    RecordDict({}),
                    message_type="train.get_params",
                    dst_node_id=node_id,
                    group_id=group_id,
                )
                for node_id in node_ids
            ]
            param_replies = list(grid.send_and_receive(msgs))
            return [
                record_to_state_dict(reply.content["client_model"])
                for reply in param_replies
            ]

        def set_client_states(record, group_id):
            msgs = [
                grid.create_message(
                    RecordDict({"client_model": record}),
                    message_type="train.set_params",
                    dst_node_id=node_id,
                    group_id=group_id,
                )
                for node_id in node_ids
            ]
            list(grid.send_and_receive(msgs))

        def evaluate_current_models(label):
            if evaluate_fn is None:
                return

            group_label = label.lower().replace(" ", "-")
            client_states = get_client_states(f"{group_label}-eval-client-params")
            states_to_evaluate = (
                client_states[:1]
                if use_client_fedavg
                else client_states
            )
            losses = []
            accuracies = []
            for client_id, client_state in enumerate(states_to_evaluate):
                client_model = client_model_cls().to(device)
                client_model.load_state_dict(client_state)
                loss, accuracy = evaluate_fn(client_model, server_model, device)
                losses.append(loss)
                accuracies.append(accuracy)
                if len(states_to_evaluate) > 1:
                    if print_metrics_fn is None:
                        print(f"{label} client {client_id} test loss: {loss:.4f}")
                        print(f"{label} client {client_id} test acc:  {accuracy * 100:.2f}%")
                    else:
                        print_metrics_fn(f"{label} client {client_id}", loss, accuracy)

            avg_loss = sum(losses) / len(losses)
            avg_accuracy = sum(accuracies) / len(accuracies)
            metric_label = (
                label
                if len(states_to_evaluate) == 1
                else f"{label} average"
            )
            if print_metrics_fn is None:
                print(f"{metric_label} test loss: {avg_loss:.4f}")
                print(f"{metric_label} test acc:  {avg_accuracy * 100:.2f}%")
            else:
                print_metrics_fn(metric_label, avg_loss, avg_accuracy)

        if initial_client_states is not None:
            load_msgs = [
                grid.create_message(
                    RecordDict({
                        "client_model": ArrayRecord.from_torch_state_dict(client_state)
                    }),
                    message_type="train.set_params",
                    dst_node_id=node_id,
                    group_id="load-client-params",
                )
                for node_id, client_state in zip(node_ids, initial_client_states)
            ]
            list(grid.send_and_receive(load_msgs))

        print(f"Device: {device}")
        print(f"Number of clients: {num_clients}")
        print(f"Client data sizes: {sizes}")
        print("Start Flower Message API Split Learning simulation")

        for round_idx in range(1, num_rounds + 1):
            total_loss = 0.0
            total_correct = 0
            total_examples = 0
            print(f"\n========== Round {round_idx} ==========")

            for _ in range(local_epochs):
                max_round_batches = max(num_batches_by_client)
                for batch_index in range(max_round_batches):
                    forward_msgs = []
                    active_clients = []
                    for client_id, node_id in enumerate(node_ids):
                        if batch_index >= num_batches_by_client[client_id]:
                            continue

                        config = ConfigRecord({
                            "client_id": client_id,
                            "num_clients": num_clients,
                            "batch_index": batch_index,
                            "batch_size": batch_size,
                        })
                        forward_msgs.append(
                            grid.create_message(
                                RecordDict({"config": config}),
                                message_type="train.forward",
                                dst_node_id=node_id,
                                group_id=f"{round_idx}-forward-{batch_index}",
                            )
                        )
                        active_clients.append((client_id, node_id))

                    if not forward_msgs:
                        continue

                    forward_replies_by_client = {
                        int(reply.content["metrics"]["client_id"]): reply
                        for reply in grid.send_and_receive(forward_msgs)
                    }

                    backward_msgs = []
                    for client_id, node_id in active_clients:
                        forward_reply = forward_replies_by_client[client_id]
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
                        backward_msgs.append(
                            grid.create_message(
                                RecordDict({
                                    "config": backward_config,
                                    "gradient": ArrayRecord.from_numpy_ndarrays([grad]),
                                }),
                                message_type="train.backward",
                                dst_node_id=node_id,
                                group_id=f"{round_idx}-backward-{batch_index}",
                            )
                        )

                        num_examples = labels.size(0)
                        total_loss += loss.item() * num_examples
                        total_correct += (
                            outputs.argmax(dim=1) == labels
                        ).sum().item()
                        total_examples += num_examples

                    list(grid.send_and_receive(backward_msgs))

            for client_id, num_batches in enumerate(num_batches_by_client):
                print(f"Client {client_id} -> server: finished {num_batches} batches")

            if use_client_fedavg:
                client_states = get_client_states(f"{round_idx}-get-client-params")
                #get the average value
                avg_state = fedavg_fn(client_states, sizes)
                #switch the dict form into message form
                avg_record = ArrayRecord.from_torch_state_dict(avg_state)

                set_client_states(avg_record, f"{round_idx}-set-client-params")
                print("SFL client-side FedAvg: completed")

            print("--------------------------------")
            print(f"Round {round_idx} summary:")
            print(f"Average train loss: {total_loss / total_examples:.4f}")
            print(f"Average train acc:  {total_correct / total_examples * 100:.2f}%")
            if eval_every_round:
                evaluate_current_models(f"Round {round_idx}")

        print("\nTraining finished.")
        evaluate_current_models("Final")
        if on_finished_fn is not None:
            client_states = get_client_states("save-client-params")
            on_finished_fn(client_states, server_model)

    return app


def run_message_simulation(
    client_model_cls: Type[nn.Module],
    server_model_cls: Type[nn.Module],
    set_seed_fn: Callable,
    client_size_fn: Callable,
    get_batch_fn: Callable,
    fedavg_fn: Callable,
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    lr_client: float,
    lr_server: float,
    use_client_fedavg: bool,
    num_cpus: float,
    num_gpus: float,
    initial_client_states: list | None = None,
    initial_server_state: dict | None = None,
    on_finished_fn: Callable | None = None,
    evaluate_fn: Callable | None = None,
    print_metrics_fn: Callable | None = None,
    max_batches: int | None = None,
    eval_every_round: bool = False,
):
    total_num_cpus = check_simulation_backend(
        num_clients,
        num_cpus,
        simulation_name="Flower message simulation",
    )

    server_app = make_server_app(
        client_model_cls=client_model_cls,
        server_model_cls=server_model_cls,
        set_seed_fn=set_seed_fn,
        client_size_fn=client_size_fn,
        fedavg_fn=fedavg_fn,
        initial_client_states=initial_client_states,
        initial_server_state=initial_server_state,
        on_finished_fn=on_finished_fn,
        evaluate_fn=evaluate_fn,
        print_metrics_fn=print_metrics_fn,
        num_clients=num_clients,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        lr_client=lr_client,
        lr_server=lr_server,
        use_client_fedavg=use_client_fedavg,
        max_batches=max_batches,
        eval_every_round=eval_every_round,
    )
    client_app = make_client_app(client_model_cls, get_batch_fn)
    backend_config = build_ray_backend_config(total_num_cpus, num_cpus, num_gpus)
    print("Starting Flower Message API simulation...", flush=True)
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=num_clients,
        backend_config=backend_config,
        verbose_logging=True,
    )
