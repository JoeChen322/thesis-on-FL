import math
import os
import time
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
    client_num_threads,
    configure_thread_env,
    configure_torch_threads,
)
from noniid_jsd_switch import (
    DEFAULT_IID_JSD_THRESHOLD,
    DEFAULT_STRONG_NONIID_JSD_THRESHOLD,
    classify_noniid_condition,
    validate_jsd_thresholds,
)


def log_client_trace(
    phase,
    event,
    client_id,
    batch_index,
    num_threads,
    start_perf=None,
):
    now_perf = time.perf_counter()
    fields = [
        "CLIENT_TRACE",
        f"phase={phase}",
        f"event={event}",
        f"wall_ts={time.time():.6f}",
        f"perf_ts={now_perf:.6f}",
        f"pid={os.getpid()}",
        f"client_id={client_id}",
        f"batch_index={batch_index}",
        f"torch_threads={torch.get_num_threads()}",
        f"configured_threads={num_threads}",
    ]
    if start_perf is not None:
        fields.append(f"duration_s={now_perf - start_perf:.6f}")
    print(" ".join(fields), flush=True)
    return now_perf


class RuntimeStats:
    def __init__(self):
        self.total_start = time.perf_counter()
        self.communication_time = 0.0
        self.training_time = 0.0
        self.delay_time = 0.0

    def add_communication(self, duration):
        self.communication_time += duration

    def add_training(self, duration):
        self.training_time += duration

    def add_delay(self, duration):
        self.delay_time += duration
        self.communication_time += duration

    def total_runtime(self):
        return time.perf_counter() - self.total_start + self.delay_time

    def print_summary(self, label):
        print(
            f"{label} runtime stats: "
            f"total_runtime={self.total_runtime():.4f}s "
            f"communication_time={self.communication_time:.4f}s "
            f"delay_time={self.delay_time:.4f}s "
            f"training_time={self.training_time:.4f}s",
            flush=True,
        )


def parse_communication_delays(value):
    if isinstance(value, (int, float)):
        delays = [float(value)]
    else:
        delays = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not delays:
        delays = [0.0]
    if any(delay < 0 for delay in delays):
        raise ValueError("communication_delay must contain non-negative seconds")
    return delays


def communication_delay_for_round(delays, round_index):
    if round_index < len(delays):
        return delays[round_index]
    return delays[-1]


class PatternTogglingManager:
    def __init__(
        self,
        initial_use_client_fedavg,
        iid_threshold=DEFAULT_IID_JSD_THRESHOLD,
        strong_threshold=DEFAULT_STRONG_NONIID_JSD_THRESHOLD,
    ):
        validate_jsd_thresholds(iid_threshold, strong_threshold)
        self.use_client_fedavg = bool(initial_use_client_fedavg)
        self.iid_threshold = float(iid_threshold)
        self.strong_threshold = float(strong_threshold)
        self.condition = "unknown"
        self.mean_boundary_score = 0.0
        self.max_boundary_score = 0.0

    def consume_boundary_scores(self, scores_by_client):
        scores = [float(score) for _, score in sorted(scores_by_client.items())]
        if not scores:
            return self.use_client_fedavg

        self.mean_boundary_score = sum(scores) / len(scores)
        self.max_boundary_score = max(scores)
        self.condition = classify_noniid_condition(
            self.mean_boundary_score,
            iid_threshold=self.iid_threshold,
            strong_threshold=self.strong_threshold,
        )
        if self.condition == "strong_noniid":
            self.use_client_fedavg = True
        return self.use_client_fedavg

    def pattern_name(self):
        return "SFL" if self.use_client_fedavg else "SL"


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
    num_threads: int,
    boundary_condition_fn: Callable | None = None,
):
    app = ClientApp()

    @app.train("forward")
    def forward(message, context):
        configure_torch_threads(num_threads)
        config = message.content["config"]#get the content
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        client_id = int(config["client_id"])
        num_clients = int(config["num_clients"])
        batch_index = int(config["batch_index"])
        batch_size = int(config["batch_size"])
        start_perf = log_client_trace(
            "forward",
            "START",
            client_id,
            batch_index,
            num_threads,
        )

        client_model = load_model_from_state(
            context, "client_model", client_model_cls, device
        )
        client_model.train()
        x, y = get_batch_fn(client_id, num_clients, batch_index, batch_size, device)

        with torch.no_grad():
            activation = client_model(x)

        duration_s = time.perf_counter() - start_perf
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
                "duration_s": duration_s,
            }),
        })
        log_client_trace(
            "forward",
            "END",
            client_id,
            batch_index,
            num_threads,
            start_perf=start_perf,
        )
        return Message(content, reply_to=message)

    @app.train("backward")
    def backward(message, context):
        configure_torch_threads(num_threads)
        config = message.content["config"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        client_id = int(config["client_id"])
        num_clients = int(config["num_clients"])
        batch_index = int(config["batch_index"])
        batch_size = int(config["batch_size"])
        lr_client = float(config["lr_client"])
        start_perf = log_client_trace(
            "backward",
            "START",
            client_id,
            batch_index,
            num_threads,
        )

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

        duration_s = time.perf_counter() - start_perf
        log_client_trace(
            "backward",
            "END",
            client_id,
            batch_index,
            num_threads,
            start_perf=start_perf,
        )
        return Message(RecordDict({
            "metrics": MetricRecord({
                "client_id": client_id,
                "batch_index": batch_index,
                "updated": 1,
                "duration_s": duration_s,
            })
        }), reply_to=message)

    @app.train("get_params")
    def get_params(message, context):
        configure_torch_threads(num_threads)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        client_model = load_model_from_state(
            context, "client_model", client_model_cls, device
        )
        return Message(RecordDict({
            "client_model": model_to_record(client_model)
        }), reply_to=message)

    @app.train("set_params")
    def set_params(message, context):
        configure_torch_threads(num_threads)
        context.state["client_model"] = message.content["client_model"]
        return Message(RecordDict({
            "metrics": MetricRecord({"updated": 1})
        }), reply_to=message)

    @app.train("boundary_condition")
    def boundary_condition(message, context):
        configure_torch_threads(num_threads)
        if boundary_condition_fn is None:
            raise RuntimeError("boundary_condition_fn is not configured")

        config = message.content["config"]
        client_id = int(config["client_id"])
        num_clients = int(config["num_clients"])
        boundary_score = float(boundary_condition_fn(client_id, num_clients))
        return Message(RecordDict({
            "metrics": MetricRecord({
                "client_id": client_id,
                "boundary_score": boundary_score,
            })
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
    boundary_switch_enabled: bool,
    iid_jsd_threshold: float,
    strong_noniid_jsd_threshold: float,
    communication_delays,
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
        pattern_manager = PatternTogglingManager(
            initial_use_client_fedavg=use_client_fedavg,
            iid_threshold=iid_jsd_threshold,
            strong_threshold=strong_noniid_jsd_threshold,
        )
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
        stats = RuntimeStats()
        round_stats = None

        def send_and_receive_timed(msgs, stat_label=None):
            start_perf = time.perf_counter()
            replies = list(grid.send_and_receive(msgs))
            elapsed = time.perf_counter() - start_perf
            stats.add_communication(elapsed)
            if round_stats is not None:
                round_stats.add_communication(elapsed)
            if stat_label is not None:
                print(
                    f"TIMING {stat_label} communication_time={elapsed:.4f}s",
                    flush=True,
                )
            return replies

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
            param_replies = send_and_receive_timed(
                msgs,
                stat_label=f"{group_id}",
            )
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
            send_and_receive_timed(
                msgs,
                stat_label=f"{group_id}",
            )

        def collect_boundary_conditions(group_id):
            msgs = [
                grid.create_message(
                    RecordDict({
                        "config": ConfigRecord({
                            "client_id": client_id,
                            "num_clients": num_clients,
                        })
                    }),
                    message_type="train.boundary_condition",
                    dst_node_id=node_id,
                    group_id=group_id,
                )
                for client_id, node_id in enumerate(node_ids)
            ]
            replies = send_and_receive_timed(
                msgs,
                stat_label=f"{group_id}",
            )
            return {
                int(reply.content["metrics"]["client_id"]): float(
                    reply.content["metrics"]["boundary_score"]
                )
                for reply in replies
            }

        def evaluate_current_models(label):
            if evaluate_fn is None:
                return

            group_label = label.lower().replace(" ", "-")
            client_states = get_client_states(f"{group_label}-eval-client-params")
            states_to_evaluate = (
                client_states[:1]
                if pattern_manager.use_client_fedavg
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
        if boundary_switch_enabled:
            scores_by_client = collect_boundary_conditions("boundary-condition")
            pattern_manager.consume_boundary_scores(scores_by_client)
            score_text = ", ".join(
                f"{client_id}:{score:.4f}"
                for client_id, score in sorted(scores_by_client.items())
            )
            print(
                "Pattern Toggling Manager boundary scores: "
                f"{score_text}",
                flush=True,
            )
            print(
                "Pattern Toggling Manager decision: "
                f"condition={pattern_manager.condition} "
                f"mean={pattern_manager.mean_boundary_score:.4f} "
                f"max={pattern_manager.max_boundary_score:.4f} "
                f"pattern={pattern_manager.pattern_name()}",
                flush=True,
            )

        for round_idx in range(1, num_rounds + 1):
            round_stats = RuntimeStats()
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

                    forward_replies = send_and_receive_timed(
                        forward_msgs,
                        stat_label=f"round={round_idx} batch={batch_index} forward",
                    )
                    forward_replies_by_client = {}
                    forward_training_times = []
                    for reply in forward_replies:
                        metrics = reply.content["metrics"]
                        forward_replies_by_client[int(metrics["client_id"])] = reply
                        if "duration_s" in metrics:
                            forward_training_times.append(float(metrics["duration_s"]))
                    if forward_training_times:
                        forward_training_time = max(forward_training_times)
                        stats.add_training(forward_training_time)
                        round_stats.add_training(forward_training_time)

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

                        train_start_perf = time.perf_counter()
                        server_optimizer.zero_grad()
                        outputs = server_model(activation)
                        #compute the loss
                        loss = criterion(outputs, labels)
                        loss.backward()
                        server_optimizer.step()
                        train_elapsed = time.perf_counter() - train_start_perf
                        stats.add_training(train_elapsed)
                        round_stats.add_training(train_elapsed)

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

                    backward_replies = send_and_receive_timed(
                        backward_msgs,
                        stat_label=f"round={round_idx} batch={batch_index} backward",
                    )
                    backward_training_times = []
                    for reply in backward_replies:
                        metrics = reply.content["metrics"]
                        if "duration_s" in metrics:
                            backward_training_times.append(float(metrics["duration_s"]))
                    if backward_training_times:
                        backward_training_time = max(backward_training_times)
                        stats.add_training(backward_training_time)
                        round_stats.add_training(backward_training_time)

            for client_id, num_batches in enumerate(num_batches_by_client):
                print(f"Client {client_id} -> server: finished {num_batches} batches")

            if pattern_manager.use_client_fedavg:
                client_states = get_client_states(f"{round_idx}-get-client-params")
                fedavg_start_perf = time.perf_counter()
                #get the average value
                avg_state = fedavg_fn(client_states, sizes)
                #switch the dict form into message form
                avg_record = ArrayRecord.from_torch_state_dict(avg_state)
                fedavg_elapsed = time.perf_counter() - fedavg_start_perf
                stats.add_training(fedavg_elapsed)
                round_stats.add_training(fedavg_elapsed)

                set_client_states(avg_record, f"{round_idx}-set-client-params")
                print("SFL client-side FedAvg: completed")

            round_delay = communication_delay_for_round(
                communication_delays,
                round_idx - 1,
            )
            if round_delay > 0:
                stats.add_delay(round_delay)
                round_stats.add_delay(round_delay)

            print("--------------------------------")
            print(f"Round {round_idx} summary:")
            print(f"Average train loss: {total_loss / total_examples:.4f}")
            print(f"Average train acc:  {total_correct / total_examples * 100:.2f}%")
            round_stats.print_summary(f"Round {round_idx}")
            if eval_every_round:
                evaluate_current_models(f"Round {round_idx}")

        print("\nTraining finished.")
        evaluate_current_models("Final")
        if on_finished_fn is not None:
            client_states = get_client_states("save-client-params")
            on_finished_fn(client_states, server_model)
        stats.print_summary("Total")

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
    boundary_condition_fn: Callable | None = None,
    boundary_switch_enabled: bool = False,
    iid_jsd_threshold: float = DEFAULT_IID_JSD_THRESHOLD,
    strong_noniid_jsd_threshold: float = DEFAULT_STRONG_NONIID_JSD_THRESHOLD,
    communication_delay=0.0,
):
    communication_delays = parse_communication_delays(communication_delay)
    num_threads = client_num_threads(num_cpus)
    configure_thread_env(num_threads)
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
        boundary_switch_enabled=boundary_switch_enabled,
        iid_jsd_threshold=iid_jsd_threshold,
        strong_noniid_jsd_threshold=strong_noniid_jsd_threshold,
        communication_delays=communication_delays,
    )
    client_app = make_client_app(
        client_model_cls,
        get_batch_fn,
        num_threads,
        boundary_condition_fn=boundary_condition_fn,
    )
    backend_config = build_ray_backend_config(total_num_cpus, num_cpus, num_gpus)
    print(f"Client PyTorch threads per actor: {num_threads}", flush=True)
    print("Starting Flower Message API simulation...", flush=True)
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=num_clients,
        backend_config=backend_config,
        verbose_logging=True,
    )
