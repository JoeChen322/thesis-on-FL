import argparse
import time
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mnist_evaluation import evaluate_model, print_test_metrics
from split_learning_utils import (
    DATASET_CHOICES,
    FullNet,
    add_resnet_model_args,
    client_num_threads,
    client_subset,
    configure_torch_threads,
    get_model_classes,
    get_resnet_model_config_from_args,
    load_split_checkpoint,
    save_split_checkpoint,
    set_seed,
)


class RuntimeStats:
    def __init__(self):
        self.total_start = time.perf_counter()
        self.training_time = 0.0

    def add_training(self, duration):
        self.training_time += duration

    def total_runtime(self):
        return time.perf_counter() - self.total_start

    def print_summary(self, label):
        print(
            f"{label} runtime stats: "
            f"total_runtime={self.total_runtime():.4f}s "
            "communication_time=0.0000s "
            "delay_time=0.0000s "
            f"training_time={self.training_time:.4f}s",
            flush=True,
        )


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def train(model, trainloader, epochs, device, max_batches=None):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for _ in range(epochs):
        for batch_index, (x, y) in enumerate(trainloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()


def load_checkpoint(
    checkpoint_path,
    model,
    device,
    noniid_alpha,
    dataset_name,
    model_config,
):
    client_states, server_state = load_split_checkpoint(
        checkpoint_path,
        num_clients=1,
        device=device,
        noniid_alpha=noniid_alpha,
        dataset_name=dataset_name,
        model_config=model_config,
    )
    if client_states is None:
        return
    model.client_model.load_state_dict(client_states[0])
    model.server_model.load_state_dict(server_state)


def save_checkpoint(
    checkpoint_path,
    model,
    noniid_alpha,
    dataset_name,
    model_config,
):
    save_split_checkpoint(
        checkpoint_path,
        [model.client_model.state_dict()],
        model.server_model,
        num_clients=1,
        noniid_alpha=noniid_alpha,
        dataset_name=dataset_name,
        model_config=model_config,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Local single-client ResNet baseline using the shared split checkpoint format."
    )
    parser.add_argument("--num-clients", type=int, default=1)
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--client-num-cpus", type=float, default=1.0)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="mnist")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--eval-every-round",
        action="store_true",
        help="Evaluate the full test set after every round.",
    )
    parser.add_argument(
        "--noniid-alpha",
        type=float,
        default=1.0,
        help="Saved in checkpoint metadata for compatibility; one client receives all data.",
    )
    add_resnet_model_args(parser, default_split_after="layer2")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_clients != 1:
        raise ValueError("The local baseline is defined for exactly one client.")
    if args.num_rounds < 1:
        raise ValueError("--num-rounds must be at least 1")
    if args.local_epochs < 1:
        raise ValueError("--local-epochs must be at least 1")
    if args.client_num_cpus <= 0:
        raise ValueError("--client-num-cpus must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    set_seed()
    num_threads = client_num_threads(args.client_num_cpus)
    configure_torch_threads(num_threads)
    device = resolve_device(args.device)
    model_config = get_resnet_model_config_from_args(args)
    client_model_cls, server_model_cls = get_model_classes(
        args.dataset,
        model_config=model_config,
    )
    model = FullNet(client_model_cls, server_model_cls).to(device)
    load_checkpoint(
        args.checkpoint_path,
        model,
        device,
        args.noniid_alpha,
        args.dataset,
        model_config,
    )

    trainloader = DataLoader(
        client_subset(
            client_id=0,
            num_clients=1,
            noniid_alpha=args.noniid_alpha,
            dataset_name=args.dataset,
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    evaluate_for_dataset = partial(evaluate_model, dataset_name=args.dataset)
    max_batches = args.max_batches or None
    total_stats = RuntimeStats()

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print("Number of clients: 1")
    print(f"Non-IID alpha: {args.noniid_alpha}")
    print(f"PyTorch threads: {num_threads}")
    print(f"Model config: {model_config}")
    print("Start local ResNet baseline")

    for round_idx in range(1, args.num_rounds + 1):
        round_stats = RuntimeStats()
        train_start = time.perf_counter()
        train(
            model,
            trainloader,
            epochs=args.local_epochs,
            device=device,
            max_batches=max_batches,
        )
        training_time = time.perf_counter() - train_start
        round_stats.add_training(training_time)
        total_stats.add_training(training_time)

        print(f"\n========== Round {round_idx} ==========")
        print(f"Round {round_idx} summary:")
        if args.eval_every_round:
            loss, accuracy = evaluate_for_dataset(model, device)
            print_test_metrics("Round", loss, accuracy)
        round_stats.print_summary(f"Round {round_idx}")

    print("\nTraining finished.")
    loss, accuracy = evaluate_for_dataset(model, device)
    print_test_metrics("Final", loss, accuracy)
    save_checkpoint(
        args.checkpoint_path,
        model,
        args.noniid_alpha,
        args.dataset,
        model_config,
    )
    total_stats.print_summary("Total")


if __name__ == "__main__":
    main()
