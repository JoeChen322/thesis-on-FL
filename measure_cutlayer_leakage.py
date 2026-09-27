"""
Measure leakage for one trained SFL client cut layer.

This script is for a concrete trained client model, for example a client
checkpoint whose split point is layer4. It does not use forward hooks. The
client model's forward output is exactly the smashed data Z sent to the server.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from dcor_privacy import ReconstructionAttackTester, SimpleDecoder, dist_corr
from split_learning_utils import (
    DATASET_CHOICES,
    DATASET_SPECS,
    RESNET_SPLIT_POINTS,
    SplitResNetClientNet,
    add_resnet_model_args,
    client_size,
    fedavg_state_dicts,
    get_resnet_model_config_from_args,
    load_dataset,
    normalize_resnet_config,
    set_seed,
)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_checkpoint(path: str) -> dict:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu")


def checkpoint_model_config(checkpoint: dict) -> dict | None:
    partition = checkpoint.get("partition")
    if not partition:
        return None
    model_config = partition.get("model")
    if not model_config:
        return None
    return normalize_resnet_config(**model_config)


def looks_like_state_dict(value) -> bool:
    return isinstance(value, dict) and value and all(
        isinstance(item, torch.Tensor) for item in value.values()
    )


def select_client_state(
    checkpoint,
    dataset_name: str,
    fallback_num_clients: int,
    fallback_noniid_alpha: float,
    checkpoint_client_id: int,
) -> tuple[dict, str]:
    """Return a client state_dict from supported checkpoint formats."""
    if looks_like_state_dict(checkpoint):
        return checkpoint, "standalone client state_dict"

    if "client_model" in checkpoint and looks_like_state_dict(checkpoint["client_model"]):
        return checkpoint["client_model"], "checkpoint['client_model']"

    if "model_state_dict" in checkpoint and looks_like_state_dict(checkpoint["model_state_dict"]):
        return checkpoint["model_state_dict"], "checkpoint['model_state_dict']"

    if "client_models" not in checkpoint:
        raise ValueError(
            "Unsupported checkpoint format. Expected a standalone client state_dict, "
            "a dict with 'client_model'/'model_state_dict', or an SFL checkpoint "
            "with 'client_models'."
        )

    client_states = checkpoint["client_models"]
    if checkpoint_client_id >= 0:
        if checkpoint_client_id >= len(client_states):
            raise ValueError(
                f"--checkpoint-client-id {checkpoint_client_id} is outside checkpoint "
                f"client range [0, {len(client_states) - 1}]"
            )
        return client_states[checkpoint_client_id], f"SFL client {checkpoint_client_id}"

    partition = checkpoint.get("partition") or {}
    num_clients = int(partition.get("num_clients", fallback_num_clients))
    noniid_alpha = float(partition.get("noniid_alpha", fallback_noniid_alpha))
    checkpoint_dataset = partition.get("dataset", dataset_name)
    if len(client_states) == num_clients:
        sizes = [
            client_size(
                num_clients,
                client_id,
                noniid_alpha=noniid_alpha,
                dataset_name=checkpoint_dataset,
            )
            for client_id in range(num_clients)
        ]
    else:
        sizes = [1 for _ in client_states]
    return fedavg_state_dicts(client_states, sizes), "weighted FedAvg client state"


def load_client_model(
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: str,
) -> tuple[torch.nn.Module, str, dict]:
    checkpoint = load_checkpoint(checkpoint_path)

    model_config = get_resnet_model_config_from_args(args)
    checkpoint_config = checkpoint_model_config(checkpoint) if isinstance(checkpoint, dict) else None
    if checkpoint_config is not None:
        model_config = checkpoint_config

    if model_config["split_after"] != args.expected_split_after:
        raise ValueError(
            f"Checkpoint/model config split_after is '{model_config['split_after']}', "
            f"but --expected-split-after is '{args.expected_split_after}'. "
            "Use the matching split point or pass the correct expectation."
        )

    if isinstance(checkpoint, dict):
        checkpoint_dataset = (checkpoint.get("partition") or {}).get("dataset")
        if checkpoint_dataset is not None and checkpoint_dataset != args.dataset:
            raise ValueError(
                f"Checkpoint dataset is '{checkpoint_dataset}', "
                f"but --dataset is '{args.dataset}'"
            )

    input_channels = DATASET_SPECS[args.dataset]["input_channels"]
    client_model = SplitResNetClientNet(
        input_channels=input_channels,
        model_config=model_config,
    )
    client_state, state_label = select_client_state(
        checkpoint,
        args.dataset,
        args.num_clients,
        args.noniid_alpha,
        args.checkpoint_client_id,
    )
    client_model.load_state_dict(client_state, strict=True)
    client_model = copy.deepcopy(client_model).to(device).eval()
    for parameter in client_model.parameters():
        parameter.requires_grad_(False)
    return client_model, state_label, model_config


def make_subset_loader(
    dataset_name: str,
    batch_size: int,
    start: int,
    num_samples: int,
) -> DataLoader:
    dataset = load_dataset(dataset_name, train=False)
    if num_samples > 0:
        stop = min(start + num_samples, len(dataset))
        indices = range(start, stop)
        dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def measure_dcor(
    client_model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_batches: int,
) -> tuple[dict, tuple[int, ...]]:
    values = []
    total_samples = 0
    z_shape = None
    z_bytes_per_sample = None

    for batch_index, (x, _y) in enumerate(dataloader):
        if batch_index >= num_batches:
            break
        x = x.to(device)
        z = client_model(x)
        values.append(float(dist_corr(x, z).item()))
        total_samples += int(x.shape[0])
        z_shape = tuple(int(dim) for dim in z.shape[1:])
        z_bytes_per_sample = int(math.prod(z_shape) * z.element_size())

    if not values:
        raise RuntimeError("No DCOR batches were measured.")

    mean = sum(values) / len(values)
    std = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
    return (
        {
            "mean_dcor": mean,
            "std_dcor": std,
            "batches": len(values),
            "samples": total_samples,
            "z_shape": "x".join(str(dim) for dim in z_shape),
            "z_dim": int(math.prod(z_shape)),
            "z_bytes_per_sample": z_bytes_per_sample,
        },
        z_shape,
    )


@torch.no_grad()
def build_activation_pairs(
    client_model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> TensorDataset:
    z_batches = []
    x_batches = []
    for x, _y in dataloader:
        x = x.to(device)
        z = client_model(x)
        z_batches.append(z.detach().cpu())
        x_batches.append(x.detach().cpu())
    if not z_batches:
        raise RuntimeError("No activation pairs were generated.")
    return TensorDataset(torch.cat(z_batches, dim=0), torch.cat(x_batches, dim=0))


def run_reconstruction_attack(
    client_model: torch.nn.Module,
    dataset_name: str,
    batch_size: int,
    device: torch.device,
    train_samples: int,
    eval_samples: int,
    epochs: int,
    lr: float,
    z_shape: Sequence[int],
) -> dict:
    dcor_sample_count = 0
    train_loader = make_subset_loader(
        dataset_name,
        batch_size=batch_size,
        start=dcor_sample_count,
        num_samples=train_samples,
    )
    eval_loader = make_subset_loader(
        dataset_name,
        batch_size=batch_size,
        start=train_samples,
        num_samples=eval_samples,
    )
    attack_train_pairs = build_activation_pairs(client_model, train_loader, device)
    attack_eval_pairs = build_activation_pairs(client_model, eval_loader, device)

    input_channels = DATASET_SPECS[dataset_name]["input_channels"]
    input_size = 28 if dataset_name == "mnist" else 32
    decoder = SimpleDecoder(
        in_channels=int(z_shape[0]),
        out_channels=input_channels,
        out_size=input_size,
    )
    tester = ReconstructionAttackTester(decoder, device=str(device), lr=lr)
    tester.fit(
        DataLoader(attack_train_pairs, batch_size=batch_size, shuffle=True),
        epochs=epochs,
    )
    mse = tester.evaluate(
        DataLoader(attack_eval_pairs, batch_size=batch_size, shuffle=False)
    )
    return {
        "attack_mse": float(mse),
        "attack_train_samples": len(attack_train_pairs),
        "attack_eval_samples": len(attack_eval_pairs),
        "attack_epochs": epochs,
    }


def write_csv(path: str, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not path or not rows:
        return
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV: {csv_path}")


def parse_checkpoint_spec(value: str) -> tuple[str, str]:
    if "=" in value:
        label, path = value.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label:
            raise ValueError(f"Checkpoint label is empty in spec: {value}")
        if not path:
            raise ValueError(f"Checkpoint path is empty in spec: {value}")
        return label, path

    path = value.strip()
    if not path:
        raise ValueError("Checkpoint path cannot be empty")
    return Path(path).stem, path


def checkpoint_specs(args: argparse.Namespace) -> list[tuple[str, str]]:
    specs = [parse_checkpoint_spec(value) for value in args.checkpoint]
    if args.checkpoint_path:
        label = args.label or Path(args.checkpoint_path).stem
        specs.insert(0, (label, args.checkpoint_path))
    if not specs:
        raise ValueError(
            "Pass --checkpoint-path for one checkpoint or one or more "
            "--checkpoint LABEL=PATH entries for comparison."
        )
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure true leakage of one trained SFL client checkpoint."
    )
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument(
        "--label",
        default="",
        help="Optional label for --checkpoint-path in the report/CSV.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help=(
            "Checkpoint comparison entry. Use LABEL=PATH. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="mnist")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dcor-batches", type=int, default=5)
    parser.add_argument(
        "--dcor-samples",
        type=int,
        default=0,
        help="Optional test-set cap for DCOR. 0 means dcor-batches * batch-size.",
    )
    parser.add_argument(
        "--expected-split-after",
        choices=RESNET_SPLIT_POINTS,
        default="layer4",
        help="Sanity check that the loaded client is cut at this layer.",
    )
    parser.add_argument(
        "--checkpoint-client-id",
        type=int,
        default=0,
        help="Client model to load from an SFL checkpoint. Use -1 for weighted FedAvg.",
    )
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--noniid-alpha", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--csv-path", default="")
    parser.add_argument(
        "--run-reconstruction-attack",
        action="store_true",
        help="Train a small decoder attack on held-out test activations.",
    )
    parser.add_argument("--attack-train-samples", type=int, default=512)
    parser.add_argument("--attack-eval-samples", type=int, default=256)
    parser.add_argument(
        "--attack-epochs",
        type=int,
        nargs="?",
        const=10,
        default=10,
        help="Decoder attack training epochs. If the flag is passed without a value, uses 10.",
    )
    parser.add_argument("--attack-lr", type=float, default=1e-3)
    add_resnet_model_args(parser, default_split_after="layer4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2 for DCOR")
    if args.dcor_batches < 1:
        raise ValueError("--dcor-batches must be at least 1")
    if args.dcor_samples < 0:
        raise ValueError("--dcor-samples must be non-negative")
    if args.num_clients < 1:
        raise ValueError("--num-clients must be at least 1")
    if args.attack_train_samples < 1 or args.attack_eval_samples < 1:
        raise ValueError("Attack sample counts must be positive")
    if args.attack_epochs < 1:
        raise ValueError("--attack-epochs must be at least 1")

    set_seed()
    device = resolve_device(args.device)
    rows = []

    for label, path in checkpoint_specs(args):
        client_model, state_label, model_config = load_client_model(args, device, path)
        dcor_samples = args.dcor_samples or args.dcor_batches * args.batch_size
        dcor_loader = make_subset_loader(
            args.dataset,
            batch_size=args.batch_size,
            start=0,
            num_samples=dcor_samples,
        )
        dcor_result, z_shape = measure_dcor(
            client_model,
            dcor_loader,
            device,
            num_batches=args.dcor_batches,
        )

        result = {
            "paradigm": label,
            "checkpoint_path": path,
            "dataset": args.dataset,
            "split_after": model_config["split_after"],
            "weights": state_label,
            **dcor_result,
        }

        if args.run_reconstruction_attack:
            attack_result = run_reconstruction_attack(
                client_model,
                dataset_name=args.dataset,
                batch_size=args.batch_size,
                device=device,
                train_samples=args.attack_train_samples,
                eval_samples=args.attack_eval_samples,
                epochs=args.attack_epochs,
                lr=args.attack_lr,
                z_shape=z_shape,
            )
            result.update(attack_result)

        rows.append(result)

        print("SFL checkpoint leakage report")
        print(f"Paradigm: {label}")
        print(f"Checkpoint: {path}")
        print(f"Device: {device}")
        print(f"Dataset: {args.dataset} (test split)")
        print(f"Loaded weights: {state_label}")
        print(f"Model config: {model_config}")
        print(f"Smashed data Z shape: {result['z_shape']}")
        print(f"Smashed data Z dim: {result['z_dim']}")
        print(f"Smashed data bytes/sample: {result['z_bytes_per_sample']}")
        print(
            f"DCOR(X, Z): mean={result['mean_dcor']:.6f}, "
            f"std={result['std_dcor']:.6f}, "
            f"batches={result['batches']}, samples={result['samples']}"
        )
        if args.run_reconstruction_attack:
            print(
                f"Reconstruction attack MSE: {result['attack_mse']:.6f} "
                f"(train_samples={result['attack_train_samples']}, "
                f"eval_samples={result['attack_eval_samples']}, "
                f"epochs={result['attack_epochs']})"
            )
        else:
            print("Reconstruction attack: skipped; pass --run-reconstruction-attack to run it.")
        print()

    print(
        "Interpretation: lower DCOR means weaker statistical dependence between X "
        "and Z. Higher reconstruction MSE means the trained decoder attack had "
        "more difficulty reconstructing X from Z."
    )

    write_csv(args.csv_path, rows)


if __name__ == "__main__":
    main()
