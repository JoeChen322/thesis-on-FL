import argparse
from itertools import product
import math
import os
import subprocess
import sys
from pathlib import Path

from split_learning_utils import DATASET_CHOICES, RESNET_SPLIT_POINTS


PARADIGM_SCRIPTS = {
    "local": "local_resnet_baseline.py",
    "fl": "fl_mnist_minimal.py",
    "sl": "sl_mnist_minimal.py",
    "sfl": "fsl_mnist_minimal.py",
}


DEFAULT_PARADIGMS = ("fl", "sl", "sfl")


def checkpoint_label_value(value):
    return str(value).replace(",", "-").replace(".", "p")


def cpu_checkpoint_label(value):
    cpu_value = float(value)
    if cpu_value.is_integer():
        return str(int(cpu_value))
    return checkpoint_label_value(value)


def repeat_label(args, repeat_index):
    if args.repeats == 1:
        return ""
    return f"_rep{repeat_index}"


def group_label(args):
    alpha_label = checkpoint_label_value(args.noniid_alpha)
    return (
        f"{args.dataset}_{args.num_clients}clients_"
        f"{cpu_checkpoint_label(args.client_cpus)}cpu_"
        f"{args.num_rounds}rounds_alpha{alpha_label}_"
        f"resnet18_split{args.resnet_split_after}"
    )


def checkpoint_path(args, paradigm, repeat_index):
    alpha_label = checkpoint_label_value(args.noniid_alpha)
    return (
        Path(args.checkpoint_dir)
        / (
            f"{paradigm}_{args.dataset}_{args.num_clients}clients_"
            f"{cpu_checkpoint_label(args.client_cpus)}cpu_"
            f"{args.num_rounds}rounds_alpha{alpha_label}_"
            f"resnet18_split{args.resnet_split_after}"
            f"{repeat_label(args, repeat_index)}_checkpoint.pt"
        )
    )


def csv_path(args, repeat_index, multi_group):
    path = Path(args.csv_path)
    if not multi_group:
        return path
    suffix = path.suffix or ".csv"
    return path.with_name(
        f"{path.stem}_{group_label(args)}{repeat_label(args, repeat_index)}{suffix}"
    )


def build_training_command(args, project_root, paradigm, checkpoint):
    command = [
        sys.executable,
        str(project_root / PARADIGM_SCRIPTS[paradigm]),
        "--num-clients",
        str(args.num_clients),
        "--num-rounds",
        str(args.num_rounds),
        "--local-epochs",
        str(args.local_epochs),
        "--checkpoint-path",
        str(checkpoint),
        "--dataset",
        args.dataset,
        "--noniid-alpha",
        str(args.noniid_alpha),
        "--resnet-depth",
        "18",
        "--resnet-split-after",
        args.resnet_split_after,
    ]
    if paradigm == "local":
        command.extend(["--client-num-cpus", str(args.client_cpus)])
        command.extend(["--device", args.local_device])
    else:
        command.extend(["--client-num-cpus", str(args.client_cpus)])
        command.extend(["--client-num-gpus", str(args.client_gpus)])
        command.extend(["--communication-delay", args.communication_delay])
    if paradigm in ("local", "sl", "sfl") and args.max_batches:
        command.extend(["--max-batches", str(args.max_batches)])
    if args.eval_every_round:
        command.append("--eval-every-round")
    return command


def build_leakage_command(args, project_root, checkpoints, csv_output_path):
    command = [
        sys.executable,
        str(project_root / "measure_cutlayer_leakage.py"),
        "--dataset",
        args.dataset,
        "--num-clients",
        str(args.num_clients),
        "--noniid-alpha",
        str(args.noniid_alpha),
        "--expected-split-after",
        args.resnet_split_after,
        "--device",
        args.leakage_device,
        "--batch-size",
        str(args.leakage_batch_size),
        "--dcor-batches",
        str(args.dcor_batches),
        "--csv-path",
        str(csv_output_path),
    ]
    if args.dcor_samples:
        command.extend(["--dcor-samples", str(args.dcor_samples)])
    if args.run_reconstruction_attack:
        command.append("--run-reconstruction-attack")
        command.extend(["--attack-train-samples", str(args.attack_train_samples)])
        command.extend(["--attack-eval-samples", str(args.attack_eval_samples)])
        command.extend(["--attack-epochs", str(args.attack_epochs)])
        command.extend(["--attack-lr", str(args.attack_lr)])
    for paradigm, path in checkpoints.items():
        command.extend(["--checkpoint", f"{paradigm}={path}"])
    return command


def run(command, project_root, env):
    print("Running:", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=project_root, env=env, check=True)


def csv_items(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_csv_values(value, cast, choices=None):
    values = [cast(item) for item in csv_items(value)]
    if not values:
        raise ValueError("CSV option must contain at least one value")
    if choices is not None:
        invalid = [value for value in values if value not in choices]
        if invalid:
            raise ValueError(
                f"Unsupported value(s): {', '.join(str(value) for value in invalid)}"
            )
    return values


def list_or_single(args, list_attr, single_attr, cast, choices=None):
    list_value = getattr(args, list_attr)
    if list_value:
        return parse_csv_values(list_value, cast, choices)
    return [getattr(args, single_attr)]


def selected_paradigms(args):
    paradigms = parse_csv_values(args.paradigms, str.lower, PARADIGM_SCRIPTS)
    duplicates = sorted(
        paradigm for paradigm in set(paradigms) if paradigms.count(paradigm) > 1
    )
    if duplicates:
        raise ValueError(
            f"--paradigms contains duplicate value(s): {', '.join(duplicates)}"
        )
    return paradigms


def experiment_args(args):
    datasets = list_or_single(args, "datasets", "dataset", str, DATASET_CHOICES)
    num_clients_values = list_or_single(args, "num_clients_list", "num_clients", int)
    client_cpus_values = list_or_single(args, "client_cpus_list", "client_cpus", float)
    client_gpus_values = list_or_single(args, "client_gpus_list", "client_gpus", float)
    alpha_values = list_or_single(args, "noniid_alphas", "noniid_alpha", float)
    split_values = list_or_single(
        args,
        "resnet_split_afters",
        "resnet_split_after",
        str,
        RESNET_SPLIT_POINTS,
    )

    for (
        dataset,
        num_clients,
        client_cpus,
        client_gpus,
        noniid_alpha,
        resnet_split_after,
    ) in product(
        datasets,
        num_clients_values,
        client_cpus_values,
        client_gpus_values,
        alpha_values,
        split_values,
    ):
        combo_args = argparse.Namespace(**vars(args))
        combo_args.dataset = dataset
        combo_args.num_clients = num_clients
        combo_args.client_cpus = client_cpus
        combo_args.client_gpus = client_gpus
        combo_args.noniid_alpha = noniid_alpha
        combo_args.resnet_split_after = resnet_split_after
        yield combo_args


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train local/FL/SL/SFL ResNet18 runs with isolated checkpoints, "
            "then measure cut-layer leakage for comparison."
        )
    )
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="mnist")
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated dataset grid. Defaults to --dataset.",
    )
    parser.add_argument("--num-clients", type=int, default=1)
    parser.add_argument(
        "--num-clients-list",
        default="",
        help="Comma-separated num-client grid. Defaults to --num-clients.",
    )
    parser.add_argument("--client-cpus", type=float, default=1.0)
    parser.add_argument(
        "--client-cpus-list",
        default="",
        help="Comma-separated client CPU grid. Defaults to --client-cpus.",
    )
    parser.add_argument("--client-gpus", type=float, default=0.0)
    parser.add_argument(
        "--client-gpus-list",
        default="",
        help="Comma-separated client GPU grid. Defaults to --client-gpus.",
    )
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--noniid-alpha", type=float, default=1.0)
    parser.add_argument(
        "--noniid-alphas",
        default="",
        help="Comma-separated non-IID alpha grid. Defaults to --noniid-alpha.",
    )
    parser.add_argument(
        "--resnet-split-after",
        choices=RESNET_SPLIT_POINTS,
        default="layer2",
    )
    parser.add_argument(
        "--resnet-split-afters",
        default="",
        help="Comma-separated split-point grid. Defaults to --resnet-split-after.",
    )
    parser.add_argument(
        "--paradigms",
        default=",".join(DEFAULT_PARADIGMS),
        help="Comma-separated paradigms to train in each group. Default: fl,sl,sfl.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat the full parameter grid this many times. Default: 1.",
    )
    parser.add_argument("--checkpoint-dir", default=".checkpoints")
    parser.add_argument(
        "--csv-path",
        default=".checkpoints/resnet18_1client_1cpu_5rounds_leakage.csv",
    )
    parser.add_argument("--local-device", default="cpu")
    parser.add_argument("--leakage-device", default="cpu")
    parser.add_argument("--communication-delay", default="0")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--eval-every-round", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-leakage", action="store_true")
    parser.add_argument("--leakage-batch-size", type=int, default=64)
    parser.add_argument("--dcor-batches", type=int, default=5)
    parser.add_argument("--dcor-samples", type=int, default=0)
    parser.add_argument("--run-reconstruction-attack", action="store_true")
    parser.add_argument("--attack-train-samples", type=int, default=512)
    parser.add_argument("--attack-eval-samples", type=int, default=256)
    parser.add_argument("--attack-epochs", type=int, default=10)
    parser.add_argument("--attack-lr", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    paradigms = selected_paradigms(args)
    combinations = list(experiment_args(args))
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.num_rounds < 1:
        raise ValueError("--num-rounds must be at least 1")
    for combo_args in combinations:
        if combo_args.num_clients < 1:
            raise ValueError("--num-clients values must be at least 1")
        if combo_args.client_cpus <= 0:
            raise ValueError("--client-cpus values must be positive")
        if combo_args.client_gpus < 0:
            raise ValueError("--client-gpus values must be non-negative")
        if combo_args.noniid_alpha < 0.0 or combo_args.noniid_alpha > 1.0:
            raise ValueError("--noniid-alpha values must be in the range [0, 1]")
        if "local" in paradigms and combo_args.num_clients != 1:
            raise ValueError("The local baseline is defined for exactly one client.")

    project_root = Path(__file__).resolve().parent
    env = os.environ.copy()
    multi_group = args.repeats > 1 or len(combinations) > 1

    for repeat_index in range(1, args.repeats + 1):
        print(f"\n========== Repeat {repeat_index}/{args.repeats} ==========", flush=True)
        for combo_index, combo_args in enumerate(combinations, start=1):
            print(
                f"\n========== Parameter Group {combo_index}/{len(combinations)}: "
                f"{group_label(combo_args)} ==========",
                flush=True,
            )
            checkpoints = {
                paradigm: checkpoint_path(combo_args, paradigm, repeat_index)
                for paradigm in paradigms
            }
            env["SIMULATION_TOTAL_CPUS"] = str(
                max(1, math.ceil(combo_args.num_clients * combo_args.client_cpus))
            )

            if not combo_args.skip_training:
                for paradigm, path in checkpoints.items():
                    print(f"\n========== Train {paradigm.upper()} ==========", flush=True)
                    run(
                        build_training_command(
                            combo_args,
                            project_root,
                            paradigm,
                            path,
                        ),
                        project_root,
                        env,
                    )

            leakage_csv_path = csv_path(combo_args, repeat_index, multi_group)
            if not combo_args.skip_leakage:
                missing = [str(path) for path in checkpoints.values() if not path.exists()]
                if missing:
                    raise FileNotFoundError(
                        "Missing checkpoint(s). Run without --skip-training first: "
                        + ", ".join(missing)
                    )
                print("\n========== Measure Leakage ==========", flush=True)
                run(
                    build_leakage_command(
                        combo_args,
                        project_root,
                        checkpoints,
                        leakage_csv_path,
                    ),
                    project_root,
                    env,
                )

            print("\nComparison checkpoints:")
            for paradigm, path in checkpoints.items():
                print(f"{paradigm}: {path}")
            if not combo_args.skip_leakage:
                print(f"Leakage CSV: {leakage_csv_path}")


if __name__ == "__main__":
    main()
