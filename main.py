import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time

from noniid_jsd_switch import (
    DEFAULT_IID_JSD_THRESHOLD,
    DEFAULT_STRONG_NONIID_JSD_THRESHOLD,
)
from split_learning_utils import add_resnet_model_args


def choose_method(args):
    if args.method != "auto":
        return args.method, f"manual override: --method {args.method}"

    client_cpus = parse_client_cpus(args.client_cpus, args.num_clients)
    if args.num_clients > 5 and all(cpu_count >= 2 for cpu_count in client_cpus):
        return "fl", "more than 5 clients and every client has at least 2 CPUs"

    if args.num_clients < 5:
        return "sl", "fewer than 5 clients"

    return "sfl", "middle case uses SFL"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clients", type=int, required=True)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--method", choices=("auto", "fl", "sl", "sfl"), default="auto")
    parser.add_argument("--dataset", choices=("mnist", "cifar10"), default="mnist")
    parser.add_argument(
        "--client-cpus",
        default="1",
        help="Client CPU counts. Use one integer for all clients or comma-separated values.",
    )
    parser.add_argument(
        "--client-gpus",
        type=float,
        default=0.0,
        help="GPU count reserved for each Flower simulation client.",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
        help="Local training epochs per client round.",
    )
    parser.add_argument(
        "--accuracy-priority",
        choices=("low", "normal", "high"),
        default="normal",
        help="How strongly to prefer the method with better observed accuracy.",
    )
    parser.add_argument(
        "--adaptive-communication-switch",
        action="store_true",
        help="Measure communication time each round and switch method if it grows too much.",
    )
    parser.add_argument(
        "--communication-delay",
        default="0",
        help="Extra delay seconds per round. Use one value or comma-separated values.",
    )
    parser.add_argument(
        "--switch-threshold",
        type=float,
        default=0.2,
        help="Switch method when communication time is higher than the previous round by this ratio.",
    )
    parser.add_argument(
        "--time-threshold",
        type=float,
        default=None,
        help=(
            "Upper per-round total time threshold in seconds. When set with "
            "--adaptive-communication-switch, any round above this threshold "
            "switches the next round to FL."
        ),
    )
    parser.add_argument(
        "--time-threshold-low",
        type=float,
        default=None,
        help=(
            "Lower per-round total time threshold in seconds. When set with "
            "--time-threshold, an FL round below this threshold switches back "
            "to the most recent SL/SFL method."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=".checkpoints",
        help="Directory used to save the shared checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="",
        help="Shared checkpoint path used by FL/SL/SFL. Defaults to checkpoint-dir/shared_<dataset>_<num-clients>clients_alpha<noniid-alpha>_checkpoint.pt.",
    )
    parser.add_argument(
        "--noniid-alpha",
        type=float,
        default=1.0,
        help="Shared Dirichlet non-IID degree in [0, 1]. 1.0 keeps IID splitting.",
    )
    parser.add_argument(
        "--adaptive-noniid-switch",
        action="store_true",
        help="Let clients report one local JSD scalar for server-side pattern toggling.",
    )
    parser.add_argument(
        "--iid-jsd-threshold",
        type=float,
        default=DEFAULT_IID_JSD_THRESHOLD,
        help="Mean client boundary log2-JSD at or below this value is treated as IID.",
    )
    parser.add_argument(
        "--strong-noniid-jsd-threshold",
        type=float,
        default=DEFAULT_STRONG_NONIID_JSD_THRESHOLD,
        help="Mean client boundary log2-JSD at or above this value is treated as strong non-IID.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Debug limit for SL/SFL batches per client. 0 means no limit.",
    )
    parser.add_argument(
        "--eval-every-round",
        action="store_true",
        help="Evaluate the full test set after every SL/SFL round.",
    )
    add_resnet_model_args(parser)
    return parser.parse_args()


def resolve_python_executable():
    return sys.executable


def shared_checkpoint_path(project_root, args):
    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path)
        if checkpoint_path.is_absolute():
            return checkpoint_path
        return project_root / checkpoint_path
    alpha_label = str(args.noniid_alpha).replace(".", "p")
    model_label = model_checkpoint_label(args)
    return (
        project_root
        / args.checkpoint_dir
        / (
            f"shared_{args.dataset}_{args.num_clients}clients_"
            f"alpha{alpha_label}_{model_label}_checkpoint.pt"
        )
    )


def checkpoint_label_value(value):
    return str(value).replace(",", "-").replace(".", "p")


def model_checkpoint_label(args):
    parts = [f"resnet{args.resnet_depth}"]
    if args.resnet_block:
        parts.append(args.resnet_block)
    if args.resnet_blocks:
        parts.append(f"blocks{checkpoint_label_value(args.resnet_blocks)}")
    if args.resnet_widths != "64,128,256,512":
        parts.append(f"widths{checkpoint_label_value(args.resnet_widths)}")
    if args.resnet_split_after != "layer1":
        parts.append(f"split{args.resnet_split_after}")
    if args.resnet_stem_kernel != 7:
        parts.append(f"stemk{args.resnet_stem_kernel}")
    if args.resnet_stem_stride != 2:
        parts.append(f"stems{args.resnet_stem_stride}")
    return "_".join(parts)


def extend_resnet_command_args(command, args):
    command.extend(["--resnet-depth", str(args.resnet_depth)])
    if args.resnet_block:
        command.extend(["--resnet-block", args.resnet_block])
    if args.resnet_blocks:
        command.extend(["--resnet-blocks", args.resnet_blocks])
    command.extend(["--resnet-widths", args.resnet_widths])
    command.extend(["--resnet-split-after", args.resnet_split_after])
    command.extend(["--resnet-stem-kernel", str(args.resnet_stem_kernel)])
    command.extend(["--resnet-stem-stride", str(args.resnet_stem_stride)])
    return command


def parse_client_cpus(value, num_clients):
    cpu_counts = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not cpu_counts:
        raise ValueError("--client-cpus must contain at least one positive integer")
    if any(cpu_count < 1 for cpu_count in cpu_counts):
        raise ValueError("--client-cpus values must be positive integers")

    if len(cpu_counts) == 1:
        return cpu_counts * num_clients
    if len(cpu_counts) != num_clients:
        raise ValueError("--client-cpus list length must equal --num-clients")
    return cpu_counts


def simulation_client_num_cpus(args):
    cpu_counts = parse_client_cpus(args.client_cpus, args.num_clients)
    unique_cpu_counts = set(cpu_counts)
    if len(unique_cpu_counts) != 1:
        raise ValueError(
            "Flower Ray simulation supports one CPU resource value for all clients; "
            "use a single --client-cpus value, for example --client-cpus 4."
        )
    return float(cpu_counts[0])


def build_command(
    args,
    project_root,
    python_executable,
    method,
    num_rounds,
    checkpoint_path=None,
    communication_delay=None,
):
    client_num_cpus = simulation_client_num_cpus(args)
    if args.noniid_alpha < 0.0 or args.noniid_alpha > 1.0:
        raise ValueError("--noniid-alpha must be in the range [0, 1]")
    if args.client_gpus < 0.0:
        raise ValueError("--client-gpus must be non-negative")
    if args.local_epochs < 1:
        raise ValueError("--local-epochs must be at least 1")

    if method == "fl":
        command = [
            python_executable,
            str(project_root / "fl_mnist_minimal.py"),
            "--num-clients",
            str(args.num_clients),
            "--num-rounds",
            str(num_rounds),
            "--client-num-cpus",
            str(client_num_cpus),
            "--client-num-gpus",
            str(args.client_gpus),
            "--local-epochs",
            str(args.local_epochs),
            "--dataset",
            args.dataset,
        ]
        if checkpoint_path is not None:
            command.extend(["--checkpoint-path", str(checkpoint_path)])
        command.extend(["--noniid-alpha", str(args.noniid_alpha)])
        if communication_delay is not None:
            command.extend(["--communication-delay", str(communication_delay)])
        extend_resnet_command_args(command, args)
        return command

    if method == "sfl":
        command = [
            python_executable,
            str(project_root / "fsl_mnist_minimal.py"),
            "--num-clients",
            str(args.num_clients),
            "--num-rounds",
            str(num_rounds),
            "--client-num-cpus",
            str(client_num_cpus),
            "--client-num-gpus",
            str(args.client_gpus),
            "--local-epochs",
            str(args.local_epochs),
            "--dataset",
            args.dataset,
        ]
        if checkpoint_path is not None:
            command.extend(["--checkpoint-path", str(checkpoint_path)])
        command.extend(["--noniid-alpha", str(args.noniid_alpha)])
        if communication_delay is not None:
            command.extend(["--communication-delay", str(communication_delay)])
        if args.max_batches:
            command.extend(["--max-batches", str(args.max_batches)])
        if args.eval_every_round:
            command.append("--eval-every-round")
        if args.adaptive_noniid_switch:
            command.append("--boundary-noniid-switch")
            command.extend(["--iid-jsd-threshold", str(args.iid_jsd_threshold)])
            command.extend([
                "--strong-noniid-jsd-threshold",
                str(args.strong_noniid_jsd_threshold),
            ])
        extend_resnet_command_args(command, args)
        return command

    if method == "sl":
        command = [
            python_executable,
            str(project_root / "sl_mnist_minimal.py"),
            "--num-clients",
            str(args.num_clients),
            "--num-rounds",
            str(num_rounds),
            "--client-num-cpus",
            str(client_num_cpus),
            "--client-num-gpus",
            str(args.client_gpus),
            "--local-epochs",
            str(args.local_epochs),
            "--dataset",
            args.dataset,
        ]
        if checkpoint_path is not None:
            command.extend(["--checkpoint-path", str(checkpoint_path)])
        command.extend(["--noniid-alpha", str(args.noniid_alpha)])
        if communication_delay is not None:
            command.extend(["--communication-delay", str(communication_delay)])
        if args.max_batches:
            command.extend(["--max-batches", str(args.max_batches)])
        if args.eval_every_round:
            command.append("--eval-every-round")
        if args.adaptive_noniid_switch:
            command.append("--boundary-noniid-switch")
            command.extend(["--iid-jsd-threshold", str(args.iid_jsd_threshold)])
            command.extend([
                "--strong-noniid-jsd-threshold",
                str(args.strong_noniid_jsd_threshold),
            ])
        extend_resnet_command_args(command, args)
        return command

    raise ValueError(f"Unknown method: {method}")


def switch_method(method):
    if method == "sl":
        return "sfl"
    if method == "sfl":
        return "fl"
    return method


def default_split_method(args):
    if args.method in ("sl", "sfl"):
        return args.method
    if args.num_clients < 5:
        return "sl"
    return "sfl"


def parse_communication_delay(value):
    delays = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not delays:
        return [0.0]
    if any(delay < 0 for delay in delays):
        raise ValueError("--communication-delay must contain non-negative seconds")
    return delays


def communication_delay_for_round(delays, round_index):
    if round_index < len(delays):
        return delays[round_index]
    return delays[-1]


def timestamp():
    return datetime.now().isoformat(timespec="milliseconds")


def run_command(command, project_root, simulated_delay=0.0):
    start_timestamp = timestamp()
    start_time = time.perf_counter()

    print(f"Communication start timestamp: {start_timestamp}", flush=True)

    subprocess.run(command, check=True, cwd=project_root)

    end_timestamp = timestamp()
    real_elapsed_time = time.perf_counter() - start_time
    communication_time = real_elapsed_time + simulated_delay
    print(f"Communication end timestamp:   {end_timestamp}", flush=True)
    print(f"Measured real time:            {real_elapsed_time:.4f}s", flush=True)
    if simulated_delay > 0:
        print(f"Manual simulated delay:        {simulated_delay:.4f}s", flush=True)
    print(f"Total communication time:      {communication_time:.4f}s", flush=True)
    return communication_time


def run_once(args, project_root, python_executable, method, selection_reason):
    communication_delays = parse_communication_delay(args.communication_delay)
    total_simulated_delay = sum(
        communication_delay_for_round(communication_delays, round_index)
        for round_index in range(args.num_rounds)
    )
    command = build_command(
        args=args,
        project_root=project_root,
        python_executable=python_executable,
        method=method,
        num_rounds=args.num_rounds,
        checkpoint_path=shared_checkpoint_path(project_root, args),
        communication_delay=args.communication_delay,
    )

    print(f"Selected method: {method.upper()} for {args.num_clients} clients", flush=True)
    print(f"Selection reason: {selection_reason}", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    if args.adaptive_noniid_switch:
        print(
            "Boundary non-IID switch: enabled; clients report one local JSD scalar",
            flush=True,
        )
    print(f"Shared non-IID alpha: {args.noniid_alpha}", flush=True)
    print(f"Using Python: {python_executable}", flush=True)
    run_command(
        command,
        project_root,
        simulated_delay=total_simulated_delay,
    )


def run_with_adaptive_switch(args, project_root, python_executable, method, selection_reason):
    communication_delays = parse_communication_delay(args.communication_delay)
    previous_communication_time = None
    last_split_method = method if method in ("sl", "sfl") else default_split_method(args)
    if args.time_threshold is not None and args.time_threshold < 0:
        raise ValueError("--time-threshold must be non-negative")
    if args.time_threshold_low is not None:
        if args.time_threshold_low < 0:
            raise ValueError("--time-threshold-low must be non-negative")
        if args.time_threshold is None:
            raise ValueError("--time-threshold-low requires --time-threshold")
        if args.time_threshold_low >= args.time_threshold:
            raise ValueError("--time-threshold-low must be lower than --time-threshold")

    print(f"Initial method: {method.upper()} for {args.num_clients} clients", flush=True)
    print(f"Initial selection reason: {selection_reason}", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    if args.adaptive_noniid_switch:
        print(
            "Boundary non-IID switch: enabled; clients report one local JSD scalar",
            flush=True,
        )
    print(f"Shared non-IID alpha: {args.noniid_alpha}", flush=True)
    print(f"Using Python: {python_executable}", flush=True)
    if args.time_threshold is not None:
        if args.time_threshold_low is None:
            print(
                "Adaptive rule: switch to FL when total round time "
                f"exceeds {args.time_threshold:.4f}s",
                flush=True,
            )
        else:
            print(
                "Adaptive rule: use time threshold interval "
                f"[{args.time_threshold_low:.4f}s, {args.time_threshold:.4f}s]; "
                "above upper switches to FL, below lower switches back to "
                f"{last_split_method.upper()}",
                flush=True,
            )
    else:
        print(
            "Adaptive rule: switch when communication time grows "
            f"more than {args.switch_threshold * 100:.1f}% from the previous round",
            flush=True,
        )

    for round_index in range(args.num_rounds):
        round_number = round_index + 1
        print(f"\n========== Adaptive Round {round_number}/{args.num_rounds} ==========", flush=True)
        if method in ("sl", "sfl"):
            last_split_method = method

        if previous_communication_time is not None:
            print(f"Round {round_number}: previous communication time {previous_communication_time:.4f}s", flush=True)
        else:
            print(f"Round {round_number}: start with {method.upper()}", flush=True)

        command = build_command(
            args=args,
            project_root=project_root,
            python_executable=python_executable,
            method=method,
            num_rounds=1,
            checkpoint_path=shared_checkpoint_path(project_root, args),
            communication_delay=communication_delay_for_round(
                communication_delays,
                round_index,
            ),
        )
        round_time = run_command(
            command,
            project_root,
            simulated_delay=communication_delay_for_round(communication_delays, round_index),
        )

        if args.time_threshold is not None:
            if round_time > args.time_threshold:
                old_method = method
                method = "fl"
                if method != old_method:
                    print(
                        f"Round {round_number}: {round_time:.4f}s > "
                        f"{args.time_threshold:.4f}s, next round switches "
                        f"{old_method.upper()} -> FL",
                        flush=True,
                    )
                else:
                    print(
                        f"Round {round_number}: {round_time:.4f}s > "
                        f"{args.time_threshold:.4f}s, next round keeps FL",
                        flush=True,
                    )
            else:
                if (
                    args.time_threshold_low is not None
                    and method == "fl"
                    and round_time < args.time_threshold_low
                ):
                    old_method = method
                    method = last_split_method
                    print(
                        f"Round {round_number}: {round_time:.4f}s < "
                        f"{args.time_threshold_low:.4f}s, next round switches "
                        f"{old_method.upper()} -> {method.upper()}",
                        flush=True,
                    )
                else:
                    print(
                        f"Round {round_number}: {round_time:.4f}s within fixed threshold, "
                        f"next round keeps {method.upper()}",
                        flush=True,
                    )
        elif previous_communication_time is not None:
            switch_limit = previous_communication_time * (1 + args.switch_threshold)
            if round_time > switch_limit:
                old_method = method
                method = switch_method(method)
                if method != old_method:
                    print(
                        f"Round {round_number}: {round_time:.4f}s > "
                        f"{previous_communication_time:.4f}s + "
                        f"{args.switch_threshold * 100:.1f}%, next round switches "
                        f"{old_method.upper()} -> {method.upper()}",
                        flush=True,
                    )
                else:
                    print(
                        f"Round {round_number}: {round_time:.4f}s > "
                        f"{previous_communication_time:.4f}s + "
                        f"{args.switch_threshold * 100:.1f}%, "
                        f"but {method.upper()} delay switching is disabled",
                        flush=True,
                    )
            else:
                print(
                    f"Round {round_number}: {round_time:.4f}s within threshold, "
                    f"next round keeps {method.upper()}",
                    flush=True,
                )

        previous_communication_time = round_time


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent
    method, selection_reason = choose_method(args)
    python_executable = resolve_python_executable()

    if args.adaptive_communication_switch:
        run_with_adaptive_switch(args, project_root, python_executable, method, selection_reason)
    else:
        run_once(args, project_root, python_executable, method, selection_reason)


if __name__ == "__main__":
    main()
