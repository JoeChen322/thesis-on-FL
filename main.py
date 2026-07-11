import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time


def choose_method(args):
    if args.method != "auto":
        return args.method, f"manual override: --method {args.method}"

    if args.networking_environment in ("unstable", "limited"):
        return "fl", f"{args.networking_environment} network favors fewer communication rounds"

    if args.accuracy_priority == "high":
        return "fl", "high accuracy priority favors the FL CNN model"

    if args.performance_priority == "high" and args.num_clients <= 3:
        return "sl", "high performance priority with few clients favors SL"

    if args.num_clients > 3:
        return "fl", "more than 3 clients favors FL"

    return "sl", "stable network with few clients favors SL"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clients", type=int, required=True)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--method", choices=("auto", "fl", "sl"), default="auto")
    parser.add_argument(
        "--accuracy-priority",
        choices=("low", "normal", "high"),
        default="normal",
        help="How strongly to prefer the method with better observed accuracy.",
    )
    parser.add_argument(
        "--performance-priority",
        choices=("low", "normal", "high"),
        default="normal",
        help="How strongly to prefer faster local runtime.",
    )
    parser.add_argument(
        "--networking-environment",
        choices=("local", "stable", "unstable", "limited"),
        default="stable",
        help="Network condition used by --method auto.",
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
        "--dry-run",
        action="store_true",
        help="Print selected commands without running training.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=".checkpoints",
        help="Directory used to save per-method checkpoints in adaptive mode.",
    )
    parser.add_argument("--fl-mode", choices=("simulation", "server", "client"), default="simulation")
    parser.add_argument("--client-id", type=int, default=0)
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    return parser.parse_args()


def resolve_python_executable():
    project_root = Path(__file__).resolve().parent
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"

    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def checkpoint_path_for_method(project_root, checkpoint_dir, method):
    return project_root / checkpoint_dir / f"{method}_checkpoint.pt"


def build_command(args, project_root, python_executable, method, num_rounds, checkpoint_path=None):
    if method == "fl":
        command = [
            python_executable,
            str(project_root / "fl_mnist_minimal.py"),
            "--mode",
            args.fl_mode,
            "--num-clients",
            str(args.num_clients),
            "--num-rounds",
            str(num_rounds),
            "--client-id",
            str(args.client_id),
            "--server-address",
            args.server_address,
        ]
        if checkpoint_path is not None:
            command.extend(["--checkpoint-path", str(checkpoint_path)])
        return command

    command = [
        python_executable,
        str(project_root / "sl_mnist_minimal.py"),
        "--num-clients",
        str(args.num_clients),
        "--num-rounds",
        str(num_rounds),
    ]
    if checkpoint_path is not None:
        command.extend(["--checkpoint-path", str(checkpoint_path)])
    return command


def switch_method(method):
    if method == "fl":
        return "sl"
    return "fl"


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


def run_command(command, project_root, dry_run, simulated_delay=0.0):
    start_timestamp = timestamp()
    start_time = time.perf_counter()

    print(f"Communication start timestamp: {start_timestamp}", flush=True)

    if dry_run:
        print(f"Dry run command: {' '.join(command)}", flush=True)
    else:
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
    )

    print(f"Selected method: {method.upper()} for {args.num_clients} clients", flush=True)
    print(f"Selection reason: {selection_reason}", flush=True)
    print(f"Using Python: {python_executable}", flush=True)
    run_command(
        command,
        project_root,
        args.dry_run,
        simulated_delay=total_simulated_delay,
    )


def run_with_adaptive_switch(args, project_root, python_executable, method, selection_reason):
    communication_delays = parse_communication_delay(args.communication_delay)
    previous_communication_time = None

    print(f"Initial method: {method.upper()} for {args.num_clients} clients", flush=True)
    print(f"Initial selection reason: {selection_reason}", flush=True)
    print(f"Using Python: {python_executable}", flush=True)
    print(
        "Adaptive rule: switch when communication time grows "
        f"more than {args.switch_threshold * 100:.1f}% from the previous round",
        flush=True,
    )

    for round_index in range(args.num_rounds):
        round_number = round_index + 1
        print(f"\n========== Adaptive Round {round_number}/{args.num_rounds} ==========", flush=True)

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
            checkpoint_path=checkpoint_path_for_method(project_root, args.checkpoint_dir, method),
        )
        communication_time = run_command(
            command,
            project_root,
            args.dry_run,
            simulated_delay=communication_delay_for_round(communication_delays, round_index),
        )

        if previous_communication_time is not None:
            switch_limit = previous_communication_time * (1 + args.switch_threshold)
            if communication_time > switch_limit:
                old_method = method
                method = switch_method(method)
                print(
                    f"Round {round_number}: {communication_time:.4f}s > "
                    f"{previous_communication_time:.4f}s + "
                    f"{args.switch_threshold * 100:.1f}%, next round switches "
                    f"{old_method.upper()} -> {method.upper()}",
                    flush=True,
                )
            else:
                print(
                    f"Round {round_number}: {communication_time:.4f}s within threshold, "
                    f"next round keeps {method.upper()}",
                    flush=True,
                )

        previous_communication_time = communication_time


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
