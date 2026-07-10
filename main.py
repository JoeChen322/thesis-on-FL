import argparse
from pathlib import Path
import subprocess
import sys


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


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    method, selection_reason = choose_method(args)
    python_executable = resolve_python_executable()

    if method == "fl":
        command = [
            python_executable,
            str(project_root / "fl_mnist_minimal.py"),
            "--mode",
            args.fl_mode,
            "--num-clients",
            str(args.num_clients),
            "--num-rounds",
            str(args.num_rounds),
            "--client-id",
            str(args.client_id),
            "--server-address",
            args.server_address,
        ]
    else:
        command = [
            python_executable,
            str(project_root / "sl_mnist_minimal.py"),
            "--num-clients",
            str(args.num_clients),
            "--num-rounds",
            str(args.num_rounds),
        ]

    print(f"Selected method: {method.upper()} for {args.num_clients} clients", flush=True)
    print(f"Selection reason: {selection_reason}", flush=True)
    print(f"Using Python: {python_executable}", flush=True)
    subprocess.run(command, check=True, cwd=project_root)


if __name__ == "__main__":
    main()
