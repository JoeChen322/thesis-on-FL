import argparse
from pathlib import Path
import subprocess
import sys


def choose_method(num_clients):
    if num_clients > 3:
        return "fl"
    return "sl"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clients", type=int, required=True)
    parser.add_argument("--num-rounds", type=int, default=3)
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
    method = choose_method(args.num_clients)
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
    print(f"Using Python: {python_executable}", flush=True)
    subprocess.run(command, check=True, cwd=project_root)


if __name__ == "__main__":
    main()
