import argparse
from functools import partial
import torch
from flower_split_message import run_message_simulation
from split_learning_utils import (
    ClientNet,
    ServerNet,
    client_size as message_client_size,
    fedavg_state_dicts as fedavg,
    get_batch as message_get_batch,
    load_split_checkpoint,
    save_split_checkpoint,
    set_seed,
)


# -------------Basic settings---------------------
set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
#learning rate
LR_CLIENT = 0.01
LR_SERVER = 0.01

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--client-num-cpus", type=float, default=1.0)
    parser.add_argument("--client-num-gpus", type=float, default=0.0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument(
        "--noniid-alpha",
        type=float,
        default=1.0,
        help="Shared Dirichlet non-IID degree in [0, 1]. 1.0 keeps IID splitting.",
    )
    return parser.parse_args()

# -----------------Main training process-------------
def main():
    args = parse_args()
    if args.num_clients < 1:
        raise ValueError("--num-clients must be at least 1")

    initial_client_states, initial_server_state = load_split_checkpoint(
        args.checkpoint_path,
        args.num_clients,
        device,
        args.noniid_alpha,
    )
    on_finished_fn = None
    if args.checkpoint_path:
        on_finished_fn = partial(
            save_split_checkpoint,
            args.checkpoint_path,
            num_clients=args.num_clients,
            noniid_alpha=args.noniid_alpha,
        )

    message_client_size_with_alpha = partial(
        message_client_size,
        noniid_alpha=args.noniid_alpha,
    )
    message_get_batch_with_alpha = partial(
        message_get_batch,
        noniid_alpha=args.noniid_alpha,
    )

    run_message_simulation(
        client_model_cls=ClientNet,
        server_model_cls=ServerNet,
        set_seed_fn=set_seed,
        client_size_fn=message_client_size_with_alpha,
        get_batch_fn=message_get_batch_with_alpha,
        fedavg_fn=fedavg,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=BATCH_SIZE,
        lr_client=LR_CLIENT,
        lr_server=LR_SERVER,
        use_client_fedavg=True,
        num_cpus=args.client_num_cpus,
        num_gpus=args.client_num_gpus,
        initial_client_states=initial_client_states,
        initial_server_state=initial_server_state,
        on_finished_fn=on_finished_fn,
        max_batches=args.max_batches or None,
    )


if __name__ == "__main__":
    main()
