"""
Privacy-driven cut-layer switching for SFL (Section 4.2).

SFL is trained one round at a time from a shared checkpoint, as in run_switch.py. After every
round the leakage of the smashed data is measured on the client-side model of client 0 with the
decoder attack of measure_all_leakage.py. The privacy level is the ratio

    rho = MSE(attack) / MSE(mean image)

i.e. how far the attack is from an attacker that always outputs the mean training image
(rho close to 1: little beyond the average image is recovered; rho close to 0: the input is
reconstructed well). If rho falls below --rho-min, the cut is moved one stage deeper
(layer1 -> layer2 -> layer3 -> layer4) before the next round: the layers between the old and the
new cut are moved from the server model into every client model of the checkpoint, so training
continues from the same weights. The cut is never moved back.

With --static the cut stays where it starts (reference run).

Output: <out>/<name>/rounds.csv (one row per round) and logs/.

Run from the repository folder, e.g.
  venv/bin/python run_cutswitch.py --name adaptive
  venv/bin/python run_cutswitch.py --name static_layer1 --static
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

import measure_all_leakage as mal
import measure_cutlayer_leakage as mcl
from run_baseline import parse_log
from split_learning_utils import (
    RESNET_SPLIT_POINTS,
    get_model_classes,
    normalize_model_config,
    normalize_resnet_config,
    set_seed,
)

SCRIPT = "fsl_mnist_minimal.py"
MODEL_ARGS = ["--model", "resnet", "--resnet-depth", "18"]


# ---------------------------------------------------------------------------------------------
# checkpoint re-split
# ---------------------------------------------------------------------------------------------
def resplit_checkpoint(path: Path, new_split: str) -> str:
    """Move the cut of an SFL checkpoint deeper; returns the old split point."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    partition = ck.get("partition") or {}
    old_cfg = normalize_model_config(partition.get("model"))
    old_split = old_cfg["split_after"]
    old_i, new_i = RESNET_SPLIT_POINTS.index(old_split), RESNET_SPLIT_POINTS.index(new_split)
    if new_i <= old_i:
        raise ValueError(f"cut can only move deeper ({old_split} -> {new_split})")
    prefixes = tuple(f"layer{j + 1}." for j in range(old_i + 1, new_i + 1))
    server = ck["server_model"]
    moved = {k: v for k, v in server.items() if k.startswith(prefixes)}
    if not moved:
        raise ValueError(f"no server parameters with prefixes {prefixes}")
    for state in ck["client_models"]:
        for k, v in moved.items():
            state[k] = v.clone()
    for k in moved:
        del server[k]
    new_cfg = dict(old_cfg, split_after=new_split)
    partition["model"] = normalize_model_config(new_cfg)
    ck["partition"] = partition
    tmp = path.with_name(f".{path.name}.resplit.tmp")
    torch.save(ck, tmp)
    os.replace(tmp, path)
    return old_split


# ---------------------------------------------------------------------------------------------
# one round and its leakage
# ---------------------------------------------------------------------------------------------
def train_round(a, split: str, checkpoint: Path, log_path: Path) -> dict:
    cmd = [a.python, str(Path(a.code_dir).resolve() / SCRIPT),
           "--num-clients", str(a.num_clients), "--num-rounds", "1", "--local-epochs", "1",
           "--client-num-cpus", "1", "--client-num-gpus", "0", "--dataset", a.dataset,
           "--noniid-alpha", str(a.alpha), "--communication-delay", "0",
           "--checkpoint-path", str(checkpoint), *MODEL_ARGS, "--resnet-split-after", split,
           "--eval-every-round"]
    if a.max_batches:
        cmd += ["--max-batches", str(a.max_batches)]
    start = time.perf_counter()
    with open(log_path, "w") as log:
        log.write("# " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=Path(a.code_dir).resolve(), stdout=log,
                              stderr=subprocess.STDOUT, text=True, timeout=a.timeout)
    wall = time.perf_counter() - start
    parsed = parse_log(log_path.read_text(errors="replace").split("\n", 1)[-1])
    if proc.returncode != 0 or not parsed["rounds"]:
        raise RuntimeError(f"round failed (rc={proc.returncode}), see {log_path}")
    row = parsed["rounds"][-1]
    return {"acc": row.get("acc", parsed["final"].get("acc")), "loss": row.get("loss"),
            "train": row.get("training_time"), "comm": row.get("communication_time"),
            "total": row.get("total_runtime"), "wall": round(wall, 1)}


def measure(a, checkpoint: Path) -> dict:
    set_seed()
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = normalize_model_config((ck.get("partition") or {}).get("model"))
    model = get_model_classes(a.dataset, cfg)[0]()
    model.load_state_dict(ck["client_models"][0], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = torch.device("cpu")
    t0 = time.perf_counter()
    loader = mcl.make_subset_loader(a.dataset, 64, 0, a.dcor_batches * 64)
    dc, z_shape = mal.dcor_with_baseline(model, loader, device, a.dcor_batches)
    at = mal.reconstruction_attack(model, a.dataset, 64, device, a.attack_train_samples,
                                   a.attack_eval_samples, a.attack_epochs, 1e-3, z_shape)
    return {"split_measured": cfg["split_after"], "z_shape": dc["z_shape"],
            "z_bytes_per_sample": dc["z_bytes_per_sample"],
            "dcor": dc["dcor_mean"], "dcor_perm": dc["dcor_perm_mean"], "delta_dcor": dc["delta_dcor"],
            "attack_mse": at["attack_mse"], "attack_psnr": at["attack_psnr"],
            "mse_mean_image": at["attack_mse_mean_image"],
            "rho": at["attack_mse"] / at["attack_mse_mean_image"],
            "measure_s": round(time.perf_counter() - t0, 1)}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="adaptive")
    p.add_argument("--out", default="cutswitch_results")
    p.add_argument("--dataset", choices=["cifar10", "mnist"], default="cifar10")
    p.add_argument("--num-clients", type=int, default=3)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--start-layer", choices=RESNET_SPLIT_POINTS, default="layer1")
    p.add_argument("--rho-min", type=float, default=0.4,
                   help="move the cut deeper when MSE(attack)/MSE(mean image) < rho-min")
    p.add_argument("--static", action="store_true", help="never move the cut (reference run)")
    p.add_argument("--dcor-batches", type=int, default=10)
    p.add_argument("--attack-train-samples", type=int, default=512)
    p.add_argument("--attack-eval-samples", type=int, default=256)
    p.add_argument("--attack-epochs", type=int, default=10)
    p.add_argument("--max-batches", type=int, default=0, help="batches per round; 0 = all (only for a quick test)")
    p.add_argument("--code-dir", default=".")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--timeout", type=float, default=6 * 3600)
    a = p.parse_args()

    run_dir = Path(a.out) / a.name
    if run_dir.exists():
        shutil.move(str(run_dir), str(run_dir) + "_old_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    (run_dir / "logs").mkdir(parents=True)
    checkpoint = run_dir / "shared_checkpoint.pt"
    split = a.start_layer
    rows = []
    mode = "static" if a.static else f"adaptive, rho_min={a.rho_min}"
    print(f"== {a.name}: SFL ResNet-18 {a.dataset} K={a.num_clients} alpha={a.alpha} "
          f"start={split} ({mode}), {a.rounds} rounds", flush=True)
    for r in range(1, a.rounds + 1):
        m = train_round(a, split, checkpoint, run_dir / "logs" / f"round{r:02d}_{split}.log")
        leak = measure(a, checkpoint)
        nxt, rule = split, ""
        i = RESNET_SPLIT_POINTS.index(split)
        if not a.static and leak["rho"] < a.rho_min and i + 1 < len(RESNET_SPLIT_POINTS):
            nxt, rule = RESNET_SPLIT_POINTS[i + 1], "privacy"
            resplit_checkpoint(checkpoint, nxt)
        rows.append({"name": a.name, "round": r, "split": split, **m, **leak,
                     "rho_min": "" if a.static else a.rho_min, "next_split": nxt, "rule": rule})
        write_csv(run_dir / "rounds.csv", rows)
        print(f"  r{r:02d} {split}: acc={m['acc']:.2f}% round={m['total']:.0f}s "
              f"mse={leak['attack_mse']:.4f} rho={leak['rho']:.3f} dDcor={leak['delta_dcor']:.3f}"
              + (f" -> {nxt} [privacy: rho < {a.rho_min}]" if rule else ""), flush=True)
        split = nxt
    print(f"done -> {run_dir / 'rounds.csv'}")


if __name__ == "__main__":
    main()
