"""
Measure cut-layer leakage for every round-10 checkpoint of a baseline folder.

For each checkpoint  <OUT>/checkpoints/<paradigm>_resnet18_<dataset>_K<K>_c<c>_a<alpha>_r<rep>_round<R>.pt
the script loads the client-side model (client 0 by default), and on the test split computes

  * dCor(X, Z): mean and std over --dcor-batches batches of --batch-size samples
  * a permutation baseline dCor(X, Z_perm), where the samples of Z are shuffled within
    each batch so that Z is independent of X; delta_dcor = dcor - dcor_perm
  * optionally the reconstruction-attack MSE (same decoder attack as
    measure_cutlayer_leakage.py)

One row per checkpoint is appended to --csv. Rows already in the CSV are skipped,
so the script can be restarted after an interruption.

Example:
  python measure_all_leakage.py --results baseline_results_mnist_resnet_cloud \
      --dataset mnist --csv leakage/leakage_mnist.csv --run-reconstruction-attack
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from pathlib import Path

import torch

import measure_cutlayer_leakage as mcl
from dcor_privacy import dist_corr, SimpleDecoder, ReconstructionAttackTester
from torch.utils.data import DataLoader, TensorDataset
from split_learning_utils import DATASET_SPECS
from split_learning_utils import set_seed, get_model_classes, normalize_model_config

NAME_RE = re.compile(
    r"^(?P<paradigm>fl|sl|sfl)_(?P<model>resnet\d+|cnn)_(?P<dataset>[a-z0-9]+)"
    r"_K(?P<K>\d+)_c(?P<cpus>\d+)_a(?P<alpha>[\d.]+)_r(?P<rep>\d+)_round(?P<round>\d+)\.pt$"
)


@torch.no_grad()
def dcor_with_baseline(client_model, loader, device, num_batches, seed=0):
    gen = torch.Generator().manual_seed(seed)
    raw, perm = [], []
    z_shape = None
    n = 0
    for i, (x, _y) in enumerate(loader):
        if i >= num_batches:
            break
        x = x.to(device)
        z = client_model(x)
        raw.append(float(dist_corr(x, z).item()))
        idx = torch.randperm(z.shape[0], generator=gen).to(z.device)
        perm.append(float(dist_corr(x, z[idx]).item()))
        z_shape = tuple(int(d) for d in z.shape[1:])
        bytes_per_sample = int(z[0].numel() * z.element_size())
        n += int(x.shape[0])
    mean = lambda v: sum(v) / len(v)
    std = lambda v: (sum((a - mean(v)) ** 2 for a in v) / len(v)) ** 0.5
    return {
        "dcor_mean": mean(raw), "dcor_std": std(raw),
        "dcor_perm_mean": mean(perm), "dcor_perm_std": std(perm),
        "delta_dcor": mean(raw) - mean(perm),
        "batches": len(raw), "samples": n,
        "z_shape": "x".join(map(str, z_shape)), "z_dim": int(torch.tensor(z_shape).prod()),
        "z_bytes_per_sample": bytes_per_sample,
    }, z_shape


def use_npz_testset(path):
    """Replace the CIFAR-10 test split by the images exported in cifar10_test.npz."""
    import numpy as np
    from torch.utils.data import TensorDataset
    from split_learning_utils import DATASET_SPECS
    d = np.load(path)
    x = torch.tensor(d["data"], dtype=torch.float32).reshape(-1, 3, 32, 32) / 255.0
    spec = DATASET_SPECS["cifar10"]
    mean = torch.tensor(spec["normalize_mean"]).view(1, 3, 1, 1)
    std = torch.tensor(spec["normalize_std"]).view(1, 3, 1, 1)
    ds = TensorDataset((x - mean) / std, torch.tensor(d["labels"]))
    orig = mcl.load_dataset
    mcl.load_dataset = lambda name, train: ds if (name == "cifar10" and not train) else orig(name, train)


@torch.no_grad()
def _pairs(model, loader, device, mean, std):
    zs, xs = [], []
    for x, _y in loader:
        zs.append(model(x.to(device)).cpu())
        xs.append((x * std + mean).clamp(0, 1))      # target in pixel space [0, 1]
    return torch.cat(zs), torch.cat(xs)


def reconstruction_attack(model, dataset, batch_size, device, train_samples, eval_samples, epochs, lr, z_shape):
    """Decoder attack with the target in pixel space [0, 1], matching the sigmoid output of
    SimpleDecoder. (measure_cutlayer_leakage.py compares the sigmoid output with the
    *normalized* image, which no decoder can reach, so its MSE is dominated by that offset.)"""
    spec = DATASET_SPECS[dataset]
    c = spec["input_channels"]
    mean = torch.tensor(spec["normalize_mean"]).view(1, c, 1, 1)
    std = torch.tensor(spec["normalize_std"]).view(1, c, 1, 1)
    tr = mcl.make_subset_loader(dataset, batch_size, 0, train_samples)
    ev = mcl.make_subset_loader(dataset, batch_size, train_samples, eval_samples)
    ztr, xtr = _pairs(model, tr, device, mean, std)
    zev, xev = _pairs(model, ev, device, mean, std)
    torch.manual_seed(0)
    dec = SimpleDecoder(in_channels=int(z_shape[0]), out_channels=c, out_size=28 if dataset == "mnist" else 32)
    t = ReconstructionAttackTester(dec, device=str(device), lr=lr)
    t.fit(DataLoader(TensorDataset(ztr, xtr), batch_size=batch_size, shuffle=True), epochs=epochs)
    mse = t.evaluate(DataLoader(TensorDataset(zev, xev), batch_size=batch_size))
    base = float(((xev - xtr.mean(0, keepdim=True)) ** 2).mean())   # attacker that outputs the mean image
    return {"attack_mse": mse, "attack_psnr": 10 * math.log10(1 / max(mse, 1e-12)),
            "attack_mse_mean_image": base, "attack_train_samples": len(xtr),
            "attack_eval_samples": len(xev), "attack_epochs": epochs}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, help="baseline output folder that contains checkpoints/")
    p.add_argument("--dataset", required=True, choices=["mnist", "cifar10"])
    p.add_argument("--csv", required=True)
    p.add_argument("--round", type=int, default=10, help="checkpoint round to use")
    p.add_argument("--paradigms", default="fl,sl,sfl")
    p.add_argument("--split-after", default="layer2", help="expected ResNet cut layer (ignored for the CNN)")
    p.add_argument("--client-id", type=int, default=0, help="-1 = sample-weighted average of the client models")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dcor-batches", type=int, default=10)
    p.add_argument("--run-reconstruction-attack", action="store_true")
    p.add_argument("--attack-train-samples", type=int, default=512)
    p.add_argument("--attack-eval-samples", type=int, default=256)
    p.add_argument("--attack-epochs", type=int, default=10)
    p.add_argument("--attack-lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--cifar-npz", default="", help="use cifar10_test.npz (from export_leakage_inputs.py) instead of ./data")
    a = p.parse_args()

    if a.cifar_npz:
        use_npz_testset(a.cifar_npz)
    ckdir = Path(a.results) / "checkpoints"
    files = sorted(ckdir.glob("*.pt"))
    jobs = []
    for f in files:
        m = NAME_RE.match(f.name)
        if not m or m["dataset"] != a.dataset or int(m["round"]) != a.round:
            continue
        if m["paradigm"] not in a.paradigms.split(","):
            continue
        jobs.append((f, m.groupdict()))
    print(f"{len(jobs)} checkpoints in {ckdir} for dataset={a.dataset}, round={a.round}")
    if not jobs:
        sys.exit("nothing to measure (check --results / --dataset / --round)")

    out = Path(a.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with out.open() as fh:
            done = {r["checkpoint"] for r in csv.DictReader(fh)}

    device = mcl.resolve_device(a.device)
    fields = None
    for i, (f, g) in enumerate(jobs, 1):
        if f.name in done:
            print(f"[{i}/{len(jobs)}] {f.name}: already in CSV, skipped")
            continue
        t0 = time.time()
        set_seed()
        try:
            ck = torch.load(f, map_location="cpu", weights_only=False)
            cfg = normalize_model_config((ck.get("partition") or {}).get("model"))
            if g["model"] != "cnn" and cfg["split_after"] != a.split_after:
                raise ValueError(f"checkpoint is cut after {cfg['split_after']}, expected {a.split_after}")
            model = get_model_classes(a.dataset, cfg)[0]()
            state, state_label = mcl.select_client_state(ck, a.dataset, int(g["K"]), float(g["alpha"]), a.client_id)
            model.load_state_dict(state, strict=True)
            model = model.to(device).eval()
            for prm in model.parameters():
                prm.requires_grad_(False)
        except Exception as e:  # keep going, record the error
            print(f"[{i}/{len(jobs)}] {f.name}: ERROR {e}")
            continue
        loader = mcl.make_subset_loader(a.dataset, a.batch_size, 0, a.dcor_batches * a.batch_size)
        res, z_shape = dcor_with_baseline(model, loader, device, a.dcor_batches)
        row = {"checkpoint": f.name, "paradigm": g["paradigm"], "dataset": a.dataset,
               "num_clients": int(g["K"]), "alpha": float(g["alpha"]), "repeat": int(g["rep"]),
               "round": int(g["round"]), "split_after": cfg["split_after"], "weights": state_label, **res}
        if a.run_reconstruction_attack:
            row.update(reconstruction_attack(model, a.dataset, a.batch_size, device, a.attack_train_samples,
                                             a.attack_eval_samples, a.attack_epochs, a.attack_lr, z_shape))
        row["seconds"] = round(time.time() - t0, 1)
        new = not out.exists() or out.stat().st_size == 0
        fields = fields or list(row)
        with out.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow(row)
        msg = f"dcor={res['dcor_mean']:.4f} perm={res['dcor_perm_mean']:.4f} delta={res['delta_dcor']:.4f}"
        if "attack_mse" in row:
            msg += f" mse={row['attack_mse']:.4f} (mean-image {row['attack_mse_mean_image']:.4f})"
        print(f"[{i}/{len(jobs)}] {f.name}: {msg} ({row['seconds']}s)", flush=True)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
