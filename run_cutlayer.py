"""
Cut-layer experiment: how the position of the cut changes privacy leakage (Section 4.1, "Cut Layer
and Privacy Leakage").

For every cut layer (layer1..layer4 of ResNet-18) it
  1. trains SFL (default; --paradigms sl,sfl for both) with run_baseline.py (same settings as the baseline: K clients, IID,
     1 CPU per client, --rounds rounds), one output folder per cut layer:
         <out>/<dataset>_<layer>/   (runs.jsonl, logs/, checkpoints/, runs.csv ...)
  2. measures the leakage of the client-side model of every final checkpoint with
     measure_all_leakage.py: dCor(X, Z), the permutation baseline, delta dCor, and the
     reconstruction attack (MSE / PSNR, and the MSE of an attacker that outputs the mean image)
  3. measures the same indicators on an untrained client model (random initialization) of the
     same cut layer, as a reference ("init" rows)
  4. writes one merged table:  <out>/cutlayer_<dataset>.csv

Every step skips what is already done, so the script can be restarted after an interruption.

Run from the repository folder (where run_baseline.py and the entry programs are), e.g.
  venv/bin/python run_cutlayer.py --dataset cifar10 --rounds 20
  venv/bin/python run_cutlayer.py --dataset cifar10 --rounds 20 --step measure     # only measure
Extra options after "--" are passed to "run_baseline.py run", e.g.
  venv/bin/python run_cutlayer.py --dataset cifar10 -- --max-workers 2
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

import run_baseline as rb


def layer_dir(out: Path, dataset: str, layer: str) -> Path:
    return out / f"{dataset}_{layer}"


def configure_baseline(a, layer: str) -> None:
    rb.GRID = {
        "paradigm": a.paradigms.split(","),
        "model": ["resnet18"],
        "dataset": [a.dataset],
        "num_clients": [a.num_clients],
        "cpus": [1],
        "alpha": [a.alpha],
    }
    rb.ROUNDS = [a.rounds]
    rb.REPEATS = 1
    rb.SPLIT_AFTER = layer
    rb.MAX_BATCHES = a.max_batches


def call_baseline(argv: list[str]) -> None:
    saved = sys.argv
    sys.argv = ["run_baseline.py", *argv]
    try:
        rb.main()
    finally:
        sys.argv = saved


def step_train(a, extra: list[str]) -> None:
    for layer in a.layers.split(","):
        d = layer_dir(a.out, a.dataset, layer)
        configure_baseline(a, layer)
        print(f"\n=== train {a.dataset} cut after {layer} -> {d}", flush=True)
        call_baseline(["--out", str(d), "run", *extra])
        call_baseline(["--out", str(d), "summarize"])


def step_measure(a) -> None:
    import measure_all_leakage as mal

    for layer in a.layers.split(","):
        d = layer_dir(a.out, a.dataset, layer)
        if not (d / "checkpoints").exists():
            print(f"[measure] {d}/checkpoints missing, skipped")
            continue
        argv = ["measure_all_leakage.py", "--results", str(d), "--dataset", a.dataset,
                "--csv", str(d / "leakage.csv"), "--round", str(a.rounds),
                "--paradigms", a.paradigms, "--split-after", layer,
                "--client-id", str(a.client_id), "--dcor-batches", str(a.dcor_batches),
                "--attack-epochs", str(a.attack_epochs), "--run-reconstruction-attack"]
        print(f"\n=== measure {d}", flush=True)
        saved = sys.argv
        sys.argv = argv
        try:
            mal.main()
        except SystemExit as e:  # "nothing to measure" etc.
            print(f"[measure] {d}: {e}")
        finally:
            sys.argv = saved


def step_init_reference(a) -> None:
    """Leakage of an untrained client model with the same cut layer."""
    import measure_all_leakage as mal
    import measure_cutlayer_leakage as mcl
    from split_learning_utils import get_model_classes, normalize_resnet_config, set_seed

    out = a.out / f"init_{a.dataset}.csv"
    done = set()
    if out.exists():
        with out.open() as fh:
            done = {r["split_after"] for r in csv.DictReader(fh)}
    for layer in a.layers.split(","):
        if layer in done:
            continue
        t0 = time.time()
        set_seed()
        cfg = normalize_resnet_config(resnet_depth=18, resnet_split_after=layer)
        model = get_model_classes(a.dataset, cfg)[0]().eval()
        for p in model.parameters():
            p.requires_grad_(False)
        loader = mcl.make_subset_loader(a.dataset, 64, 0, a.dcor_batches * 64)
        res, z_shape = mal.dcor_with_baseline(model, loader, torch.device("cpu"), a.dcor_batches)
        row = {"checkpoint": "init", "paradigm": "init", "dataset": a.dataset, "split_after": layer,
               **res, **mal.reconstruction_attack(model, a.dataset, 64, torch.device("cpu"), 512, 256,
                                                  a.attack_epochs, 1e-3, z_shape)}
        row["seconds"] = round(time.time() - t0, 1)
        new = not out.exists()
        with out.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"[init] {layer}: dcor={res['dcor_mean']:.4f} delta={res['delta_dcor']:.4f} "
              f"mse={row['attack_mse']:.4f}", flush=True)


def client_params(dataset: str, layer: str) -> int:
    from split_learning_utils import get_model_classes, normalize_resnet_config

    cfg = normalize_resnet_config(resnet_depth=18, resnet_split_after=layer)
    return sum(p.numel() for p in get_model_classes(dataset, cfg)[0]().parameters())


def step_merge(a) -> None:
    cols = ["dataset", "split_after", "paradigm", "round", "final_acc", "client_params",
            "z_shape", "z_dim", "z_bytes_per_sample", "dcor_mean", "dcor_std", "dcor_perm_mean",
            "delta_dcor", "attack_mse", "attack_psnr", "attack_mse_mean_image", "train_s",
            "wall_clock_s", "checkpoint"]
    rows = []
    for layer in a.layers.split(","):
        d = layer_dir(a.out, a.dataset, layer)
        acc = {}
        if (d / "runs.csv").exists():
            with (d / "runs.csv").open() as fh:
                for r in csv.DictReader(fh):
                    acc[r["paradigm"]] = r
        if (d / "leakage.csv").exists():
            with (d / "leakage.csv").open() as fh:
                for r in csv.DictReader(fh):
                    run = acc.get(r["paradigm"], {})
                    rows.append({**r, "final_acc": run.get("final_acc", ""),
                                 "train_s": run.get("training_time", run.get("train_time", "")),
                                 "wall_clock_s": run.get("wall_clock_s", ""),
                                 "client_params": client_params(a.dataset, layer)})
    init = a.out / f"init_{a.dataset}.csv"
    if init.exists():
        with init.open() as fh:
            for r in csv.DictReader(fh):
                rows.append({**r, "round": 0, "client_params": client_params(a.dataset, r["split_after"])})
    order = {l: i for i, l in enumerate(a.layers.split(","))}
    rows.sort(key=lambda r: (order.get(r["split_after"], 9), r["paradigm"]))
    out = a.out / f"cutlayer_{a.dataset}.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} rows -> {out}")
    for r in rows:
        print(f"  {r['split_after']:7s} {r['paradigm']:5s} acc={r.get('final_acc', '') or '-':>6} "
              f"dcor={float(r['dcor_mean']):.3f} delta={float(r['delta_dcor']):.3f} "
              f"mse={float(r['attack_mse']):.4f} psnr={float(r['attack_psnr']):.2f}")


def main() -> None:
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["mnist", "cifar10"], default="cifar10")
    p.add_argument("--layers", default="layer1,layer2,layer3,layer4")
    p.add_argument("--paradigms", default="sfl")
    p.add_argument("--num-clients", type=int, default=3)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--client-id", type=int, default=0)
    p.add_argument("--dcor-batches", type=int, default=10)
    p.add_argument("--attack-epochs", type=int, default=10)
    p.add_argument("--max-batches", type=int, default=0, help="SL/SFL batches per round; 0 = all (only for a quick test)")
    p.add_argument("--out", type=Path, default=Path("cutlayer_results"))
    p.add_argument("--step", choices=["all", "train", "measure", "merge"], default="all")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.step in ("all", "train"):
        step_train(a, extra)
    if a.step in ("all", "measure"):
        step_measure(a)
        step_init_reference(a)
    if a.step in ("all", "measure", "merge"):
        step_merge(a)


if __name__ == "__main__":
    main()
