"""Run the full baseline grid (FL / SL / SFL, no switching) and record every result.

Usage
-----
  python run_baseline.py plan                 # show the grid and how many runs it needs
  python run_baseline.py run                  # run everything (resumable: finished runs are skipped)
  python run_baseline.py run --max-workers 2  # run local experiments in parallel
  python run_baseline.py run --only-paradigm fl --limit 3      # partial runs, e.g. for a smoke test
  python run_baseline.py parse logs/<run>.log # test the log parser on one existing log
  python run_baseline.py summarize            # build the CSV tables used in Chapter 4

Everything the script produces goes to ./baseline_results_mnist_cnn/ by default:
  runs.jsonl                 one JSON record per executed run
  logs/<run_id>.log          full stdout/stderr of every staged run
  checkpoints/<run_id>_round010.pt
                             retained model checkpoint at the final reported round cut
  runs.csv                   one row per run (final metrics)
  rounds.csv                 one row per run and round (per-round metrics)
  baseline_table.csv         one row per logical configuration
  best_paradigm.csv          preferred paradigm P*(c) per configuration

Adjust the GRID and the command-building section below to match your code.
Places that must be checked against your code are marked with  # CHECK
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import itertools
import json
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# =====================================================================
# 1. Experimental grid  (Table 4.1)
# =====================================================================
GRID = {
    "paradigm":    ["fl", "sl", "sfl"],
    "model":       ["cnn"],
    "dataset":     ["mnist"],                     # CHECK: must match DATASET_CHOICES
    "num_clients": [3, 5, 10],
    "cpus":        [1],
    "alpha":       [0.1, 0.5, 0.9, 1.0],          # 1.0 = IID in your code
}

# Logical factors that are derived instead of executed (Section 4.1, "Experimental grid").
ROUNDS = [5, 10]                 # checkpoints are retained at each listed round
NETWORK = {
    "stable": "0",
}
REPEATS = 1                      # repetitions of each executed run, to measure timing noise
SPLIT_AFTER = "layer2"           # same cut layer for SL and SFL, so they differ only in aggregation
LOCAL_EPOCHS = 1
MAX_BATCHES = 0                  # SL/SFL only; 0 = full local dataset
TIMEOUT_S = 6 * 3600             # a run that takes longer is killed and marked "timeout"
COOLDOWN_S = 10                  # pause between runs so Ray fully shuts down

# Entry programs (paths relative to --code-dir)
SCRIPTS = {"fl": "fl_mnist_minimal.py", "sl": "sl_mnist_minimal.py", "sfl": "fsl_mnist_minimal.py"}

# How each model name is passed on the command line.
# CHECK: use the flag names defined in add_resnet_model_args() in split_learning_utils.py
MODEL_ARGS = {
    "cnn": ["--model", "cnn", "--cnn-split-after", "conv1"],
    "resnet18": ["--model", "resnet", "--resnet-depth", "18"],
    "resnet34": ["--model", "resnet", "--resnet-depth", "34"],
}

OUT = Path("baseline_results_mnist_cnn")
CHECKPOINTS_DIRNAME = "checkpoints"


# =====================================================================
# 2. Command construction
# =====================================================================
def build_command(cfg, python=sys.executable, num_rounds=None, checkpoint_path=None):
    n_rounds = max(ROUNDS) if num_rounds is None else num_rounds
    cmd = [python, SCRIPTS[cfg["paradigm"]],
           "--num-clients", str(cfg["num_clients"]),
           "--num-rounds", str(n_rounds),
           "--local-epochs", str(LOCAL_EPOCHS),
           "--client-num-cpus", str(cfg["cpus"]),
           "--dataset", cfg["dataset"],
           "--noniid-alpha", str(cfg["alpha"]),
           "--communication-delay", "0",
           *MODEL_ARGS[cfg["model"]]]
    if cfg["model"].startswith("resnet"):
        cmd += ["--resnet-split-after", SPLIT_AFTER]
    if checkpoint_path is not None:
        cmd += ["--checkpoint-path", str(checkpoint_path)]
    if cfg["paradigm"] in ("sl", "sfl"):
        cmd += ["--eval-every-round"]
        if MAX_BATCHES:
            cmd += ["--max-batches", str(MAX_BATCHES)]
    # FL evaluates every round by default (client-side, see note in summarize()).
    return cmd


def all_configs():
    keys = list(GRID)
    for values in itertools.product(*(GRID[k] for k in keys)):
        cfg = dict(zip(keys, values))
        for rep in range(REPEATS):
            yield {**cfg, "repeat": rep}


def run_id(cfg):
    base = "{paradigm}_{model}_{dataset}_K{num_clients}_c{cpus}_a{alpha}_r{repeat}".format(**cfg)
    return base


def checkpoint_path_for(rid, round_count):
    checkpoints = OUT / CHECKPOINTS_DIRNAME
    return checkpoints / f"{rid}_round{round_count:03d}.pt"


# =====================================================================
# 3. Log parsing
# =====================================================================
RE_ROUND = re.compile(r"\bRound\s+(\d+)\b", re.I)
RE_STAGE_MARKER = re.compile(r"^# Stage to global round\s+(\d+):.*$", re.I | re.M)
RE_LOSS = re.compile(r"test\s+loss\s*[:=]\s*([-+\d.eE]+)", re.I)
RE_ACC = re.compile(r"test\s+acc(?:uracy)?\s*[:=]\s*([-+\d.eE]+)\s*(%?)", re.I)
RE_STATS = re.compile(r"(Round\s+(\d+)|Total)\s+runtime stats:\s*(.*)", re.I)
RE_KV = re.compile(r"(\w+)=([-+\d.eE]+)s?")


def parse_log(text):
    """Extract per-round and final metrics from the stdout of one run.

    Expected lines (as printed by fl_mnist_minimal.py / print_test_metrics):
      ========== Round 3 ==========
      Average test loss: 0.1234        (any label; 'Final ...' lines are the final metrics)
      Average test acc:  97.12%
      Round 3 runtime stats: total_runtime=...s communication_time=...s delay_time=...s training_time=...s
      Final test loss / Final test acc
      Total runtime stats: ...
    """
    stage_matches = list(RE_STAGE_MARKER.finditer(text))
    if stage_matches:
        stages = []
        completed = 0
        for index, match in enumerate(stage_matches):
            target_round = int(match.group(1))
            start = match.end()
            end = stage_matches[index + 1].start() if index + 1 < len(stage_matches) else len(text)
            stages.append({
                "rounds": target_round - completed,
                "parsed": parse_log(text[start:end]),
            })
            completed = target_round
        return combine_stage_metrics(stages)

    rounds, final, total_stats = {}, {}, {}
    current = None
    for line in text.splitlines():
        m = RE_STATS.search(line)
        if m:
            stats = {k: float(v) for k, v in RE_KV.findall(m.group(3))}
            if m.group(2):
                rounds.setdefault(int(m.group(2)), {}).update(stats)
            else:
                total_stats = stats
            continue
        is_final = line.strip().lower().startswith("final")
        m = RE_ROUND.search(line)
        if m and not is_final:
            current = int(m.group(1))
        target = final if is_final else (rounds.setdefault(current, {}) if current else None)
        if target is None:
            continue
        m = RE_LOSS.search(line)
        if m:
            target["loss"] = float(m.group(1))
        m = RE_ACC.search(line)
        if m:
            acc = float(m.group(1))
            target["acc"] = acc if m.group(2) == "%" or acc > 1.0 else acc * 100.0
    return {
        "rounds": [{"round": r, **v} for r, v in sorted(rounds.items())],
        "final": final,
        "total": total_stats,
    }


# =====================================================================
# 4. Running
# =====================================================================
def load_done():
    done = {}
    path = OUT / "runs.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["run_id"]] = rec
    return done


def env_info(code_dir):
    info = {"host": platform.node(), "python": platform.python_version(),
            "os": platform.platform(), "cpu_count": os.cpu_count()}
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=code_dir,
            capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        info["git_commit"] = None
    return info


def combine_stage_metrics(stages):
    rounds, total_stats = [], {}
    final = {}
    completed = 0
    for stage in stages:
        parsed = stage["parsed"]
        for row in parsed["rounds"]:
            local_round = row.get("round")
            if local_round is None:
                continue
            shifted = dict(row)
            shifted["round"] = completed + local_round
            rounds.append(shifted)
        for key, value in parsed["total"].items():
            total_stats[key] = total_stats.get(key, 0.0) + value
        if parsed["final"]:
            final = parsed["final"]
        completed += stage["rounds"]
    return {
        "rounds": sorted(rounds, key=lambda row: row["round"]),
        "final": final,
        "total": total_stats,
    }


def run_staged_command(cfg, rid, args, code_dir, log_path):
    checkpoint_path = (OUT / CHECKPOINTS_DIRNAME / f".{rid}_{os.getpid()}_working.pt")
    stages, commands = [], []
    completed_rounds = 0
    retained_round = max(ROUNDS)
    status, rc = "ok", None

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "w") as log:
            for target_round in sorted(ROUNDS):
                stage_rounds = target_round - completed_rounds
                if stage_rounds <= 0:
                    continue

                cmd = build_command(
                    cfg,
                    python=args.python,
                    num_rounds=stage_rounds,
                    checkpoint_path=checkpoint_path,
                )
                commands.append(cmd)
                log.write(
                    f"# Stage to global round {target_round}: "
                    + " ".join(str(part) for part in cmd)
                    + "\n"
                )
                log.flush()

                stage_start = time.perf_counter()
                try:
                    proc = subprocess.run(
                        cmd,
                        cwd=code_dir,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=TIMEOUT_S,
                        text=True,
                    )
                    rc = proc.returncode
                    if rc != 0:
                        status = "failed"
                except subprocess.TimeoutExpired:
                    status = "timeout"

                stage_wall = time.perf_counter() - stage_start
                log.flush()

                stage_text = log_path.read_text(errors="replace").split(
                    f"# Stage to global round {target_round}: ",
                    1,
                )[-1]
                parsed = parse_log(stage_text)
                stages.append({
                    "target_round": target_round,
                    "rounds": stage_rounds,
                    "wall_clock_s": round(stage_wall, 3),
                    "parsed": parsed,
                })

                if (
                    status == "ok"
                    and target_round == retained_round
                    and checkpoint_path.exists()
                ):
                    shutil.copy2(checkpoint_path, checkpoint_path_for(rid, target_round))

                if status != "ok":
                    break
                if not parsed["final"]:
                    status = "no_metrics"
                    break

                completed_rounds = target_round
    finally:
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    return status, rc, commands, combine_stage_metrics(stages)


def execute_run(index, total, cfg, args, code_dir, info):
    rid = run_id(cfg)
    log_path = OUT / "logs" / f"{rid}.log"
    start = time.perf_counter()
    status, rc, commands, parsed = run_staged_command(cfg, rid, args, code_dir, log_path)
    wall = time.perf_counter() - start
    if status == "ok" and not parsed["final"]:
        status = "no_metrics"          # finished but the parser found nothing: check the log
    checkpoint_paths = {str(max(ROUNDS)): str(checkpoint_path_for(rid, max(ROUNDS)))}
    rec = {"run_id": rid, "config": cfg, "command": commands, "status": status,
           "checkpoint_paths": checkpoint_paths,
           "returncode": rc, "wall_clock_s": round(wall, 3),
           "finished_at": datetime.now().isoformat(timespec="seconds"),
           "env": info, **parsed}
    return index, total, rid, wall, parsed["final"].get("acc"), rec


def cmd_run(args):
    code_dir = Path(args.code_dir).resolve()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / CHECKPOINTS_DIRNAME).mkdir(parents=True, exist_ok=True)
    done = load_done()
    todo = [c for c in all_configs()
            if (not args.only_paradigm or c["paradigm"] == args.only_paradigm)
            and not (run_id(c) in done
                     and (done[run_id(c)]["status"] == "ok" or not args.retry_failed))]
    if args.shuffle:
        # Random order spreads slow drifts (thermal throttling, background load)
        # over all configurations instead of biasing one paradigm.
        random.Random(args.shuffle_seed).shuffle(todo)
    if args.limit:
        todo = todo[: args.limit]
    info = env_info(code_dir)
    max_workers = max(1, int(args.max_workers))
    max_workers = min(max_workers, len(todo) or 1)
    print(
        f"{len(todo)} runs to execute ({len(done)} already recorded), "
        f"max_workers={max_workers}.",
        flush=True,
    )

    def write_record(rec):
        with open(OUT / "runs.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    if max_workers == 1:
        for i, cfg in enumerate(todo, 1):
            rid = run_id(cfg)
            print(f"[{i}/{len(todo)}] {rid}", flush=True)
            _index, _total, _rid, wall, acc, rec = execute_run(i, len(todo), cfg, args, code_dir, info)
            write_record(rec)
            print(f"    -> {rec['status']}, {wall:.1f}s, final acc={acc}", flush=True)
            if i < len(todo):
                time.sleep(COOLDOWN_S)
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, cfg in enumerate(todo, 1):
            rid = run_id(cfg)
            print(f"[{i}/{len(todo)}] started {rid}", flush=True)
            futures.append(executor.submit(execute_run, i, len(todo), cfg, args, code_dir, info))

        for future in as_completed(futures):
            i, total, rid, wall, acc, rec = future.result()
            write_record(rec)
            print(f"[{i}/{total}] finished {rid} -> {rec['status']}, {wall:.1f}s, final acc={acc}", flush=True)


def cmd_plan(_args):
    executed = 1
    for k, v in GRID.items():
        executed *= len(v)
    logical = executed * len(ROUNDS) * len(NETWORK)
    print("Grid:")
    for k, v in GRID.items():
        print(f"  {k:12s} {v}")
    print(f"  {'rounds':12s} {ROUNDS} (only round {max(ROUNDS)} checkpoint is retained)")
    print(f"  {'network':12s} {list(NETWORK)} (reported network conditions)")
    print(f"\nLogical configurations : {logical}")
    print(f"Executed configurations: {executed}  x {REPEATS} repeats = {executed * REPEATS} staged runs")
    example = next(all_configs())
    example_checkpoint = OUT / CHECKPOINTS_DIRNAME / f".{run_id(example)}_<pid>_working.pt"
    print(
        "\nExample first-stage command:\n  "
        + " ".join(
            build_command(
                example,
                num_rounds=ROUNDS[0],
                checkpoint_path=example_checkpoint,
            )
        )
    )


def cmd_parse(args):
    print(json.dumps(parse_log(Path(args.log).read_text(errors="replace")), indent=2))


# =====================================================================
# 5. Summaries for Chapter 4
# =====================================================================
FACTORS = ["paradigm", "model", "dataset", "num_clients", "cpus", "alpha"]
TIME_KEYS = ["total_runtime", "training_time", "communication_time"]


def delays_for(schedule, n_rounds):
    vals = [float(x) for x in schedule.split(",") if x.strip()] or [0.0]
    return [vals[min(r, len(vals) - 1)] for r in range(n_rounds)]


def write_csv(path, rows):
    import csv
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_excel(path, sheets):
    try:
        import pandas as pd
    except ImportError:
        print("Skipped Excel export: pandas/openpyxl is not installed.")
        return

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, rows in sheets.items():
            if rows:
                pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name, index=False)


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def cmd_summarize(args):
    records = [r for r in load_done().values() if r["status"] == "ok"]
    runs, rounds_rows, logical = [], [], {}

    for rec in records:
        cfg = rec["config"]
        runs.append({"run_id": rec["run_id"], **cfg,
                     "final_acc": rec["final"].get("acc"), "final_loss": rec["final"].get("loss"),
                     **{k: rec["total"].get(k) for k in TIME_KEYS},
                     "wall_clock_s": rec["wall_clock_s"]})
        per_round = {r["round"]: r for r in rec["rounds"]}
        for r in rec["rounds"]:
            rounds_rows.append({"run_id": rec["run_id"], **cfg, **r})

        for R in ROUNDS:
            last = per_round.get(R, {})
            if R == max(ROUNDS):
                acc = rec["final"].get("acc", last.get("acc"))
                loss = rec["final"].get("loss", last.get("loss"))
                times = {k: rec["total"].get(k) for k in TIME_KEYS}
            else:
                acc, loss = last.get("acc"), last.get("loss")
                # cumulative time of the first R rounds; None if the program does not print per-round stats
                times = {}
                for k in TIME_KEYS:
                    vals = [per_round.get(i, {}).get(k) for i in range(1, R + 1)]
                    times[k] = sum(vals) if all(v is not None for v in vals) else None
            for net, schedule in NETWORK.items():
                extra = sum(delays_for(schedule, R))
                t = dict(times)
                for k in ("total_runtime", "communication_time"):
                    if t.get(k) is not None:
                        t[k] = t[k] + extra
                key = tuple(cfg[f] for f in FACTORS) + (R, net)
                logical.setdefault(key, []).append({"acc": acc, "loss": loss, **t})

    table = []
    for key, reps in sorted(logical.items(), key=lambda kv: str(kv[0])):
        row = dict(zip(FACTORS + ["rounds", "network"], key))
        row["repeats"] = len(reps)
        for m in ["acc", "loss"] + TIME_KEYS:
            mu, sd = mean_std([r.get(m) for r in reps])
            row[f"{m}_mean"], row[f"{m}_std"] = mu, sd
        table.append(row)

    # Preferred paradigm P*(c): fastest among paradigms within EPS of the best accuracy.
    eps = args.eps
    groups = {}
    for row in table:
        c = tuple(row[f] for f in FACTORS[1:] + ["rounds", "network"])
        groups.setdefault(c, []).append(row)
    best = []
    for c, rows in groups.items():
        rows = [r for r in rows if r["acc_mean"] is not None]
        if not rows:
            continue
        top = max(r["acc_mean"] for r in rows)
        cand = [r for r in rows if r["acc_mean"] >= top - eps and r["total_runtime_mean"] is not None]
        winner = min(cand, key=lambda r: r["total_runtime_mean"]) if cand else None
        best.append({**dict(zip(FACTORS[1:] + ["rounds", "network"], c)),
                     "best_paradigm": winner["paradigm"] if winner else None,
                     **{f"acc_{r['paradigm']}": r["acc_mean"] for r in rows},
                     **{f"time_{r['paradigm']}": r["total_runtime_mean"] for r in rows},
                     "paradigms_compared": len(rows)})

    write_csv(OUT / "runs.csv", runs)
    write_csv(OUT / "rounds.csv", rounds_rows)
    write_csv(OUT / "baseline_table.csv", table)
    write_csv(OUT / "best_paradigm.csv", best)
    write_excel(
        OUT / "baseline_results.xlsx",
        {
            "runs": runs,
            "rounds": rounds_rows,
            "baseline_table": table,
            "best_paradigm": best,
        },
    )
    n_all = len(load_done())
    print(f"Summarized {len(records)} successful runs ({n_all - len(records)} failed/timeout/no_metrics).")
    print(f"Wrote runs.csv, rounds.csv, baseline_table.csv ({len(table)} rows), "
          f"best_paradigm.csv ({len(best)} rows), baseline_results.xlsx in {OUT}/")


# =====================================================================
def main():
    global OUT
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(OUT), help=f"output folder (default: {OUT})")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    r = sub.add_parser("run")
    r.add_argument("--code-dir", default=".", help="folder containing the three entry programs")
    r.add_argument("--python", default=sys.executable)
    r.add_argument("--only-paradigm", choices=list(SCRIPTS))
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--retry-failed", action="store_true",
                   help="re-run runs recorded as failed/timeout/no_metrics (default: skip them)")
    r.add_argument("--max-workers", type=int, default=1,
                   help="number of runs to execute concurrently on this machine")
    r.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    r.add_argument("--shuffle-seed", type=int, default=0)
    q = sub.add_parser("parse")
    q.add_argument("log")
    s = sub.add_parser("summarize")
    s.add_argument("--eps", type=float, default=1.0, help="accuracy tolerance in percentage points")
    args = p.parse_args()
    OUT = Path(args.out)
    {"plan": cmd_plan, "run": cmd_run, "parse": cmd_parse, "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
