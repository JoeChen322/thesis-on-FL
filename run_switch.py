#!/usr/bin/env python3
"""Run adaptive paradigm switching experiments.

This script runs FL / SL / SFL one round at a time and decides the next
paradigm from the metrics of the round that just finished.  All paradigms use
the same checkpoint, so switching continues from the previous model state.

Typical VM workflow:

  python3 run_baseline.py run
  python3 run_baseline.py summarize
  python3 run_switch.py plan --setup resnet_cifar10 --baseline baseline_results_mnist_cnn/rounds.csv
  python3 run_switch.py run  --setup resnet_cifar10 --baseline baseline_results_mnist_cnn/rounds.csv --only S1
  python3 run_switch.py report --setup resnet_cifar10 --baseline baseline_results_mnist_cnn/rounds.csv
"""

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from run_baseline import parse_log


TAU = 0.2
ACC_DROP_MAX = 10.0
ACC_DROP_MIN = 0.1
T_HIGH_FACTOR = 2.5
MARGIN = 1.25
RETURN_MARGIN = 0.8
DWELL = 1
DEFAULT_ROUNDS = 5
BATCH_SIZE = 64
TIMEOUT_S = 6 * 3600

SCRIPTS = {
    "fl": "fl_mnist_minimal.py",
    "sl": "sl_mnist_minimal.py",
    "sfl": "fsl_mnist_minimal.py",
}
NEXT = {"sl": "sfl", "sfl": "fl", "fl": "fl"}

SETUPS = {
    "cnn_mnist": {
        "model": "cnn",
        "dataset": "mnist",
        "train_size": 60000,
        "args": ["--model", "cnn", "--cnn-split-after", "conv1"],
    },
    "resnet_mnist": {
        "model": "resnet18",
        "dataset": "mnist",
        "train_size": 60000,
        "args": [
            "--model", "resnet",
            "--resnet-depth", "18",
            "--resnet-split-after", "layer2",
        ],
    },
    "resnet_cifar10": {
        "model": "resnet18",
        "dataset": "cifar10",
        "train_size": 50000,
        "args": [
            "--model", "resnet",
            "--resnet-depth", "18",
            "--resnet-split-after", "layer2",
        ],
    },
}


def _latency_for_comm_growth(cal, paradigm):
    return (
        ((1 + TAU) * MARGIN * cal["Cmax"][paradigm] - cal["Cmin"][paradigm])
        / cal["M"][paradigm]
    )


def _latency_for_time_high(cal, paradigm):
    return (MARGIN * cal["T_high"] - cal["Rmin"][paradigm]) / cal["M"][paradigm]


def schedule_zero(cal, rounds):
    return [0.0] * rounds


def schedule_sl_latency(cal, rounds):
    return [0.0] + [_latency_for_comm_growth(cal, "sl")] * (rounds - 1)


def schedule_sfl_latency(cal, rounds):
    return [0.0] + [_latency_for_comm_growth(cal, "sfl")] * (rounds - 1)


def schedule_sl_time_high(cal, rounds):
    return [0.0] + [_latency_for_time_high(cal, "sl")] * (rounds - 1)


def schedule_sfl_time_recovery(cal, rounds):
    return [_latency_for_time_high(cal, "sfl")] + [0.0] * (rounds - 1)


SCENARIOS = {
    "S1": {
        "rule": "latency",
        "K": 3,
        "alpha": 1.0,
        "start": "auto",
        "schedule": schedule_sl_latency,
        "expect": "SL->SFL after a communication spike",
    },
    "S2": {
        "rule": "latency control",
        "K": 10,
        "alpha": 1.0,
        "start": "fl",
        "schedule": schedule_sfl_latency,
        "expect": "none; FL is not left by latency",
    },
    "S3": {
        "rule": "time high",
        "K": 3,
        "alpha": 1.0,
        "start": "auto",
        "schedule": schedule_sl_time_high,
        "expect": "SL->FL after a high round time",
    },
    "S4": {
        "rule": "time high and return",
        "K": 10,
        "alpha": 1.0,
        "start": "auto",
        "schedule": schedule_sfl_time_recovery,
        "expect": "SFL->FL while slow, then FL->SFL after recovery",
    },
    "S5": {
        "rule": "accuracy watch",
        "K": 3,
        "alpha": 1.0,
        "start": "sl",
        "schedule": schedule_zero,
        "expect": "switches only if real accuracy drops past the calibrated threshold",
    },
    "S6": {
        "rule": "static FL",
        "K": 3,
        "alpha": 1.0,
        "start": "fl",
        "forced": {},
        "schedule": schedule_zero,
        "expect": "none; static FL reference",
    },
    "S7": {
        "rule": "forced switch",
        "K": 3,
        "alpha": 1.0,
        "start": "fl",
        "forced": {3: "sfl"},
        "schedule": schedule_zero,
        "expect": "FL x3 -> SFL x2",
    },
}


def load_baseline(path, setup, baseline_cpus):
    setup_cfg = SETUPS[setup]
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] != setup_cfg["model"] or row["dataset"] != setup_cfg["dataset"]:
                continue
            if baseline_cpus is not None and float(row.get("cpus", 1)) != float(baseline_cpus):
                continue
            delay = float(row.get("delay_time") or 0.0)
            rows.append({
                "paradigm": row["paradigm"],
                "K": int(row["num_clients"]),
                "alpha": float(row["alpha"]),
                "round": int(row["round"]),
                "acc": float(row["acc"]) if row.get("acc") else float("nan"),
                "total": float(row["total_runtime"]) - delay,
                "comm": float(row["communication_time"]) - delay,
                "train": float(row["training_time"]),
            })
    if not rows:
        raise SystemExit(
            f"no baseline rows for {setup_cfg['model']} / {setup_cfg['dataset']} "
            f"in {path}"
        )
    return rows


def baseline_curve(rows, paradigm, K, alpha):
    return sorted(
        (
            row for row in rows
            if row["paradigm"] == paradigm
            and row["K"] == K
            and abs(row["alpha"] - alpha) < 1e-9
        ),
        key=lambda row: row["round"],
    )


_PARTITION_CACHE = {}


def client_sizes(setup, K, alpha):
    key = (setup, K, alpha)
    if key in _PARTITION_CACHE:
        return _PARTITION_CACHE[key]
    try:
        from split_learning_utils import load_dataset, split_indices

        dataset = load_dataset(SETUPS[setup]["dataset"], train=True)
        sizes = [len(indices) for indices in split_indices(dataset, K, alpha)]
    except Exception as exc:
        print(f"  partition unavailable ({exc!r}); using equal client sizes")
        n = SETUPS[setup]["train_size"]
        sizes = [n // K] * K
    _PARTITION_CACHE[key] = sizes
    return sizes


def messages(setup, K, alpha):
    batches = sum(math.ceil(size / BATCH_SIZE) for size in client_sizes(setup, K, alpha) if size > 0)
    return {
        "fl": 2 * K,
        "sl": 2 * batches,
        "sfl": 2 * batches + 2 * K,
    }


def calibrate(rows, setup, K, alpha):
    cal = {key: {} for key in ("R", "C", "Rmin", "Rmax", "Cmin", "Cmax", "drop", "ratio", "n")}
    cal["M"] = messages(setup, K, alpha)
    for paradigm in ("fl", "sl", "sfl"):
        curve = baseline_curve(rows, paradigm, K, alpha)
        if not curve:
            raise SystemExit(f"baseline has no {paradigm} rows for K={K}, alpha={alpha}")
        round_times = [row["total"] for row in curve]
        comm_times = [row["comm"] for row in curve]
        accs = [row["acc"] for row in curve]
        cal["n"][paradigm] = len(curve)
        cal["R"][paradigm] = statistics.median(round_times)
        cal["C"][paradigm] = statistics.median(comm_times)
        cal["Rmin"][paradigm] = min(round_times)
        cal["Rmax"][paradigm] = max(round_times)
        cal["Cmin"][paradigm] = min(comm_times)
        cal["Cmax"][paradigm] = max(comm_times)
        cal["drop"][paradigm] = max([0.0] + [accs[i - 1] - accs[i] for i in range(1, len(accs))])
        cal["ratio"][paradigm] = max(
            [0.0] + [
                comm_times[i] / comm_times[i - 1]
                for i in range(1, len(comm_times))
                if comm_times[i - 1] > 0
            ]
        )
    cal["T_high"] = T_HIGH_FACTOR * max(cal["Rmax"]["sl"], cal["Rmax"]["sfl"])
    return cal


def accuracy_threshold(rows):
    best = (0.0, None)
    keys = {
        (row["paradigm"], row["K"], row["alpha"])
        for row in rows
        if row["paradigm"] in ("sl", "sfl")
    }
    for paradigm, K, alpha in sorted(keys):
        accs = [row["acc"] for row in baseline_curve(rows, paradigm, K, alpha)]
        for i in range(1, len(accs)):
            drop = accs[i - 1] - accs[i]
            if drop > best[0]:
                best = (drop, (paradigm, K, alpha, i + 1))
    return max(ACC_DROP_MIN, min(ACC_DROP_MAX, 0.5 * best[0])), best


def resolve_scenario(name, rows, num_rounds):
    scenario = dict(SCENARIOS[name])
    threshold, largest = accuracy_threshold(rows)
    scenario["acc_drop"] = threshold
    scenario["largest_baseline_drop"] = largest
    scenario["rounds"] = num_rounds
    return scenario


def initial_method(setup, K, alpha, cpus=1.0):
    try:
        from noniid_jsd_switch import detect_noniid_condition

        mean_jsd = detect_noniid_condition(
            K,
            alpha,
            dataset_name=SETUPS[setup]["dataset"],
        ).mean_jsd
    except Exception:
        mean_jsd = 0.0 if alpha >= 1.0 else 1.0
    if mean_jsd >= 0.2:
        return "sfl", mean_jsd
    if K > 5 and cpus >= 2:
        return "fl", mean_jsd
    if K < 5:
        return "sl", mean_jsd
    return "sfl", mean_jsd


class Controller:
    def __init__(self, method, cal, acc_drop):
        self.method = method
        self.cal = cal
        self.acc_drop = acc_drop
        self.rounds_in_method = 0
        self.entered_by = "start"
        self.last_split = method if method != "fl" else None
        self.last_split_round = None
        self.prev = None

    def decide(self, record, latency_per_message):
        paradigm = self.method
        self.rounds_in_method += 1
        previous, self.prev = self.prev, record
        same_paradigm = (
            previous is not None
            and previous["method"] == paradigm
            and self.rounds_in_method >= 2
        )

        if paradigm != "fl":
            self.last_split = paradigm
            self.last_split_round = record["total"] - record["delay"]
            if record["total"] > self.cal["T_high"]:
                return self._go(
                    "fl",
                    "time",
                    f"T_round {record['total']:.1f}s > T_high {self.cal['T_high']:.1f}s",
                )
            if same_paradigm and record["acc"] < previous["acc"] - self.acc_drop:
                return self._go(
                    NEXT[paradigm],
                    "accuracy",
                    f"acc {record['acc']:.2f}% < {previous['acc']:.2f}% - {self.acc_drop:.2f}",
                )
            if same_paradigm and record["comm"] > (1 + TAU) * previous["comm"]:
                return self._go(
                    NEXT[paradigm],
                    "latency",
                    f"T_comm {record['comm']:.1f}s > {1 + TAU:.2f} x {previous['comm']:.1f}s",
                )
            return paradigm, "", "keep"

        if (
            self.entered_by == "time"
            and self.last_split
            and self.rounds_in_method >= DWELL
        ):
            predicted = self.last_split_round + latency_per_message * self.cal["M"][self.last_split]
            if predicted < RETURN_MARGIN * self.cal["T_high"]:
                return self._go(
                    self.last_split,
                    "return",
                    f"predicted {self.last_split.upper()} {predicted:.1f}s < "
                    f"{RETURN_MARGIN} x T_high {self.cal['T_high']:.1f}s",
                )
        return paradigm, "", "keep"

    def _go(self, new_method, rule, reason):
        if new_method != self.method:
            self.method = new_method
            self.rounds_in_method = 0
            self.entered_by = rule
        return new_method, rule, reason


def build_training_command(args, setup, method, checkpoint_path, alpha):
    setup_cfg = SETUPS[setup]
    script_path = Path(args.code_dir).resolve() / SCRIPTS[method]
    command = [
        args.python,
        str(script_path),
        "--num-clients", str(args.num_clients),
        "--num-rounds", "1",
        "--local-epochs", str(args.local_epochs),
        "--client-num-cpus", str(args.client_num_cpus),
        "--dataset", setup_cfg["dataset"],
        "--noniid-alpha", str(alpha),
        "--communication-delay", "0",
        "--checkpoint-path", str(checkpoint_path),
        *setup_cfg["args"],
    ]
    if method in ("fl", "sl", "sfl"):
        command.extend(["--client-num-gpus", str(args.client_num_gpus)])
    if method in ("sl", "sfl"):
        command.append("--eval-every-round")
        if args.max_batches:
            command.extend(["--max-batches", str(args.max_batches)])
    return command


def real_round(args, setup, method, checkpoint_path, log_path, alpha):
    command = build_training_command(args, setup, method, checkpoint_path, alpha)
    code_dir = Path(args.code_dir).resolve()
    env = os.environ.copy()

    start = time.perf_counter()
    with open(log_path, "w") as log:
        log.write("# " + " ".join(str(part) for part in command) + "\n")
        log.flush()
        proc = subprocess.run(
            command,
            cwd=code_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout,
        )
    wall = time.perf_counter() - start
    parsed = parse_log(log_path.read_text(errors="replace").split("\n", 1)[-1])
    if proc.returncode != 0 or not parsed["rounds"]:
        raise RuntimeError(f"{method} round failed (rc={proc.returncode}), see {log_path}")
    row = parsed["rounds"][-1]
    acc = row.get("acc", parsed["final"].get("acc", float("nan")))
    return {
        "acc": acc,
        "loss": row.get("loss", float("nan")),
        "train": row.get("training_time", 0.0),
        "comm_meas": row.get("communication_time", 0.0),
        "total_meas": row.get("total_runtime", 0.0),
        "wall": wall,
    }


def simulated_round(rows, scenario, method, global_round):
    curve = baseline_curve(rows, method, scenario["K"], scenario["alpha"])
    row = curve[min(global_round, len(curve)) - 1]
    return {
        "acc": row["acc"],
        "loss": float("nan"),
        "train": row["train"],
        "comm_meas": row["comm"],
        "total_meas": row["total"],
        "wall": float("nan"),
    }


def run_scenario(name, args, rows, simulate=False):
    scenario = resolve_scenario(name, rows, args.num_rounds)
    args.num_clients = scenario["K"]
    cal = calibrate(rows, args.setup, scenario["K"], scenario["alpha"])
    schedule = scenario["schedule"](cal, scenario["rounds"])
    if scenario["start"] == "auto":
        start, jsd = initial_method(args.setup, scenario["K"], scenario["alpha"], args.client_num_cpus)
        start_reason = f"initialization rule (J={jsd:.3f})"
    else:
        start = scenario["start"]
        start_reason = "forced"

    controller = Controller(start, cal, scenario["acc_drop"])
    run_dir = Path(args.out) / args.setup / name
    if not simulate:
        if run_dir.exists():
            archived = run_dir.with_name(
                run_dir.name + "_old_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            shutil.move(str(run_dir), str(archived))
        (run_dir / "logs").mkdir(parents=True)
        checkpoint_path = run_dir / "shared_checkpoint.pt"

    records = []
    method = start
    say = (lambda *parts, **kwargs: None) if args.quiet else print
    say(
        f"\n== {args.setup} {name} ({scenario['rule']}): "
        f"K={scenario['K']} alpha={scenario['alpha']} start={start.upper()} "
        f"[{start_reason}] expected: {scenario['expect']}"
    )

    for round_number in range(1, scenario["rounds"] + 1):
        latency_per_message = schedule[round_number - 1]
        if simulate:
            metrics = simulated_round(rows, scenario, method, round_number)
        else:
            metrics = real_round(
                args,
                args.setup,
                method,
                checkpoint_path,
                run_dir / "logs" / f"round{round_number:02d}_{method}.log",
                scenario["alpha"],
            )

        delay = latency_per_message * cal["M"][method]
        record = {
            "scenario": name,
            "setup": args.setup,
            "round": round_number,
            "method": method,
            "alpha": scenario["alpha"],
            "latency_per_msg": latency_per_message,
            "messages": cal["M"][method],
            "delay": delay,
            "acc": metrics["acc"],
            "loss": metrics["loss"],
            "train": metrics["train"],
            "comm": metrics["comm_meas"] + delay,
            "total": metrics["total_meas"] + delay,
            "total_meas": metrics["total_meas"],
            "wall": metrics["wall"],
            "overhead": metrics["wall"] - metrics["total_meas"],
        }

        if scenario.get("forced") is not None:
            next_method = scenario["forced"].get(round_number, method)
            rule = "forced" if next_method != method else ""
            reason = f"switch forced after round {round_number}" if rule else "keep"
        else:
            next_method, rule, reason = controller.decide(record, latency_per_message)

        record.update({"next": next_method, "rule": rule, "reason": reason})
        records.append(record)
        say(
            f"  r{round_number:02d} {method.upper():3s} "
            f"d={latency_per_message:.4f}s/msg acc={record['acc']:6.2f}% "
            f"comm={record['comm']:8.1f}s round={record['total']:8.1f}s "
            f"-> {next_method.upper()}"
            + (f"  [{rule}: {reason}]" if rule else "")
        )

        if not simulate:
            write_csv(run_dir / "rounds.csv", records)
        method = next_method

    transitions = [
        f"r{row['round']}:{row['method'].upper()}->{row['next'].upper()}({row['rule']})"
        for row in records
        if row["next"] != row["method"]
    ]
    switch_points = [
        {
            "round": row["round"],
            "from": row["method"],
            "to": row["next"],
            "rule": row["rule"],
            "acc_before": row["acc"],
            "acc_after": records[index + 1]["acc"],
        }
        for index, row in enumerate(records[:-1])
        if row["next"] != row["method"]
    ]
    summary = {
        "scenario": name,
        "setup": args.setup,
        "K": scenario["K"],
        "alpha": scenario["alpha"],
        "start": start,
        "start_reason": start_reason,
        "expected": scenario["expect"],
        "transitions": transitions,
        "switch_points": switch_points,
        "acc_drop": scenario["acc_drop"],
        "calibration": cal,
        "schedule": schedule,
        "final_acc": records[-1]["acc"],
        "total_time": sum(row["total"] for row in records),
    }
    say(f"  transitions: {', '.join(transitions) or 'none'}")
    if not simulate:
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return records, summary


def write_csv(path, records):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in records for key in row))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def selected_scenarios(args):
    if not args.only:
        return list(SCENARIOS)
    names = [item.strip().upper() for item in args.only.split(",") if item.strip()]
    unknown = [name for name in names if name not in SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown scenario(s): {', '.join(unknown)}")
    return names


def verify(name, scenario, cal, schedule):
    checks = []

    def add(text, ok, detail):
        checks.append((text, bool(ok), detail))

    def latency_fires(paradigm, old_delay, new_delay):
        new = cal["Cmin"][paradigm] + new_delay * cal["M"][paradigm]
        old = cal["Cmax"][paradigm] + old_delay * cal["M"][paradigm]
        add(
            f"{paradigm.upper()} latency rule fires",
            new > (1 + TAU) * old,
            f"{new:.1f}s > {1 + TAU:.2f} x {old:.1f}s",
        )

    def time_fires(paradigm, delay):
        value = cal["Rmin"][paradigm] + delay * cal["M"][paradigm]
        add(
            f"{paradigm.upper()} time rule fires",
            value > cal["T_high"],
            f"{value:.1f}s > T_high {cal['T_high']:.1f}s",
        )

    if name == "S1":
        latency_fires("sl", 0.0, schedule[1])
    elif name == "S2":
        add("FL is not left by latency", True, "controller rule")
    elif name == "S3":
        time_fires("sl", schedule[1])
    elif name == "S4":
        time_fires("sfl", schedule[0])
        add(
            "FL return condition after recovery",
            cal["Rmax"]["sfl"] < RETURN_MARGIN * cal["T_high"],
            f"SFL {cal['Rmax']['sfl']:.1f}s < {RETURN_MARGIN} x {cal['T_high']:.1f}s",
        )
    elif name == "S5":
        add(
            "accuracy rule threshold is positive",
            scenario["acc_drop"] > 0,
            f"threshold {scenario['acc_drop']:.2f} points",
        )
    else:
        add("rules disabled or forced", True, "scenario definition")
    return checks


def cmd_plan(args):
    rows = load_baseline(args.baseline, args.setup, args.baseline_cpus)
    threshold, largest = accuracy_threshold(rows)
    print(
        f"accuracy threshold: {threshold:.2f} points "
        f"(largest baseline drop: {largest[0]:.2f} at {largest[1]})"
    )
    failed = []
    for name in selected_scenarios(args):
        scenario = resolve_scenario(name, rows, args.num_rounds)
        cal = calibrate(rows, args.setup, scenario["K"], scenario["alpha"])
        schedule = scenario["schedule"](cal, scenario["rounds"])
        print(
            f"\n[{name}] K={scenario['K']} alpha={scenario['alpha']} "
            f"T_high={cal['T_high']:.1f}s"
        )
        print(
            "  "
            + "  ".join(
                f"{paradigm.upper()}: R {cal['Rmin'][paradigm]:.0f}-{cal['Rmax'][paradigm]:.0f}s, "
                f"C {cal['Cmin'][paradigm]:.0f}-{cal['Cmax'][paradigm]:.0f}s, "
                f"M {cal['M'][paradigm]}"
                for paradigm in ("fl", "sl", "sfl")
            )
        )
        print("  delay schedule: " + ", ".join(f"{value:.4f}" for value in schedule))
        for text, ok, detail in verify(name, scenario, cal, schedule):
            print(f"  [{'PASS' if ok else 'FAIL'}] {text}: {detail}")
            if not ok:
                failed.append(f"{name}: {text}")
        run_scenario(name, args, rows, simulate=True)
    print("\nall checks passed" if not failed else "\nFAILED checks:\n  " + "\n  ".join(failed))


def cmd_run(args):
    rows = load_baseline(args.baseline, args.setup, args.baseline_cpus)
    for name in selected_scenarios(args):
        started = time.time()
        _, summary = run_scenario(name, args, rows, simulate=False)
        print(
            f"  {name} done in {(time.time() - started) / 60:.1f} min; "
            f"transitions: {', '.join(summary['transitions']) or 'none'} "
            f"(expected: {summary['expected']})",
            flush=True,
        )


def static_reference(rows, args, name):
    scenario = resolve_scenario(name, rows, args.num_rounds)
    cal = calibrate(rows, args.setup, scenario["K"], scenario["alpha"])
    schedule = scenario["schedule"](cal, scenario["rounds"])
    out = {}
    for paradigm in ("fl", "sl", "sfl"):
        curve = baseline_curve(rows, paradigm, scenario["K"], scenario["alpha"])[:scenario["rounds"]]
        if len(curve) < scenario["rounds"]:
            continue
        total = sum(
            row["total"] + schedule[index] * cal["M"][paradigm]
            for index, row in enumerate(curve)
        )
        out[paradigm] = {"final_acc": curve[-1]["acc"], "total_time": total}
    return out


def continuity(rows, summary):
    parts = []
    for point in summary.get("switch_points", []):
        static_parts = []
        for paradigm in (point["from"], point["to"]):
            curve = baseline_curve(rows, paradigm, summary["K"], summary["alpha"])
            if len(curve) > point["round"]:
                before = curve[point["round"] - 1]["acc"]
                after = curve[point["round"]]["acc"]
                static_parts.append(f"static {paradigm.upper()} {before:.2f}->{after:.2f}")
        parts.append(
            f"r{point['round']} {point['from'].upper()}->{point['to'].upper()}: "
            f"{point['acc_before']:.2f}->{point['acc_after']:.2f} "
            f"({', '.join(static_parts)})"
        )
    return "; ".join(parts) or "no switch"


def cmd_report(args):
    rows = load_baseline(args.baseline, args.setup, args.baseline_cpus)
    report_rows = []
    for name in selected_scenarios(args):
        run_dir = Path(args.out) / args.setup / name
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            print(f"{name}: no results yet")
            continue
        summary = json.loads(summary_path.read_text())
        refs = static_reference(rows, args, name)
        start = summary["start"]
        oracle = min(refs, key=lambda paradigm: refs[paradigm]["total_time"]) if refs else ""
        row = {
            "scenario": name,
            "start": start.upper(),
            "transitions": " ".join(summary["transitions"]) or "none",
            "expected": summary["expected"],
            "adaptive_acc": round(summary["final_acc"], 2),
            "adaptive_time_min": round(summary["total_time"] / 60, 1),
            "oracle": oracle.upper() if oracle else "",
            "switch_continuity": continuity(rows, summary),
        }
        if start in refs:
            row["initial_static_acc"] = round(refs[start]["final_acc"], 2)
            row["initial_static_time_min"] = round(refs[start]["total_time"] / 60, 1)
        for paradigm, ref in refs.items():
            row[f"{paradigm}_acc"] = round(ref["final_acc"], 2)
            row[f"{paradigm}_time_min"] = round(ref["total_time"] / 60, 1)
        report_rows.append(row)
        print(row)
    if report_rows:
        path = Path(args.out) / args.setup / "switch_summary.csv"
        write_csv(path, report_rows)
        print(f"written {path}")


def add_common_args(parser):
    parser.add_argument("--setup", choices=list(SETUPS), required=True)
    parser.add_argument("--baseline", required=True, help="baseline rounds.csv from the same VM")
    parser.add_argument("--baseline-cpus", type=float, default=1.0)
    parser.add_argument("--only", default="", help="comma-separated scenarios, e.g. S1,S4")
    parser.add_argument("--out", default="switch_results")
    parser.add_argument("--code-dir", default=".")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num-rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--client-num-cpus", type=float, default=1.0)
    parser.add_argument("--client-num-gpus", type=float, default=0.0)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=TIMEOUT_S)
    parser.add_argument("--quiet", action="store_true")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    for command in ("plan", "run", "report"):
        subparser = subparsers.add_parser(command)
        add_common_args(subparser)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_rounds < 1:
        raise ValueError("--num-rounds must be at least 1")
    if args.client_num_cpus <= 0:
        raise ValueError("--client-num-cpus must be positive")
    {"plan": cmd_plan, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
