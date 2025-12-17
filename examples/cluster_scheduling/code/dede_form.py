#!/usr/bin/env python3

from benchmark_helpers import get_args, print_

import os
import pickle
import json
import cvxpy as cp

from lib.scheduler import Scheduler
from lib.utils import get_policy


TOP_DIR = "dede-form-logs"

# Extended to include DP metadata (you’ll still need matching changes
# in scheduler.simulate() if you want these columns filled per row).
HEADERS = [
    "num_workers",
    "num_jobs",
    "objective",
    "obj_val",
    "runtime",
    "num_cpu",
    "rho",
    "admm_steps",
    "fix_steps",
    "dp_mode",  # NEW
    "dp_sigma",  # NEW
    "dp_seed",  # NEW
]
PLACEHOLDER = ",".join("{}" for _ in HEADERS)

OUTPUT_CSV_TEMPLATE = "dede-form-{}.csv"


def benchmark(args, output_csv):
    policy = get_policy(args.obj)
    cluster_spec = json.load(open(args.cluster_spec_file, "r"))
    if args.num_worker_types > 0:
        cluster_spec = {
            worker_type: n
            for worker_type, n in cluster_spec.items()
            if int(worker_type.split("_")[1]) < args.num_worker_types
        }

    # Pass DP configuration down into the Scheduler, which passes it
    # further into DeDeFormulation.
    sched = Scheduler(
        policy,
        throughputs_file=args.throughputs_file,
        enable_dede=True,
        num_cpus=args.num_cpus,
        rho=args.rho,
        warmup=args.warmup,
        warmup_admm_steps=args.warmup_admm_steps,
        admm_steps=args.admm_steps,
        fix_steps=args.fix_steps,
        dp_mode=args.dp_mode,  # NEW
        dp_sigma=args.dp_sigma,  # NEW
        dp_seed=args.dp_seed,  # NEW
    )

    run_dir = os.path.join(TOP_DIR, args.obj)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    with open(output_csv, "a") as results:
        # Header for the CSV; scheduler.simulate() will write rows.
        print_(",".join(HEADERS), file=results)
        sched.simulate(
            cluster_spec=cluster_spec,
            lam=50.0,  # relatively low arrival rate
            num_total_jobs=20,  # << fewer jobs than the full experiment
            fixed_job_duration=200,  # smaller jobs
            generate_multi_gpu_jobs=False,
            generate_multi_priority_jobs=False,
            simulate_steady_state=False,
            max_iter=200,  # hard cap on rounds
            results_file=results,  # where you’re writing the CSV lines
            results_folder=None,
        )


if __name__ == "__main__":
    if not os.path.exists(TOP_DIR):
        os.makedirs(TOP_DIR)

    args, output_csv = get_args(OUTPUT_CSV_TEMPLATE)
    benchmark(args, output_csv)
