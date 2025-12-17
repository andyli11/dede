#!/usr/bin/env python3
import subprocess
import itertools

# Use the local Python in your .venv
PYTHON = "python"

# Script name (lowercase, matches your actual file)
SCRIPT = "./dede_form.py"

# Which DP modes to test
DP_MODES = ["none", "input", "x_update", "z_update", "y_update", "output"]

# One or more sigmas to try
SIGMAS = [0.0, 0.02, 0.05]

# How many random seeds per config
SEEDS = [0, 1, 2, 3, 4]

# Objective to use (must match allowed choices: 'max_min_fairness_perf', 'max_proportional_fairness', 'gandiva')
OBJECTIVE = "max_min_fairness_perf"


def main():
    for dp_mode, sigma, seed in itertools.product(DP_MODES, SIGMAS, SEEDS):
        print(f"Running dp_mode={dp_mode}, sigma={sigma}, seed={seed}")

        cmd = [
            PYTHON,
            SCRIPT,
            "--obj",
            OBJECTIVE,
            "--dp-mode",
            dp_mode,
            "--dp-sigma",
            str(sigma),
            "--dp-seed",
            str(seed),
            # The following have sensible defaults in dede_form.py,
            # so you can omit them unless you want to override:
            # "--cluster-spec-file", "data/cluster_spec.json",
            # "--throughputs-file", "data/simulation_throughputs.npy",
            # "--max-iter", "50",
        ]

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
