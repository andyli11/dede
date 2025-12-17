# Cluster Scheduling with DeDe Formulation

Cluster scheduling experiments using DeDe (Decentralized Decomposition) with ADMM and differential privacy noise injection.

## Quick Start

**First, navigate to the code directory:**
```bash
cd code
```

### Plotting Scripts

**`plot_noise_placement.py`** - Compare noise placement impact on performance
- Reads `dede-form-{objective}.csv` results
- Shows normalized objective gap: `(obj_noisy - obj_baseline) / |obj_baseline|`
- Configure `OBJECTIVE` and `SIGMA_TO_PLOT` in the script
```bash
cd code
python plot_noise_placement.py
```

**`plot_noise_injection_points.py`** - Conceptual convergence visualization
- Simulates how noise at different points affects convergence
- Shows baseline, input noise, internal noise, and output noise
```bash
cd code
python plot_noise_injection_points.py
```

**`plot_noise_accumulation.py`** - Privacy accounting comparison
- Compares privacy cost accumulation (Pure DP, Approx DP, RDP, zCDP, GDP)
- Toy model for illustration
```bash
cd code
python plot_noise_accumulation.py
```

## Setup

**Requirements:**
- Python 3.9+
- Gurobi Optimizer with valid license (`gurobi_cl` in PATH)

**Install dependencies:**
```bash
pip install numpy pandas matplotlib cvxpy ray gurobipy
```

## Running Experiments

### Single Run
```bash
cd code
python dede_form.py --obj max_min_fairness_perf --dp-mode none --dp-sigma 0.0 --dp-seed 0
```

### Batch Experiments
```bash
cd code
python run_noise_placement_experiments.py
```
Runs all combinations of:
- DP modes: `none`, `input`, `x_update`, `z_update`, `y_update`, `output`
- Sigmas: `0.0`, `0.02`, `0.05`
- Seeds: `0-4`

### Key Arguments

**Required:**
- `--obj`: `max_min_fairness_perf`, `max_proportional_fairness`, or `gandiva`

**DP Parameters:**
- `--dp-mode`: Where to inject noise (`none`, `input`, `x_update`, `z_update`, `y_update`, `output`)
- `--dp-sigma`: Noise standard deviation (default: `0.0`)
- `--dp-seed`: Random seed (default: `0`)

**Other (defaults usually fine):**
- `--cluster-spec-file`: Cluster config (default: `data/cluster_spec.json`)
- `--throughputs-file`: Throughput data (default: `data/simulation_throughputs.npy`)
- `--admm-steps`: ADMM iterations (default: `20`)
- `--rho`: ADMM penalty (default: `0.1`)
- `--num-cpus`: CPU cores (default: all available)

## Output

**CSV Results:** `dede-form-{objective}.csv`
- Columns: `num_workers`, `num_jobs`, `objective`, `obj_val`, `runtime`, `dp_mode`, `dp_sigma`, `dp_seed`, etc.

**Logs:** `dede-form-logs/{objective}/`

## Noise Injection Points

- `none`: No noise (baseline)
- `input`: Noise on throughput values before optimization
- `x_update`: Noise on resource variables after each ADMM iteration
- `z_update`: Noise on consensus variable (once before ADMM)
- `y_update`: Noise on dual variables
- `output`: Noise on final allocation

## Troubleshooting

**Gurobi license not found:**
- Ensure `gurobi_cl` is in PATH
- Check license at `~/gurobi.lic`
- Run `gurobi_cl --license` to verify

**Empty array / NaN errors:**
- Code handles these edge cases automatically
- Check that input data is valid
