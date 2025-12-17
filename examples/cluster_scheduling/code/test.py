import numpy as np
import matplotlib.pyplot as plt


def simulate_noise_placement(T_max=50, seed=0):
    """
    Simulate convergence of an iterative optimizer under different noise placements:
      - baseline (no noise)
      - noise on demand input
      - noise injected into an internal optimizer step (e.g., z-update)
      - noise added after the output

    x-axis: iterations T
    y-axis: performance metric (higher is better, normalized to [0, 1]).
    """
    rng = np.random.default_rng(seed)

    T = np.arange(1, T_max + 1)

    # True optimum performance
    p_star = 1.0

    # 1) Baseline: clean exponential convergence
    k_base = 0.12
    perf_baseline = p_star - np.exp(-k_base * T)  # starts low, approaches p_star

    # 2) Input noise:
    #    - the optimizer "sees" noisy / biased demands,
    #    - so it converges more slowly and to a slightly worse final value.
    k_input = 0.08
    bias_input = 0.10  # final gap below optimum
    perf_input_clean = p_star - bias_input - np.exp(-k_input * T)
    noise_input = rng.normal(0.0, 0.01, size=T.shape)
    perf_input = np.clip(perf_input_clean + noise_input, 0.0, 1.0)

    # 3) Internal-step noise (e.g., z-update):
    #    - convergence rate a bit slower than baseline,
    #    - small variance,
    #    - final performance very close to optimum.
    k_internal = 0.10
    bias_internal = 0.03
    perf_internal_clean = p_star - bias_internal - np.exp(-k_internal * T)
    noise_internal = rng.normal(0.0, 0.005, size=T.shape)
    perf_internal = np.clip(perf_internal_clean + noise_internal, 0.0, 1.0)

    # 4) Output noise:
    #    - the underlying optimization behaves like baseline,
    #      but reported performance has measurement noise.
    noise_output = rng.normal(0.0, 0.02, size=T.shape)
    perf_output = np.clip(perf_baseline + noise_output, 0.0, 1.0)

    # ---- Plot ----
    plt.figure(figsize=(8, 5))
    plt.plot(T, perf_baseline, label="Baseline (no noise)", linewidth=2)
    plt.plot(T, perf_input, label="Noise on demand input")
    plt.plot(T, perf_internal, label="Noise in internal step (e.g., z-update)")
    plt.plot(T, perf_output, label="Noise after output")

    plt.xlabel("Number of iterations $T$")
    plt.ylabel("Performance metric (normalized)")
    plt.title("Effect of noise placement on convergence and performance")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optionally return the arrays if you want to use them in the paper
    return {
        "T": T,
        "baseline": perf_baseline,
        "input_noise": perf_input,
        "internal_noise": perf_internal,
        "output_noise": perf_output,
    }


simulate_noise_placement()
