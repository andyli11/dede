import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
T_max = 50
T = np.arange(1, T_max + 1)

p_star = 1.0

# Baseline (fast convergence)
k_base = 0.12
perf_baseline = p_star - np.exp(-k_base * T)

# Input noise (slow, biased)
k_input = 0.08
bias_input = 0.10
perf_input = p_star - bias_input - np.exp(-k_input * T)
perf_input += rng.normal(0, 0.01, T.shape)

# Internal noise (z-update) (slightly slower, tiny bias)
k_internal = 0.10
bias_internal = 0.03
perf_internal = p_star - bias_internal - np.exp(-k_internal * T)
perf_internal += rng.normal(0, 0.005, T.shape)

# Output noise (noisy measurements)
perf_output = perf_baseline + rng.normal(0, 0.02, T.shape)

plt.figure(figsize=(8, 5))
plt.plot(T, perf_baseline, label="Baseline (no noise)", linewidth=3)
plt.plot(T, perf_input, label="Input noise")
plt.plot(T, perf_internal, label="Internal noise (z-update)")
plt.plot(T, perf_output, label="Output noise")

plt.xlabel("Iteration")
plt.ylabel("Performance")
plt.title("Simulated Convergence Under Different Noise Placements")
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
