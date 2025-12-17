import numpy as np
import matplotlib.pyplot as plt

# Parameters for the toy model
T_max = 200  # max number of iterations
T = np.arange(1, T_max + 1)  # 1, 2, ..., 200

epsilon0 = 0.1  # per-iteration "base" privacy cost (toy)
delta = 1e-6  # fixed delta for all approximate notions

# Toy privacy accounting models
# 1) Pure DP: naive composition: eps_total = T * eps0
eps_pure = T * epsilon0

# 2) Approximate DP ((eps, delta)-DP) with advanced composition-style scaling
#    We use the dominant sqrt(T) term: eps_total ~ eps0 * sqrt(2 T log(1/delta))
eps_approx = epsilon0 * np.sqrt(2 * T * np.log(1 / delta))

# 3) RDP: tighter sqrt(T) behavior with a smaller constant
#    (Illustrative: we scale by 0.6 to show it's tighter.)
eps_rdp = 0.6 * epsilon0 * np.sqrt(T)

# 4) zCDP: similar behavior, slightly looser than RDP
eps_zcdp = 0.75 * epsilon0 * np.sqrt(T)

# 5) Gaussian DP (GDP): similar to zCDP, but we give it its own curve
eps_gdp = 0.7 * epsilon0 * np.sqrt(T)

# Plotting
plt.figure(figsize=(8, 5))

plt.plot(T, eps_pure, label="Pure DP ($\\varepsilon$-DP)", linewidth=2)
plt.plot(T, eps_approx, label="Approx DP ($(\\varepsilon, \\delta)$-DP)", linewidth=2)
plt.plot(T, eps_rdp, label="RDP", linewidth=2)
plt.plot(T, eps_zcdp, label="zCDP", linewidth=2)
plt.plot(T, eps_gdp, label="GDP", linewidth=2)

plt.xlabel("Number of iterations $T$", fontsize=12)
plt.ylabel("Total privacy cost $\\varepsilon(T)$ (toy units)", fontsize=12)
plt.title("Illustrative Comparison of Privacy Accounting for Iterative DP", fontsize=13)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
