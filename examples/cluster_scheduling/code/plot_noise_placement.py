#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OBJECTIVE = "max_min_fairness_perf"
CSV_FILE = f"dede-form-{OBJECTIVE}.csv"

# Which noisy sigma to visualize
SIGMA_TO_PLOT = 0.02


def main():
    df = pd.read_csv(CSV_FILE)

    # ---------------------------------------------------------
    # 1. Remove accidental header duplication inside the CSV
    # ---------------------------------------------------------
    df = df[df["dp_mode"] != "dp_mode"]
    df = df[df["dp_sigma"] != "dp_sigma"]

    # ---------------------------------------------------------
    # 2. Normalize dp_mode text
    # ---------------------------------------------------------
    df["dp_mode"] = df["dp_mode"].astype(str).str.strip()

    # ---------------------------------------------------------
    # 3. Convert dp_sigma to numeric
    # ---------------------------------------------------------
    df["dp_sigma"] = pd.to_numeric(df["dp_sigma"], errors="coerce")

    # ---------------------------------------------------------
    # 4. Convert obj_val to numeric by extracting the FIRST float
    # ---------------------------------------------------------
    df["obj_val"] = (
        df["obj_val"]
        .astype(str)
        .str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")[0]
        .astype(float)
    )

    # ---------------------------------------------------------
    # 5. Baseline selection: dp_mode = none AND sigma = 0.0
    # ---------------------------------------------------------
    baseline_mask = (df["dp_mode"] == "none") & (df["dp_sigma"].abs() < 1e-9)
    baseline_df = df[baseline_mask]

    print("Unique dp_mode values:", df["dp_mode"].unique())
    print("Unique dp_sigma values:", sorted(df["dp_sigma"].dropna().unique()))
    print("Baseline preview:")
    print(baseline_df.head())

    if baseline_df.empty:
        raise ValueError(
            "No baseline found: CSV must include at least one "
            "`--dp-mode none --dp-sigma 0.0` run."
        )

    baseline = baseline_df["obj_val"].mean()
    baseline_std = baseline_df["obj_val"].std()

    # ---------------------------------------------------------
    # 6. Noisy configs for the chosen sigma
    # ---------------------------------------------------------
    noisy_df = df[df["dp_sigma"] == SIGMA_TO_PLOT].dropna(subset=["obj_val"])

    grouped = noisy_df.groupby("dp_mode").agg(
        avg_obj=("obj_val", "mean"),
        std_obj=("obj_val", "std"),
        count=("obj_val", "count"),
    )

    # Insert baseline as a comparable row
    grouped.loc["none", "avg_obj"] = baseline
    grouped.loc["none", "std_obj"] = baseline_std
    grouped.loc["none", "count"] = len(baseline_df)

    # ---------------------------------------------------------
    # 7. Compute normalized gap
    # ---------------------------------------------------------
    grouped["obj_gap"] = (grouped["avg_obj"] - baseline) / abs(baseline)

    # Order for plotting
    modes_order = ["none", "input", "x_update", "z_update", "y_update", "output"]
    modes_order = [m for m in modes_order if m in grouped.index]
    grouped = grouped.loc[modes_order]

    # ---------------------------------------------------------
    # 8. Plot
    # ---------------------------------------------------------
    x = np.arange(len(modes_order))

    plt.figure(figsize=(9, 5))
    plt.bar(
        x,
        grouped["obj_gap"],
        yerr=grouped["std_obj"] / abs(baseline),
        capsize=5,
    )

    plt.xticks(x, modes_order, rotation=30)
    plt.xlabel("Noise placement in DeDe")
    plt.ylabel(
        "Normalized objective gap\n" "(obj_noisy − obj_baseline) / |obj_baseline|"
    )
    plt.title(f"Effect of noise placement on DeDe (σ={SIGMA_TO_PLOT})")

    plt.axhline(0.0, color="black", linewidth=1)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
