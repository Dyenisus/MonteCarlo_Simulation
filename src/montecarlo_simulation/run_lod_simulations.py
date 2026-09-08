import datetime
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

from .data import load_data
from .lod_methods import run_nas_lod, run_predictive_lod, run_univariate_lod


def plot_comparison(results, config, out_dir):
    unit = config["metadata"]["concentration_unit"]
    fluid = config["metadata"]["fluid_type"]

    methods = []
    lods = []
    for m, val in results.items():
        methods.append(m)
        lods.append(val)

    _, ax = plt.subplots(figsize=(10, 6))
    colors = ["royalblue", "indianred", "seagreen"]

    bars = ax.bar(methods, lods, color=colors[: len(methods)], alpha=0.8)

    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + (max(lods) * 0.02),
            f"{yval:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_ylabel(f"Limit of Detection ({unit})", fontsize=12)
    ax.set_title(
        f"Comparison of LOD Concentrations ({fluid})", fontsize=14, fontweight="bold"
    )

    cell_text = [[f"{lod:.3f} {unit}"] for lod in lods]
    table = plt.table(
        cellText=cell_text,
        rowLabels=methods,
        colLabels=["Calculated LOD"],
        loc="bottom",
        cellLoc="center",
        bbox=[0.2, -0.35, 0.6, 0.2],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    plt.subplots_adjust(bottom=0.35)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(out_dir, "lod_all_methods_comparison.png"), dpi=300)
    plt.close()


def setup_run_directory():
    run_dir = f"run_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy("config.yaml", os.path.join(run_dir, "config_log.yaml"))
    return run_dir


def main():
    if not os.path.exists("config.yaml"):
        sys.exit(1)

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    run_dir = setup_run_directory()

    freq_data, cond_data, peak_freqs = load_data(config["dataset"]["path"])

    b_start = config["dataset"].get("baseline_start", 0)
    b_end = config["dataset"].get("baseline_end", 10)

    baseline_cond = cond_data[b_start:b_end]
    mean_blank = np.mean(baseline_cond, axis=0)
    cov_blank = np.cov(baseline_cond, rowvar=False)
    cov_blank += np.eye(cov_blank.shape[0]) * 1e-12

    np.random.seed(42)
    mc_n = min(config["simulation"]["mc_samples"], 10000)
    sim_blanks = np.random.multivariate_normal(mean_blank, cov_blank, mc_n)

    baseline_freqs = peak_freqs[b_start:b_end]
    mu_real = np.mean(baseline_freqs)

    # Calculate absolute shift to handle both peaks and valleys correctly
    shifts = np.abs(peak_freqs - mu_real)
    max_shift_idx = np.argmax(shifts)
    max_shift = shifts[max_shift_idx]

    if max_shift <= 0:
        max_shift = 1.0
    sensitivity = max_shift / config["metadata"]["max_concentration"]

    results = {}

    if config["lod_methods"].get("run_univariate", False):
        lod_uni, _ = run_univariate_lod(
            freq_data, peak_freqs, sim_blanks, sensitivity, config, run_dir
        )
        results["Univariate LOD"] = lod_uni

    if config["lod_methods"].get("run_nas", False):
        lod_nas = run_nas_lod(
            freq_data,
            cond_data,
            peak_freqs,
            mu_real,
            sim_blanks,
            sensitivity,
            config,
            run_dir,
        )
        results["NAS Multivariate LOD"] = lod_nas

    if config["lod_methods"].get("run_predictive", False):
        lod_pred = run_predictive_lod(
            cond_data, peak_freqs, mu_real, sim_blanks, sensitivity, config, run_dir
        )
        results["Predictive LOD (RF)"] = lod_pred

    if len(results) > 1:
        plot_comparison(results, config, run_dir)


if __name__ == "__main__":
    main()
