import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from ..prep.extract_peak import extract_peak_freq

## Contains errors: not enough
## the moetdhology used is not enough to determine the concentration
# Worst result compared to other
def run_univariate_lod(freq_data, peak_freqs, sim_blanks, sensitivity, config, out_dir):
    unit = config["metadata"]["concentration_unit"]
    fluid = config["metadata"]["fluid_type"]

    freq_axis = freq_data[0]
    sim_peak_freqs = np.array([extract_peak_freq(freq_axis, g) for g in sim_blanks])

    mu = np.mean(sim_peak_freqs)
    sigma_blank = np.std(sim_peak_freqs, ddof=1)

    delta_f_lod = 3 * sigma_blank
    lod_uni = delta_f_lod / sensitivity

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        sim_peak_freqs,
        bins=100,
        density=True,
        alpha=0.6,
        color="royalblue",
        label="Unified Monte Carlo Distribution",
    )
    x = np.linspace(mu - 4 * sigma_blank, mu + 4 * sigma_blank, 1000)
    pdf = norm.pdf(x, mu, sigma_blank)
    ax.plot(x, pdf, "k--", linewidth=2, label="Normal Distribution Fit")
    ax.axvline(
        mu, color="black", linestyle="-", linewidth=2, label=f"Mean (μ): {mu:.2f} Hz"
    )

    # Note: Shift direction varies depending on peak vs valley, use absolute bound for plot visualization.
    # To keep it generic, plot both bounds
    ax.axvline(
        mu - 3 * sigma_blank,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"LOD Threshold (3σ)",
    )
    ax.axvline(mu + 3 * sigma_blank, color="red", linestyle="--", linewidth=2)

    summary_text = (
        f"μ = {mu:.2f} ± {sigma_blank:.2f} Hz\nLOD Concentration: {lod_uni:.3f} {unit}"
    )
    ax.text(
        0.05,
        0.95,
        summary_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.95),
    )

    ax.set_title(f"Method 1: Univariate Calibration LOD ({fluid})")
    ax.set_xlabel("Extracted Feature Frequency f₀ (Hz)")
    ax.set_ylabel("Probability Density")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "method1_univariate_lod.png"), dpi=300)
    plt.close()

    return lod_uni, mu
