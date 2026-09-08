
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

## seocond method
# PCA model 
# The best result in 
def run_nas_lod(
    freq_data, cond_data, peak_freqs, mu_freq, sim_blanks, sensitivity, config, out_dir
):
    unit = config["metadata"]["concentration_unit"]
    fluid = config["metadata"]["fluid_type"]
    signal_label = config["metadata"].get("signal_label", "Signal")
    b_start = config["dataset"].get("baseline_start", 0)
    b_end = config["dataset"].get("baseline_end", 10)

    baseline_cond = cond_data[b_start:b_end]

    # Absolute max shift index determines the highest analyte concentration
    shifts = np.abs(peak_freqs - mu_freq)
    max_shift_idx = np.argmax(shifts)

    analyte_cond = cond_data[max_shift_idx]

    delta_f_analyte = shifts[max_shift_idx]
    c_analyte = delta_f_analyte / sensitivity if delta_f_analyte > 0 else 1.0

    pca = PCA(n_components=min(3, len(baseline_cond)))
    pca.fit(baseline_cond)
    V = pca.components_
    P = V.T @ V
    I = np.eye(baseline_cond.shape[1])
    P_ortho = I - P

    s_star = P_ortho @ analyte_cond
    norm_s_star = np.linalg.norm(s_star)

    mean_blank = np.mean(baseline_cond, axis=0)

    sigma_blank_vec = np.std(sim_blanks, axis=0)
    norm_sigma_blank = np.linalg.norm(sigma_blank_vec)

    scale = 1.0 / c_analyte
    lod_nas = (3 * norm_sigma_blank) / (norm_s_star * scale)

    fig, ax = plt.subplots(figsize=(10, 6))
    freq_axis = freq_data[0]
    ax.plot(freq_axis, mean_blank, label="Mean Baseline Spectrum", color="black")
    ax.plot(freq_axis, analyte_cond, label="Maximum Analyte Spectrum", color="blue")
    ax.plot(freq_axis, s_star, label="Net Analyte Signal (NAS) Vector", color="red")

    noise_band = 3 * sigma_blank_vec
    ax.fill_between(
        freq_axis,
        mean_blank - noise_band,
        mean_blank + noise_band,
        color="gray",
        alpha=0.3,
        label="Noise Threshold Band (3σ)",
    )

    summary_text = (
        f"||s*||: {norm_s_star:.4f}\n"
        f"||σ_blank||: {norm_sigma_blank:.4f}\n"
        f"LOD Concentration: {lod_nas:.3f} {unit}"
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

    ax.set_title(f"Method 2: Net Analyte Signal (NAS) Multivariate LOD ({fluid})")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(signal_label)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "method2_nas_multivariate_lod.png"), dpi=300)
    plt.close()

    return lod_nas
