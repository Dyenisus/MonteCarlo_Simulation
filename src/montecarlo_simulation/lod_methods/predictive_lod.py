import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor

## Third method the use different models (forest regression)
def run_predictive_lod(
    cond_data, peak_freqs, mu_freq, sim_blanks, sensitivity, config, out_dir
):
    unit = config["metadata"]["concentration_unit"]
    fluid = config["metadata"]["fluid_type"]

    # Use absolute shift to prevent negative concentrations for valleys
    delta_fs = np.abs(peak_freqs - mu_freq)
    c_all = np.maximum(0, delta_fs / sensitivity)

    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(cond_data, c_all)

    c_preds = rf.predict(sim_blanks)

    sigma_c_blank = np.std(c_preds, ddof=1)
    lod_pred = 3 * sigma_c_blank

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        c_preds,
        bins=100,
        density=True,
        alpha=0.6,
        color="seagreen",
        label="Predicted Zero-Concentration Samples",
    )

    mu_pred = np.mean(c_preds)
    x = np.linspace(mu_pred - 4 * sigma_c_blank, mu_pred + 4 * sigma_c_blank, 1000)
    pdf = norm.pdf(x, mu_pred, sigma_c_blank)
    ax.plot(x, pdf, "k--", linewidth=2, label="Normal Distribution Fit")

    threshold = mu_pred + 3 * sigma_c_blank
    ax.axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label="3σ Decision Threshold",
    )

    summary_text = (
        f"σ_pred(blank): {sigma_c_blank:.4f} {unit}\n"
        f"LOD Concentration: {lod_pred:.3f} {unit}"
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

    ax.set_title(f"Method 3: Predictive LOD via Random Forest Regression ({fluid})")
    ax.set_xlabel(f"Predicted Concentration ({unit})")
    ax.set_ylabel("Probability Density")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "method3_predictive_lod.png"), dpi=300)
    plt.close()

    return lod_pred

