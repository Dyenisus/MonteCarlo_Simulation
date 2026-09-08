import numpy as np

def extract_peak_freq(f, g):
    # DYNAMIC FEATURE TRACKER
    # Determine if the curve is a peak (Conductance/Resistance)
    # or a valley (Susceptance B2) based on edge mean vs extrema.
    mean_edge = (np.mean(g[:50]) + np.mean(g[-50:])) / 2.0
    val_max = np.max(g)
    val_min = np.min(g)

    if abs(val_max - mean_edge) > abs(val_min - mean_edge):
        idx_feat = np.argmax(g)
    else:
        idx_feat = np.argmin(g)

    f0 = f[idx_feat]

    # Quadratic fit around the extracted feature (if not at edge)
    if 0 < idx_feat < len(g) - 1:
        x_fit = f[idx_feat - 1 : idx_feat + 2]
        y_fit = g[idx_feat - 1 : idx_feat + 2]
        p = np.polyfit(x_fit, y_fit, 2)
        if p[0] != 0:
            return -p[1] / (2 * p[0])
    return f0

