import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from scipy.stats import norm
import os
import sys
import yaml
import datetime
import shutil

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
        x_fit = f[idx_feat-1:idx_feat+2]
        y_fit = g[idx_feat-1:idx_feat+2]
        p = np.polyfit(x_fit, y_fit, 2)
        if p[0] != 0:
            return -p[1] / (2 * p[0])
    return f0

def load_data(data_file):
    df = pd.read_csv(data_file, sep=r'\s+', header=None)
    freq_data = df.iloc[:, :1000].values
    cond_data = df.iloc[:, 1000:2000].values
    peak_freqs = np.array([extract_peak_freq(freq_data[i], cond_data[i]) for i in range(len(df))])
    return freq_data, cond_data, peak_freqs

def run_univariate_lod(freq_data, peak_freqs, sim_blanks, sensitivity, config, out_dir):
    unit = config['metadata']['concentration_unit']
    fluid = config['metadata']['fluid_type']
    
    freq_axis = freq_data[0]
    sim_peak_freqs = np.array([extract_peak_freq(freq_axis, g) for g in sim_blanks])
    
    mu = np.mean(sim_peak_freqs)
    sigma_blank = np.std(sim_peak_freqs, ddof=1)
    
    delta_f_lod = 3 * sigma_blank
    lod_uni = delta_f_lod / sensitivity
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(sim_peak_freqs, bins=100, density=True, alpha=0.6, color='royalblue', label='Unified Monte Carlo Distribution')
    x = np.linspace(mu - 4*sigma_blank, mu + 4*sigma_blank, 1000)
    pdf = norm.pdf(x, mu, sigma_blank)
    ax.plot(x, pdf, 'k--', linewidth=2, label='Normal Distribution Fit')
    ax.axvline(mu, color='black', linestyle='-', linewidth=2, label=f'Mean (μ): {mu:.2f} Hz')
    
    # Note: Shift direction varies depending on peak vs valley, use absolute bound for plot visualization.
    # To keep it generic, plot both bounds
    ax.axvline(mu - 3*sigma_blank, color='red', linestyle='--', linewidth=2, label=f'LOD Threshold (3σ)')
    ax.axvline(mu + 3*sigma_blank, color='red', linestyle='--', linewidth=2)
    
    summary_text = (
        f"μ = {mu:.2f} ± {sigma_blank:.2f} Hz\n"
        f"LOD Concentration: {lod_uni:.3f} {unit}"
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.95))
    
    ax.set_title(f'Method 1: Univariate Calibration LOD ({fluid})')
    ax.set_xlabel('Extracted Feature Frequency f₀ (Hz)')
    ax.set_ylabel('Probability Density')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'method1_univariate_lod.png'), dpi=300)
    plt.close()
    
    return lod_uni, mu

def run_nas_lod(freq_data, cond_data, peak_freqs, mu_freq, sim_blanks, sensitivity, config, out_dir):
    unit = config['metadata']['concentration_unit']
    fluid = config['metadata']['fluid_type']
    signal_label = config['metadata'].get('signal_label', 'Signal')
    b_start = config['dataset'].get('baseline_start', 0)
    b_end = config['dataset'].get('baseline_end', 10)
    
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
    ax.plot(freq_axis, mean_blank, label='Mean Baseline Spectrum', color='black')
    ax.plot(freq_axis, analyte_cond, label='Maximum Analyte Spectrum', color='blue')
    ax.plot(freq_axis, s_star, label='Net Analyte Signal (NAS) Vector', color='red')
    
    noise_band = 3 * sigma_blank_vec
    ax.fill_between(freq_axis, mean_blank - noise_band, mean_blank + noise_band, color='gray', alpha=0.3, label='Noise Threshold Band (3σ)')
    
    summary_text = (
        f"||s*||: {norm_s_star:.4f}\n"
        f"||σ_blank||: {norm_sigma_blank:.4f}\n"
        f"LOD Concentration: {lod_nas:.3f} {unit}"
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.95))
    
    ax.set_title(f'Method 2: Net Analyte Signal (NAS) Multivariate LOD ({fluid})')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(signal_label)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'method2_nas_multivariate_lod.png'), dpi=300)
    plt.close()
    
    return lod_nas

def run_predictive_lod(cond_data, peak_freqs, mu_freq, sim_blanks, sensitivity, config, out_dir):
    unit = config['metadata']['concentration_unit']
    fluid = config['metadata']['fluid_type']
    
    # Use absolute shift to prevent negative concentrations for valleys
    delta_fs = np.abs(peak_freqs - mu_freq)
    c_all = np.maximum(0, delta_fs / sensitivity)
    
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(cond_data, c_all)
    
    c_preds = rf.predict(sim_blanks)
    
    sigma_c_blank = np.std(c_preds, ddof=1)
    lod_pred = 3 * sigma_c_blank
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(c_preds, bins=100, density=True, alpha=0.6, color='seagreen', label='Predicted Zero-Concentration Samples')
    
    mu_pred = np.mean(c_preds)
    x = np.linspace(mu_pred - 4*sigma_c_blank, mu_pred + 4*sigma_c_blank, 1000)
    pdf = norm.pdf(x, mu_pred, sigma_c_blank)
    ax.plot(x, pdf, 'k--', linewidth=2, label='Normal Distribution Fit')
    
    threshold = mu_pred + 3 * sigma_c_blank
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label='3σ Decision Threshold')
    
    summary_text = (
        f"σ_pred(blank): {sigma_c_blank:.4f} {unit}\n"
        f"LOD Concentration: {lod_pred:.3f} {unit}"
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.95))
    
    ax.set_title(f'Method 3: Predictive LOD via Random Forest Regression ({fluid})')
    ax.set_xlabel(f'Predicted Concentration ({unit})')
    ax.set_ylabel('Probability Density')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'method3_predictive_lod.png'), dpi=300)
    plt.close()
    
    return lod_pred

def plot_comparison(results, config, out_dir):
    unit = config['metadata']['concentration_unit']
    fluid = config['metadata']['fluid_type']
    
    methods = []
    lods = []
    for m, val in results.items():
        methods.append(m)
        lods.append(val)
        
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['royalblue', 'indianred', 'seagreen']
    
    bars = ax.bar(methods, lods, color=colors[:len(methods)], alpha=0.8)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + (max(lods)*0.02), f'{yval:.3f}', ha='center', va='bottom', fontweight='bold')
        
    ax.set_ylabel(f'Limit of Detection ({unit})', fontsize=12)
    ax.set_title(f'Comparison of LOD Concentrations ({fluid})', fontsize=14, fontweight='bold')
    
    cell_text = [[f"{l:.3f} {unit}"] for l in lods]
    table = plt.table(cellText=cell_text, rowLabels=methods, colLabels=['Calculated LOD'],
                      loc='bottom', cellLoc='center', bbox=[0.2, -0.35, 0.6, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    
    plt.subplots_adjust(bottom=0.35)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(out_dir, 'lod_all_methods_comparison.png'), dpi=300)
    plt.close()

def main():
    if not os.path.exists('config.yaml'):
        sys.exit(1)
        
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    run_dir = f"run_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy('config.yaml', os.path.join(run_dir, 'config_log.yaml'))
    
    freq_data, cond_data, peak_freqs = load_data(config['dataset']['path'])
    
    b_start = config['dataset'].get('baseline_start', 0)
    b_end = config['dataset'].get('baseline_end', 10)
    
    baseline_cond = cond_data[b_start:b_end]
    mean_blank = np.mean(baseline_cond, axis=0)
    cov_blank = np.cov(baseline_cond, rowvar=False)
    cov_blank += np.eye(cov_blank.shape[0]) * 1e-12
    
    np.random.seed(42)
    mc_n = min(config['simulation']['mc_samples'], 10000)
    sim_blanks = np.random.multivariate_normal(mean_blank, cov_blank, mc_n)
    
    baseline_freqs = peak_freqs[b_start:b_end]
    mu_real = np.mean(baseline_freqs)
    
    # Calculate absolute shift to handle both peaks and valleys correctly
    shifts = np.abs(peak_freqs - mu_real)
    max_shift_idx = np.argmax(shifts)
    max_shift = shifts[max_shift_idx]
    
    if max_shift <= 0:
        max_shift = 1.0
    sensitivity = max_shift / config['metadata']['max_concentration']
    
    results = {}
    
    if config['lod_methods'].get('run_univariate', False):
        lod_uni, _ = run_univariate_lod(freq_data, peak_freqs, sim_blanks, sensitivity, config, run_dir)
        results['Univariate LOD'] = lod_uni
        
    if config['lod_methods'].get('run_nas', False):
        lod_nas = run_nas_lod(freq_data, cond_data, peak_freqs, mu_real, sim_blanks, sensitivity, config, run_dir)
        results['NAS Multivariate LOD'] = lod_nas
        
    if config['lod_methods'].get('run_predictive', False):
        lod_pred = run_predictive_lod(cond_data, peak_freqs, mu_real, sim_blanks, sensitivity, config, run_dir)
        results['Predictive LOD (RF)'] = lod_pred
        
    if len(results) > 1:
        plot_comparison(results, config, run_dir)

if __name__ == '__main__':
    main()
