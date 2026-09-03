# QCM Monte Carlo LOD Simulation Documentation

This document explains the architecture and functionality of `run_lod_simulations.py`, a script designed to computationally evaluate the Limit of Detection (LOD) for Quartz Crystal Microbalance (QCM) biosensors across three distinct mathematical methods.

## 1. Overview and Purpose

The primary goal of this codebase is to validate the findings of the associated biosensors paper: **that retaining full resonance line-shape morphology (multivariate analysis) yields superior quantitative performance compared to traditional scalar reductions (univariate analysis).**

Instead of performing 10,000 physical experiments to determine the sensor's noise floor and LOD, the script uses a **Unified Monte Carlo Architecture**. It profiles the actual noise in the experimental baseline data, generates 10,000 synthetic noisy spectra, and feeds these exact same spectra into three different LOD calculation methods to compare their robustness.

---

## 2. Core Architecture: The Unified Monte Carlo

The simulation is orchestrated in the `main()` function. The pipeline works as follows:

1. **Baseline Extraction:** It reads the first $N$ rows of the dataset (defined by `baseline_start` and `baseline_end` in `config.yaml`) to represent the "blank" or zero-concentration state.
2. **Noise Profiling:** It calculates the mean spectrum (`mean_blank`) and the covariance matrix (`cov_blank`) across all 1,000 frequency bins. This captures the true, correlated noise behavior of the sensor.
3. **Synthetic Generation:** Using `np.random.multivariate_normal(mean_blank, cov_blank, mc_samples)`, it generates 10,000 new synthetic spectra.
4. **Fair Evaluation:** This single batch of synthetic spectra (`sim_blanks`) is passed to all three methods, ensuring a 1:1 mathematically fair comparison.

---

## 3. Function-by-Function Breakdown

### `extract_peak_freq(f, g)`
**Purpose:** Acts as a traditional scalar feature tracker. Given a frequency array `f` and a signal array `g`, it attempts to find the central frequency of the feature.
**How it works:**
- **Dynamic Detection:** It compares the mean of the edges to the absolute maximum and minimum of the curve to automatically determine if the signal is a "peak" (e.g., Conductance) or a "valley" (e.g., Susceptance B2).
- **Sub-bin Interpolation:** If the peak is not at the absolute edge of the window, it applies a 3-point quadratic polynomial fit (`np.polyfit`) around the extremum to estimate the peak frequency between discrete bins.
- *Note:* For monotonic shapes (slopes), the extremum is always at the edge, so this function consistently returns the edge bin, causing the known "quantization" effect that demonstrates the flaw of univariate tracking.

### `load_data(data_file)`
**Purpose:** Reads the raw `.txt` QCM data files.
**How it works:** Expects space-separated data where columns 0–999 are the frequency axis (Hz) and columns 1000–1999 are the signal amplitudes. It automatically extracts the baseline peak frequencies for initial setup.

### `run_univariate_lod(...)` — (Method 1)
**Purpose:** Calculates LOD using traditional scalar frequency shifts.
**How it works:** 
- Applies `extract_peak_freq` to all 10,000 synthetic spectra.
- Calculates the standard deviation ($\sigma$) of these extracted frequencies.
- $LOD = 3\sigma / \text{Sensitivity}$.
- **Significance:** For well-defined peaks (like Conductance $G$), this works well. For monotonic slopes, it exposes the mathematical weakness of reducing a complex line-shape to a single scalar point, often resulting in quantized, spiky histograms.

### `run_nas_lod(...)` — (Method 2)
**Purpose:** Calculates LOD using Net Analyte Signal (NAS), a multivariate chemometric technique.
**How it works:**
- Uses Principal Component Analysis (PCA) on the baseline data to define the "noise subspace".
- Takes the maximum analyte spectrum and projects it orthogonally to the noise subspace to find the true, isolated analyte signal vector ($s^*$).
- Calculates the multivariate noise norm ($||\sigma||$) from the synthetic blanks.
- $LOD = 3||\sigma|| / ||s^*||$.
- **Significance:** Highly robust across all spectral shapes because it utilizes the entire 1,000-point vector, aligning with the paper's thesis that line-shape retention is superior.

### `run_predictive_lod(...)` — (Method 3)
**Purpose:** Calculates LOD using Machine Learning.
**How it works:**
- Trains a `RandomForestRegressor` on the real dataset, using the full 1,000-point spectra as input features to predict concentration.
- Feeds the 10,000 synthetic blank spectra into the trained Random Forest to see what concentrations it predicts for "zero-concentration" samples.
- Calculates the standard deviation ($\sigma_{pred}$) of these predictions.
- $LOD = 3\sigma_{pred}$.
- **Significance:** Demonstrates the predictive capability of ML models on line-shape data, matching the methodology discussed in the paper.

### `plot_comparison(...)`
**Purpose:** Generates a unified bar chart (`lod_all_methods_comparison.png`) comparing the final calculated LODs of all enabled methods, making it easy to visually confirm which method performs best (typically NAS or Predictive).

---

## 4. Configuration (`config.yaml`)

The script is entirely data-driven via `config.yaml`. This file dictates:
- Which dataset to run (`dataset.path`).
- The rows to use for the baseline (`baseline_start`, `baseline_end`).
- Metadata for plot labeling (`signal_label`, `concentration_unit`).
- Which of the 3 LOD methods to execute (`lod_methods` booleans).

## 5. Output Artifacts

For every execution, the script creates a timestamped folder (e.g., `run_2026-09-03_11-42-42/`) containing:
1. `config_log.yaml`: A backup of the config used for that specific run, ensuring reproducibility.
2. `method1_univariate_lod.png`: Histogram of scalar frequency variation. (if method is true)
3. `method2_nas_multivariate_lod.png`: Plot of the NAS vector and noise bands. (if method is true)
4. `method3_predictive_lod.png`: Histogram of ML-predicted blank concentrations. (if method is true)
5. `lod_all_methods_comparison.png`: Summary bar chart. (if more than one method is true)
