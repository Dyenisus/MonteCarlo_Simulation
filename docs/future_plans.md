# Future Plans and Current Limitations

This document outlines the known limitations of the current QCM Monte Carlo Limit of Detection (LOD) simulation architecture and details the strategic roadmap for future development.

## Current Limitations

### 1. Isolated "2D" Single-Metric Analysis
Currently, the simulation evaluates each impedance metric (e.g., Conductance $G$, Susceptance $B$, Reactance $X$, Phase angle) completely independently in separate runs. We are looking at the dataset one metric at a time. While this is useful for comparing how a single parameter behaves across Univariate, NAS, and ML methods, it is fundamentally an incomplete way to evaluate the sensor's true capability. QCM sensors experience coupled shifts across multiple parameters simultaneously. Analyzing them in isolation deprives the multivariate and ML models of the rich, cross-parameter correlations that define true line-shape analysis.

### 2. Absence of a True Blank Dataset
Because we do not currently possess an extended, isolated "blank" dataset (e.g., a pure buffer run with zero analyte), the simulation relies on the first few rows of the lowest-concentration data to act as a pseudo-blank. We are profiling our baseline noise and covariance matrix from data that may already contain trace amounts of the analyte or initial injection drift. 

---

## Future Roadmap

To align the simulation more closely with the true multi-parameter nature of QCM biosensing and the methodology of the paper, we plan to implement the following improvements:

### 1. Transition to 3D/High-Dimensional Multi-Metric Analysis
We will upgrade the data processing pipeline to look at the sensor data in higher dimensions. Instead of running the NAS or Random Forest algorithms on a single metric's spectrum, we will concatenate multiple metrics (e.g., $G$, $B$, $R$, $X$, and Phase) into a unified, high-dimensional feature space for each sweep. By analyzing these metrics simultaneously (3D+ analysis), the machine learning models will be able to leverage the physical cross-sensitivities of the quartz crystal, which should significantly improve predictive accuracy and lower the theoretical LOD.

### 2. Integration of Actual Blank Data
In future experiments, we will dedicate time to acquiring a true, extended baseline dataset using only the buffer (e.g., 1x PBS) before any concentration gradients or target DNA are introduced. Integrating this actual blank data will allow the Monte Carlo simulation to accurately capture the true noise floor of the system. This will provide a rigorously clean covariance matrix for the synthetic noise generation, ensuring that our calculated $\sigma_{blank}$ strictly represents systemic and environmental noise rather than trace analyte fluctuations.
