
## Installation and Setup Guide

To run this simulation on a new machine, you will need Python 3.14 or newer, as
specified by the project configuration. uv manages the project virtual environment.

### Step 1: Install uv
Install uv by following the instructions at https://docs.astral.sh/uv/getting-started/installation/.

### Step 2: Create the Environment and Install Required Packages
```bash
uv sync
```

This creates the project's `.venv`, resolves dependencies from `pyproject.toml`,
and records the exact resolution in `uv.lock`. To run commands in the managed
environment without manually activating it, use `uv run`:

The primary packages used in this project are:
- `numpy` (>=1.24.0) - For numerical matrix operations and noise generation.
- `pandas` (>=2.0.0) - For loading and parsing the raw QCM text data files.
- `scipy` (>=1.10.0) - For statistical probability density functions (Norm distribution fits).
- `scikit-learn` (>=1.2.0) - For Principal Component Analysis (NAS) and Random Forest (Predictive ML).
- `matplotlib` (>=3.7.0) - For rendering and saving the LOD comparison charts.
- `PyYAML` (>=6.0) - For reading the `config.yaml` configuration file.

### Step 3: Run the Simulation
Ensure that your `config.yaml` is properly set up and points to a valid dataset (e.g., `data/data4/G_data.txt`). Then, simply run:
```bash
uv run montecarlo-simulation
```
The script will automatically create a new timestamped `run_YYYY-MM-DD_HH-MM-SS/` directory containing all your generated graphs and a backup of the configuration.
