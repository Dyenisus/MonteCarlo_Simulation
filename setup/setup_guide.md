## Installation and Setup Guide

To run this simulation on a new machine, you will need Python 3.8 or newer. We recommend using a virtual environment.

### Step 1: Create a Virtual Environment (Optional but recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows
```

### Step 2: Install Required Packages
A `requirements.txt` file is included. Install the dependencies using pip:
```bash
pip install -r setup/requirements.txt
```

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
python run_lod_simulations.py
```
The script will automatically create a new timestamped `run_YYYY-MM-DD_HH-MM-SS/` directory containing all your generated graphs and a backup of the configuration.
