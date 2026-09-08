import numpy as np
import pandas as pd

from .prep.extract_peak import extract_peak_freq


def load_data(data_file):
    df = pd.read_csv(data_file, sep=r"\s+", header=None)
    freq_data = df.iloc[:, :1000].values
    cond_data = df.iloc[:, 1000:2000].values
    peak_freqs = np.array(
        [extract_peak_freq(freq_data[i], cond_data[i]) for i in range(len(df))]
    )
    return freq_data, cond_data, peak_freqs

