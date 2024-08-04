from torch.utils.data import Dataset

import numpy as np
import pandas as pd


class SequenceDataset(Dataset):

    def __init__(self, seq_file):
        self.df = pd.read_csv(seq_file, sep='\t', header=None)
        self.length = len(self.df)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_id = self.df.iloc[idx, 0]
        seq = self.df.iloc[idx, 1]
        seq = conversion(seq)

        return data_id, seq


def conversion(seq: str):

    seq = seq.upper()
    result = list()
    for base in seq:
        if base == "A":
            val = np.array([1.0, 0.0, 0.0, 0.0])
            result.append(val)
        elif base == "C":
            val = np.array([0.0, 1.0, 0.0, 0.0])
            result.append(val)
        elif base == "G":
            val = np.array([0.0, 0.0, 1.0, 0.0])
            result.append(val)
        elif base == "U":
            val = np.array([0.0, 0.0, 0.0, 1.0])
            result.append(val)
        elif base == "0":
            val = np.array([0.0, 0.0, 0.0, 0.0])
            result.append(val)
        else:
            val = np.array([0.0, 0.0, 0.0, 0.0])
            result.append(val)

    result = np.array(result)

    return result
