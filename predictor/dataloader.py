from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import torch
import math

class PairedMatchDataset(Dataset):

    def __init__(self, pos_csv_file, neg_csv_file):
        pos_df = pd.read_csv(pos_csv_file, sep='\t', header=None)
        neg_df = pd.read_csv(neg_csv_file, sep='\t', header=None)

        self.pos_enc = get_pos_enc()

        self.df = pd.concat([pos_df, neg_df], ignore_index=True)
        self.length = len(self.df)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        mirna_id = self.df.iloc[idx, 0]
        gene_id = self.df.iloc[idx, 1]
        bind_block = self.df.iloc[idx, 2]

        pos_enc_block = self.pos_enc[bind_block]
        pos_enc_block = torch.swapaxes(pos_enc_block, 0, 1)

        mirna_block = torch.tensor(np.array([self.df.iloc[idx, 3].split(",")], dtype=float), dtype=torch.float32)
        gene_block = torch.tensor(np.array([self.df.iloc[idx, 4].split(",")], dtype=float), dtype=torch.float32)
        zeros = torch.zeros(1000, dtype=torch.float32)

        full_gene_block = torch.concat((gene_block, pos_enc_block, zeros), 0)
        val = self.df.iloc[idx, 6]

        return mirna_id, gene_id, bind_block, mirna_block, full_gene_block, val


def get_pos_enc():

    d_model = 1
    seq_len = 300000

    position = torch.arange(seq_len).unsqueeze(1)

    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(300000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    out_dict = dict()

    for idx, i in enumerate(range(0, seq_len, 1000)):
        out_dict[idx] = pe[idx:idx + 1000]

    return out_dict