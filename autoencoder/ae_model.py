import torch
import torch.nn as nn

class SequenceAutoencoder(nn.Module):

    def __init__(self, mode):
        super().__init__()

        self.mode = mode

        self.encoder = nn.Sequential(LinearBlock(4, 32),
                                     LinearBlock(32, 16),
                                     LinearBlock(16, 8),
                                     nn.Linear(8, 1))

        self.compressor = nn.Linear(10, 1)

        self.decompressor = nn.Linear(1, 10)

        self.decoder = nn.Sequential(LinearBlock(1, 8),
                                     LinearBlock(8, 16),
                                     LinearBlock(16, 32),
                                     nn.Linear(32, 4))

    def forward(self, x):

        encoded = self.encoder(x)
        encoded = torch.swapaxes(encoded, 1, 2)
        encoded = self.compressor(encoded)

        if self.mode == "train":
            decoded = self.decompressor(encoded)
            decoded = torch.swapaxes(decoded, 1, 2)
            decoded = self.decoder2(decoded)
        
            return decoded
        else:

            return encoded


def LinearBlock(in_dim, out_dim):

    return nn.Sequential(nn.Linear(in_dim, out_dim),
                         nn.ReLU(),
                         nn.BatchNorm1d(10),
                         nn.AdaptiveAvgPool1d(out_dim))
