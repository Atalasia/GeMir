import torch
from torch import nn
from torch.nn import functional as F

class GeMir(nn.Module):

    def __init__(self):
        super(GeMir, self).__init__()

        self.mirna = Mirna_Module()
        self.mrna = Gene_Module(64)

        self.classifier = nn.Sequential(
            nn.Linear(323, 64),
            nn.Linear(64, 1)
        )

    def forward(self, mirna_x, mrna_x):

        mirna_x = self.mirna(mirna_x)
        mrna_x = self.mrna(mrna_x)
        mrna_x = torch.flatten(mrna_x, 1)

        output = self.classifier(torch.cat((mirna_x, mrna_x), 1))
        output = torch.flatten(output, 0)

        return output


class Mirna_Module(nn.Module):

    def __init__(self):
        super(Mirna_Module, self).__init__()
        self.conv = nn.Conv1d(1, 8, kernel_size=1)
        self.lin = nn.Linear(24, 3)
        self.do = nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.do(x)
        x = self.lin(x)

        return x


class Gene_Module(nn.Module):

    def __init__(self, h_dim):
        super(Gene_Module, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(3, h_dim, 6, padding=7),
            Residual(ConvBlock(h_dim, h_dim)),
            AttentionPool(h_dim)
        )

        self.conv_tower = nn.Sequential(
            ConvTowerBlock(h_dim, 2 * h_dim),
            ConvTowerBlock(2 * h_dim, 4 * h_dim),
            ConvTowerBlock(4 * h_dim, h_dim),
            ConvTowerBlock(h_dim, h_dim // 2)
        )

        self.conv_block = ConvBlock(h_dim // 2, 4 * h_dim, 1)

        self.classify = nn.Sequential(
            nn.Dropout(0.0125),
            nn.GELU(),
            nn.Linear(4 * h_dim, 10),
            nn.Softplus()
        )

    def forward(self, x):

        x = self.stem(x)
        x = self.conv_tower(x)

        x = torch.swapaxes(x, 1, 2)
        x = self.conv_block(x)
        x = torch.swapaxes(x, 1, 2)

        x = self.classify(x)

        return x


class AttentionPool(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self.attn_logit = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        b, d, n = x.shape
        rem = n % 2


        if rem > 0:
            x = F.pad(x, (0, rem), value=0)
            x = x.view(b, d, -1, 2)

            mask = torch.zeros((b, 1, n), dtype=torch.bool, device=x.device)
            mask = F.pad(mask, (0, rem), value=True)
            mask = mask.view(b, 1, -1, 2)

            logits = self.attn_logit(x)
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(mask, mask_value)

        else:
            x = x.view(b, d, -1, 2)
            logits = self.attn_logit(x)

        attn = logits.softmax(dim=-1)
        x = (x * attn).sum(dim=-1)

        return x


def ConvBlock(in_dim, out_dim, kernel_size=1):

    return nn.Sequential(
        nn.BatchNorm1d(in_dim),
        nn.GELU(),
        nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding = kernel_size // 2)
    )


def ConvTowerBlock(in_dim, out_dim):

    return nn.Sequential(
            ConvBlock(in_dim, out_dim, kernel_size=5),
            Residual(ConvBlock(out_dim, out_dim, 1)),
            AttentionPool(out_dim)
    )


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
