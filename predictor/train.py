from dataloader import PairedMatchDataset
from model import GeMir

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import time

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


def metric_batch(output, target):

    pred = torch.where(output < 0, 0.0, output)
    pred = torch.where(pred > 1.0, 1.0, pred)
    pred = torch.round(pred)

    corrects = pred.eq(target.view_as(pred)).sum().item()

    return corrects


def loss_batch(loss_func, output, target, opt=None):

    loss = loss_func(output, target)
    metric_b = metric_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b


def loss_epoch(model, loss_func, dataset_dl, device, opt=None):

    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for d in dataset_dl:

        mirna_id, gene_id, bind_block, mirna_block, full_gene_block, val = d

        mirna_block = mirna_block.to(device, dtype=torch.float)
        full_gene_block = full_gene_block.to(device, dtype=torch.float)
        yb = val.to(device, dtype=torch.float)

        output = model(mirna_block, full_gene_block)
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)

        running_loss += loss_b

        if metric_b is not None:
            running_metric += metric_b

    loss = running_loss / len_data
    metric = running_metric / len_data

    return loss, metric


def train_val(model, params):
    num_epochs = params['num_epochs']
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]
    device = params["device"]

    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    best_loss = float('inf')

    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, device, opt)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, device)
        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), path2weights)
            print('model saved')

        lr_scheduler.step(val_loss)

        print('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time: %.4f min' % (
        train_loss, val_loss, 100 * val_metric, (time.time() - start_time) / 60))
        print('-' * 10)

    return model, loss_history, metric_history


def main():

    device = torch.device("cuda:0")

    seq_dataset_train = PairedMatchDataset(pos_csv_file="pos_train_data.txt", neg_csv_file="neg_train_data.txt")
    seq_dataset_val = PairedMatchDataset(pos_csv_file="pos_test_data.txt", neg_csv_file="neg_test_data.txt")

    train_loader = DataLoader(seq_dataset_train, batch_size=16, shuffle=True, num_workers=8)
    test_loader = DataLoader(seq_dataset_val, batch_size=16, shuffle=True, num_workers=8)

    model = GeMir()
    model.to(device)

    print("Starting Training")

    loss_func = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=0.0004)

    lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)

    params_train = {
        'num_epochs': 70,
        'optimizer': opt,
        'loss_func': loss_func,
        'train_dl': train_loader,
        'val_dl': test_loader,
        'lr_scheduler': lr_scheduler,
        'path2weights': './gemir.pt',
        'device':device
    }

    model, loss_hist, metric_hist = train_val(model, params_train)
    num_epochs = params_train["num_epochs"]

    plt.title("Train-Val Loss")
    plt.plot(range(1, num_epochs + 1), loss_hist["train"], label="train")
    plt.plot(range(1, num_epochs + 1), loss_hist["val"], label="val")
    plt.ylabel("Loss")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.savefig("./gemir_loss.png")
    plt.clf()

    plt.title("Train-Val Accuracy")
    plt.plot(range(1, num_epochs + 1), metric_hist["train"], label="train")
    plt.plot(range(1, num_epochs + 1), metric_hist["val"], label="val")
    plt.ylabel("Accuracy")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.savefig("./gemir_acc.png")

main()
