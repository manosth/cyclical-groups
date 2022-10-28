# system imports
import os
import time
from datetime import datetime
from skimage import io

# pythom imports
import numpy as np
import scipy.io as sio

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset, random_split

import torchvision
import torchvision.datasets as ds
import torchvision.transforms.functional as tf

from model import GroupActionUntiedLearn

def log_gradients_in_model(model, logger, step):
    for tag, value in model.named_parameters():
        if value.grad is not None:
            logger.add_histogram(tag + "/grad", value.grad.cpu(), step)

class custom_loss(torch.nn.Module):
    def __init__(self, lam_loss, base_loss=torch.nn.MSELoss()):
        super(custom_loss, self).__init__()
        self.base_loss = base_loss
        self.lam_loss = lam_loss

    def forward(self, x, out, code, params):
        base_loss = self.base_loss(x, out)
        # l1_loss = self.lam_loss * code.abs().sum(dim=1).mean()
        # the thing below looks digusting, but basically you take the norm over each filtermap
        # (you could also do the sum, this is personal preference I think, there is no consensus).
        # If there are no filtermaps and we have a dense model, then this will just give the absolute
        # value. Then, you have to take the norm over the group (in which case, if it is a sparse model
        # then this dimension will be one and again nothing will happen). Finally, you take the sum over
        # the number of groups, finalizing the L2,1 norm (and in the case of sparsity just L1). Then, you just take the mean over the batch.
        orig_shape = code.shape
        code = code.view(orig_shape[0], params.num_groups, params.group_size, orig_shape[2], orig_shape[3])
        l1_loss = self.lam_loss * code.abs().sum(dim=(-2, -1)).norm(dim=2).sum(dim=1).mean()
        loss = base_loss + l1_loss
        return loss

def report_statistics(start, idx, total_len, val=0.0):
    current = time.time()
    total = current - start
    seconds = int(total % 60)
    minutes = int((total // 60) % 60)
    hours = int((total // 60) // 60)

    if idx == -1:
        print("")
        print(f"Total time elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}")
    else:
        remain = (total_len - idx - 1) / (idx + 1) * total
        seconds_r = int(remain % 60)
        minutes_r = int((remain // 60) % 60)
        hours_r = int((remain // 60) // 60)
        print(f"progress: {(idx + 1) / total_len * 100:5.2f}%\telapsed: {hours:02d}:{minutes:02d}:{seconds:02d}\tremaining: {hours_r:02d}:{minutes_r:02d}:{seconds_r:02d}\tval: {val}", end="\r")

class Names():
        def __init__(self, params):
            name = str(params.model) + "_groups=" + str(params.group_size) + "_kernel=" + str(params.kernel_size) + "_stride=" + str(params.stride) + "_layers=" + str(params.num_layers) + "_step=" + str(params.step) + "_lam=" + str(params.lambda_) + "_lamloss=" + str(params.lam_loss) + "_lr=" + str(params.lr)

            date = datetime.now().strftime("%Y_%m_%d_T%H%M%S")
            model = name + "_" + date

            path = "results/" + params.dataset
            if params.n_channels > 1:
                path += "color_"
            else:
                path += "_"

            if params.kernel_size < params.input_width:
                path += "conv_"

            self.model = model
            self.path = path + model + "/"

def gen_names(params):
    names = Names(params)
    return names

def gen_loaders(params, workers):
    if params.dataset == "mnist":
        X_tr, Y_tr, X_te, Y_te = load_mnist(params)
    elif params.dataset == "rmnist":
        X_tr, Y_tr, X_te, Y_te = load_rmnist(params)
    else:
        if params.n_channels > 1:
            X_tr, Y_tr, X_te, Y_te = load_cifar(color=True)
        else:
            X_tr, Y_tr, X_te, Y_te = load_cifar()
    train_dl = make_loader(TensorDataset(X_tr, Y_tr), batch_size=params.batch_size, num_workers=workers)
    test_dl = make_loader(TensorDataset(X_te, Y_te), batch_size=params.batch_size, num_workers=workers)
    return train_dl, test_dl

def gen_model(params, device, init_B=None):
    # for now, just hack it out
    return GroupActionUntiedLearn(params, device).to(device)

def standardize(X):
    "Expects data in NxCxWxH."
    means = X.mean(axis=(0,2,3))
    std = X.std(axis=(0,2,3))

    X = torchvision.transforms.Normalize(means, std)(X)
    return X

def whiten(X, eps=1e-8):
    "Expects data in NxCxWxH."
    os = X.shape


    X = X.reshape(os[0], -1)
    cov = np.cov(X, rowvar=False)
    U, S, V = np.linalg.svd(cov)

    zca = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + eps)), U.T))
    X = torch.Tensor(np.dot(X, zca).reshape(os))
    return X

def load_mnist(params, datadir="~/data", five_digits=False):
    train_ds = ds.MNIST(root=datadir, train=True, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                             ]))
    test_ds = ds.MNIST(root=datadir, train=False, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                             ]))
    def to_xy(dataset):
        Y = dataset.targets.long()
        # this size is necessary to work with the matmul broadcasting when using channels
        X = dataset.data.view(dataset.data.shape[0], params.n_channels, params.input_width, -1) / 255.0
        return X, Y

    def get_five_digits(X, Y):
        digit_0 = (Y == 0)
        digit_3 = (Y == 3)
        digit_4 = (Y == 4)
        digit_6 = (Y == 6)
        digit_7 = (Y == 7)
        indexes = digit_0 | digit_3 | digit_4 | digit_6 | digit_7
        return X[indexes], Y[indexes]

    X_tr, Y_tr = to_xy(train_ds)
    X_te, Y_te = to_xy(test_ds)

    X_tr = standardize(X_tr)
    X_te = standardize(X_te)

    X_tr = whiten(X_tr)
    X_te = whiten(X_te)

    return X_tr, Y_tr, X_te, Y_te

def load_rmnist(params, datadir="/home/manos/data/rmnist/data.mat"):
    data = sio.loadmat(datadir)
    X_tr = torch.Tensor(data['x'])
    Y_tr = torch.Tensor(data['y']).squeeze().long()

    X_te = torch.Tensor(data['x_test'])
    Y_te = torch.Tensor(data['y_test']).squeeze().long()

    X_tr = standardize(X_tr)
    X_te = standardize(X_te)

    X_tr = whiten(X_tr)
    X_te = whiten(X_te)

    return X_tr, Y_tr, X_te, Y_te

def load_cifar(datadir='~/data', three_class=False, color=False):
    train_ds = ds.CIFAR10(root=datadir, train=True,
                           download=True, transform=None)
    test_ds = ds.CIFAR10(root=datadir, train=False,
                          download=True, transform=None)

    def to_xy(dataset):
        Y = torch.Tensor(np.array(dataset.targets)).long()
        X = torch.Tensor(np.transpose(dataset.data, (0, 3, 1, 2))).float() / 255.0 # [0, 1]
        if not color:
            X = torchvision.transforms.Grayscale()(X).view(X.shape[0], 1, X.shape[2], -1)
        return X, Y

    def get_three_classes(X, Y):
        cats = (Y == 3)
        horses = (Y == 7)
        boats = (Y == 8)

        indexes = cats | horses | boats
        return X[indexes], Y[indexes]

    X_tr, Y_tr = to_xy(train_ds)
    X_te, Y_te = to_xy(test_ds)

    X_tr = standardize(X_tr)
    X_te = standardize(X_te)

    X_tr = whiten(X_tr)
    X_te = whiten(X_te)
    return X_tr, Y_tr, X_te, Y_te

def make_loader(dataset, shuffle=True, batch_size=128, num_workers=4):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=True)
