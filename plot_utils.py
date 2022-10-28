# system imports
import os

# pythom imports
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# torch imports
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image

def save_conv_dictionary(W, params, epoch, k, save_path, names, cmap="gray"):
    # create a reasonable size for the image
    p = params.group_size * params.num_groups
    a = np.int(np.ceil(np.sqrt(p)))
    # note: matplotlib and gridspec have opposite notations for sizes
    # fig = plt.figure(figsize=(params.group_size, params.num_groups))
    # gs1 = gridspec.GridSpec(params.num_groups, params.group_size)
    fig = plt.figure(figsize=(a, a))
    gs1 = gridspec.GridSpec(a, a)
    gs1.update(wspace=0.0025, hspace=0.05)
    W -= np.min(W)
    W /= np.max(W)
    for col in range(p):
        ax1 = plt.subplot(gs1[col])
        wi = W[col]
        wi = np.transpose(wi, (1,2,0)).squeeze()
        plt.imshow(wi, cmap=cmap)
        plt.axis("off")
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect("equal")
        plt.subplots_adjust(wspace=None, hspace=None)
    # plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
    plt.savefig(save_path + "filters_" + names.model + "_k=" + str(k) + "_epoch=" + str(epoch) + ".pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close()

def save_whole_net(W, params, epoch, save_path, names, cmap="gray"):
    # create a reasonable size for the image
    p = params.num_layers * params.group_size
    a = np.int(np.ceil(np.sqrt(p)))
    # note: matplotlib and gridspec have opposite notations for sizes
    fig = plt.figure(figsize=(params.group_size, params.num_layers))
    gs1 = gridspec.GridSpec(params.num_layers, params.group_size)
    gs1.update(wspace=0.0025, hspace=0.05)
    for layer in range(params.num_layers):
        wi = W[layer].clone().detach().cpu().numpy()
        wi -= np.min(wi)
        wi /= np.max(wi)
        for col in range(params.group_size):
            ax1 = plt.subplot(gs1[col + layer * params.group_size])
            wi_p = wi[col]
            wi_p = np.transpose(wi_p, (1,2,0)).squeeze()
            plt.imshow(wi_p, cmap=cmap)
            plt.axis("off")
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect("equal")
            plt.subplots_adjust(wspace=None, hspace=None)
    # plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
    plt.savefig(save_path + "filters_" + names.model + "_all_epoch=" + str(epoch) + ".pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close()

def save_conv_encoding(encoding, params, path, to_save, cmap=None):
    p = params.group_size * params.num_groups
    a = np.int(np.ceil(np.sqrt(p)))
    # note: matplotlib and gridspec have opposite notations for sizes
    fig = plt.figure(figsize=(params.group_size, params.num_groups))
    gs1 = gridspec.GridSpec(params.num_groups, params.group_size)
    gs1.update(wspace=0.025, hspace=0.05)
    E = encoding.clone().detach().cpu().numpy()
    E -= np.min(E)
    E /= np.max(E)
    for col in range(p):
        ax1 = plt.subplot(gs1[col])
        wi = E[col, :, :]
        #wi = np.transpose(wi, (1,2,0)).squeeze()
        if cmap is not None:
            plt.imshow(wi, vmin=0, vmax=1, cmap=cmap)
        else:
            plt.imshow(wi, vmin=0, vmax=1)
        plt.axis("off")
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect("equal")
        plt.subplots_adjust(wspace=None, hspace=None)
    # plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
    plt.savefig(path + "encoder_size=" + str(params.group_size) + "_kernel=" + str(params.kernel_size) + "_stride=" + str(params.stride) + "_lam=" + str(params.lambda_) + "_init=" + str(params.init_mode) + "_batch=" + str(params.batch_size) + "_img=" + str(to_save) + ".pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close()
