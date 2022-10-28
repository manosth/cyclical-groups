# systme imports
import os
from datetime import datetime
import time

# pythom imports
import numpy as np
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.datasets as ds
import torchvision.transforms.functional as tf

# file imports
from utils import *
from plot_utils import *
from config import Params

if __name__ == '__main__':
    params = Params()

    # create model and loaders
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    workers = max(4 * torch.cuda.device_count(), 4)

    names = gen_names(params)
    model = gen_model(params, device)
    train_dl, test_dl = gen_loaders(params, workers)

    # housekeeping params
    times_per_epoch = 10
    report_period = len(train_dl) // times_per_epoch
    plot_period = 5

    train_log = names.path + "trainlog_" + names.model + ".txt"
    test_log = names.path + "testlog_" + names.model + ".txt"
    train_acc_log = names.path + "trainclass_" + names.model + ".txt"
    test_acc_log = names.path + "testclass_" + names.model + ".txt"

    os.makedirs(names.path)
    with open(names.path + 'params.txt', 'w') as file:
        file.write(str(vars(params)))

    filters_path = names.path + "figs/filters/"
    os.makedirs(filters_path, exist_ok=True)

    if params.tensorboard:
        writer = SummaryWriter(names.path)
        writer.add_text("params", str(vars(params)), global_step=0)

    # training params
    opt = optim.Adam(model.parameters(), lr=params.lr, eps=params.eps)
    schd = optim.lr_scheduler.MultiStepLR(opt, [int(1/3 * params.epochs), int(2/3 * params.epochs)], gamma=1/3)
    loss_func = torch.nn.CrossEntropyLoss()

    for k in range(params.num_layers):
        gen_b = model.B[k].clone().detach()
        for idx in range(1, params.num_groups):
            gen_b = torch.cat((gen_b, tf.rotate(model.B[k].clone().detach(), params.ga * idx)), dim=0)
        gen_b = gen_b.data.cpu().numpy()
        save_conv_dictionary(gen_b, params, 0, k, filters_path, names)

    # training
    total_train = params.epochs * (len(train_dl) + len(test_dl))
    start = time.time()
    local = time.localtime()
    print(f"Starting iterations...\t(Start time: {local[3]:02d}:{local[4]:02d}:{local[5]:02d})")
    for epoch in range(1, params.epochs + 1):
        net_loss = 0.0
        n_correct = 0
        n_total = 0

        model.train()
        for idx, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)

            out, code = model(x)
            loss = loss_func(out, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            # log_gradients_in_model(model, writer, epoch * len(train_dl) + idx)
            opt.step()

            net_loss += loss.item() * len(x)
            n_total += len(x)
            with torch.no_grad():
                model.normalize()
                _, preds = torch.max(out, dim=1)
                n_correct += (preds == y).float().sum().cpu()

            if idx % report_period == 0:
                train_acc = n_correct / n_total
                curr_train = (epoch - 1) * (len(train_dl) + len(test_dl)) + idx
                # reset the buffer
                report_statistics(start, curr_train, total_train, val="")
                report_statistics(start, curr_train, total_train, val=np.round(train_acc, 4))
        train_loss = net_loss / n_total
        train_acc = n_correct / n_total

        # save dicts
        if (epoch % plot_period) == 0:
            for k in range(params.num_layers):
                gen_b = model.B[k].clone().detach()
                for jdx in range(1, params.num_groups):
                    gen_b = torch.cat((gen_b, tf.rotate(model.B[k].clone().detach(), params.ga * jdx)), dim=0)
                gen_b = gen_b.data.cpu().numpy()
                save_conv_dictionary(gen_b, params, epoch, k, filters_path, names)

        net_loss = 0.0
        n_correct = 0
        n_total = 0
        model.eval()
        with torch.no_grad():
            for idx, (x, y) in enumerate(test_dl):
                x, y = x.to(device), y.to(device)

                out, code = model(x)
                loss = loss_func(out, y)

                net_loss += loss.item() * len(x)
                n_total += len(x)
                with torch.no_grad():
                    _, preds = torch.max(out, dim=1)
                    n_correct += (preds == y).float().sum().cpu()

                if idx % report_period == 0:
                    test_acc = n_correct / n_total
                    curr_train = epoch * len(train_dl) + (epoch - 1) * len(test_dl) + idx
                    report_statistics(start, curr_train, total_train, val=np.round(test_acc, 4))

        test_loss = net_loss / n_total
        test_acc = n_correct / n_total

        with open(train_log, "a") as file:
            file.write(str(train_loss) + "\n")
        with open(train_acc_log, "a") as file:
            file.write(str(train_acc) + "\n")

        with open(test_log, "a") as file:
            file.write(str(test_loss) + "\n")
        with open(test_acc_log, "a") as file:
            file.write(str(test_acc) + "\n")

        if params.tensorboard:
            writer.add_scalar("classifier_test_loss", test_loss, epoch + 1)
            writer.add_scalar("classifier_test_acc", test_acc, epoch + 1)
            writer.add_scalar("classifier_train_loss", train_loss, epoch + 1)
            writer.add_scalar("classifier_train_acc", train_acc, epoch + 1)
        schd.step()
    report_statistics(start, -1, total_train)
    if params.tensorboard:
        writer.close()

    # save model for visualization afterwards
    torch.save(model.state_dict(), names["path"] + names["model"] + ".pth")
