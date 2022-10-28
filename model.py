import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tf

def calc_pad_sizes(stride, kernel_size, height, width):
    top_pad = stride
    if (height + top_pad - kernel_size) % stride == 0:
        bot_pad = 0
    else:
        bot_pad = stride - (height + top_pad - kernel_size) % stride
    left_pad = stride
    if (width + left_pad - kernel_size) % stride == 0:
        right_pad = 0
    else:
        right_pad = stride - (width + left_pad - kernel_size) % stride
    right_pad += stride
    bot_pad += stride
    return left_pad, right_pad, top_pad, bot_pad

class GroupActionUntiedLearn(nn.Module):
    def __init__(self, params, device):
        super(GroupActionUntiedLearn, self).__init__()

        self.group_size = params.group_size
        self.num_groups = params.num_groups
        self.kernel_size = params.kernel_size
        self.stride = params.stride
        self.ga = params.ga

        self.num_layers = params.num_layers
        self.step = params.step

        self.pooler = nn.AdaptiveAvgPool2d(params.pool_size)
        self.fc = nn.Linear(self.group_size * self.num_groups * params.pool_size * params.pool_size, params.n_classes)

        left, right, top, bot = calc_pad_sizes(self.stride, self.kernel_size, params.input_height, params.input_width)
        self.pad = (left, top)
        self.h_out = int(np.floor((params.input_height + 2 * self.pad[0] - params.kernel_size) / self.stride + 1))
        self.w_out = int(np.floor((params.input_width + 2 * self.pad[1] - params.kernel_size) / self.stride + 1))

        conv2d_out = (self.h_out - 1) * self.stride - 2 * self.pad[0] + params.kernel_size
        self.output_pad = np.max(params.input_height - conv2d_out, 0)

        self.device = device

        B = []
        for layer in range(self.num_layers):
            B_i = torch.randn(
                (self.group_size, params.n_channels, params.kernel_size, params.kernel_size),
                device=self.device
            )
            B_i = F.normalize(B_i, p="fro", dim=(-1, -2))
            B.append(nn.Parameter(B_i))
        self.B = nn.ParameterList(B)

        self.bn = nn.ModuleList([nn.BatchNorm2d(self.group_size * self.num_groups) for idx in range(self.num_layers - 1)])

        lambda_ = []
        for layer in range(self.num_layers):
            lambda_.append(nn.Parameter(torch.ones(1, self.num_groups * self.group_size, 1, 1, device=self.device)))
        self.lambda_ = nn.ParameterList(lambda_)

    def activation_save(self, u):
        return F.relu(u - self.lambda_ * self.step)

        os = u.shape
        u = u.view(os[0], self.num_groups, self.group_size, os[2], os[3])
        u = F.relu(1 - self.lambda_ * self.step / u.norm(dim=2, keepdim=True)) * F.relu(u)
        return u.view(os)

    def activation(self, u, k):
        return F.relu(u - self.lambda_[k] * self.step)

        os = u.shape
        u = u.view(os[0], self.num_groups, self.group_size, os[2], os[3])
        u = F.relu(1 - self.lambda_[k] / u.norm(dim=2, keepdim=True)) * F.relu(u)
        return u.view(os)

    def normalize(self):
        for idx in range(self.num_layers):
            self.B[idx].data = F.normalize(self.B[idx].data, p="fro", dim=(-1, -2))

    def forward(self, y):
        batch_size, device = y.shape[0], y.device

        u_new = torch.zeros(batch_size, self.group_size * self.num_groups, self.h_out, self.w_out, device=self.device)
        u_old = torch.zeros(batch_size, self.group_size * self.num_groups, self.h_out, self.w_out, device=self.device)
        u_tmp = torch.zeros(batch_size, self.group_size * self.num_groups, self.h_out, self.w_out, device=self.device)
        t_old = torch.tensor(1.0, device=device)
        for k in range(self.num_layers):
            gen_b = self.B[k]
            for idx in range(1, self.num_groups):
                gen_b = torch.cat((gen_b, tf.rotate(self.B[k], self.ga * idx)), dim=0)
            Bu = F.conv_transpose2d(u_tmp, gen_b, stride=self.stride, padding=self.pad, output_padding=self.output_pad)
            res = y - Bu
            u_new = self.activation(u_tmp + self.step * F.conv2d(res, gen_b, stride=self.stride, padding=self.pad), k)
            t_new = (1 + torch.sqrt(1 + 4 * torch.pow(t_old, 2))) / 2
            u_tmp = u_new + ((t_old - 1) / t_new) * (u_new - u_old)
            if k < self.num_layers - 1:
                u_tmp = self.bn[k](u_tmp)
            u_old, t_old = u_new, t_new
        u_hat = u_old

        y_hat = self.pooler(u_hat)
        y_hat = self.fc(y_hat.view(batch_size, -1))
        return y_hat, u_hat
