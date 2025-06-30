
import time
import os
import torch
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import as_strided
import argparse
from utils.utils import load_neuron
import numpy as np
import  torch.nn as nn

def _gaussian_kernel(size, sigma):
    x = np.arange(size) - size // 2
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    return kernel / kernel.sum()
def normalize(data):
    pc = data
    max_dist = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc_norm = pc / max_dist
    return pc_norm,max_dist

def denormlize(data,max):
    for c in range(len(data)):
        data = data*max
    return


def branch_gaussian_smooth(branches, sigma=1.5, kernel_size=None):
    kernel_size = kernel_size or int(6 * sigma) // 2 * 2 + 1
    pad = kernel_size // 2
    kernel = _gaussian_kernel(kernel_size, sigma)
    smoothed = np.empty_like(branches)
    for dim in range(3):
        data = branches[..., dim]
        padded = np.pad(data, [(0, 0), (pad, pad)], mode='reflect')
        window_shape = (padded.shape[0], data.shape[1], kernel_size)
        strides = (padded.strides[0], padded.strides[1], padded.strides[1])
        windows = as_strided(padded,
                             shape=window_shape,
                             strides=strides,
                             writeable=False)

        smoothed_dim = np.einsum('...k,k->...', windows, kernel)

        smoothed[..., dim] = smoothed_dim

    return smoothed
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        out += residual
        return self.relu(out)


class ResNet18(nn.Module):
    def __init__(self, input_channels=3, sequence_length=32):
        super().__init__()
        self.hidden_channels = 64
        self.init_conv = nn.Sequential(
            nn.Conv1d(input_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.hidden_channels),
            nn.ReLU()
        )

        self.res_layers = nn.Sequential(
            *[ResidualBlock(self.hidden_channels) for _ in range(8)]  # 8个残差块
        )
        self.final_conv = nn.Conv1d(self.hidden_channels, input_channels, kernel_size=1)

    def forward(self, x):
        # (bs, 32, 3) → (bs, 3, 32)
        x = x.transpose(1, 2)  # [bs, 3, 32]
        x = self.init_conv(x)  # [bs, 64, 32]
        x = self.res_layers(x)  # [bs, 64, 32]
        x = self.final_conv(x)  # [bs, 3, 32]

        # (bs, 3, 32) → (bs, 32, 3)
        return x.transpose(1, 2)  # [bs, 32, 3]
class cnn(nn.Module):
    def __init__(self, input_size=3, hidden_size=64):
        super(cnn, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=hidden_size,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU()
        )
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        conv_out = self.conv_layers(x)
        conv_out = conv_out.permute(0, 2, 1)
        output = self.fc(conv_out)
        return output

class lstm(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super(lstm, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        output = self.fc(lstm_out)
        return output
def denormalize(pc_norm, centroid, max_dist):
    pc = pc_norm * max_dist
    pc = pc + centroid
    return pc
def visual(points,a):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    ax.scatter(x, y, z, c='r', marker='o')
    ax.plot(x, y, z, c='b')
    ax.set_title(a)
    plt.show()


def generate_a_tree(neuron,model, args, radius=0.2, type_=1):
    # branches, offsets, dataset, layer, nodes = neuron.fetch_branch_seq(align=args.align, move=True, need_angle=args.sort_angle, need_length=args.sort_length)
    ori_branches, offset,node_branch,branch_branch,max_dist =  neuron.easy_fetch_resample(align=args.align, move=True)
    # print('args.align:',args.align)

    branches = []
    # branch （32,3）ndarray

    branches = np.stack(ori_branches) # (N, 32, 3)
    branches = torch.from_numpy((branches)).float()

    max_dist, offset = np.stack(max_dist), np.stack(offset) # (521,) (521,3)
    max_dist = max_dist.reshape(offset.shape[0],1,1)
    offset = offset.reshape(max_dist.shape[0],1,offset.shape[-1])

    new_branch = model(branches).detach().numpy()


    new_branch = np.squeeze(new_branch)
    new_branch = branch_gaussian_smooth(new_branch, sigma=1.5, kernel_size=None)

    new_branch = new_branch * max_dist
    branches = new_branch + offset


    nodes = []
    node_cnt = 0
    branch_lastnode = {}
    smallnode_branch = {}
    for branch_cnt in range(len(branches)):

        branch = branches[branch_cnt]

        for cnt_32 in range(len(branch)):
            if node_cnt == 0 :
                typ = 1
                father = -2
            elif cnt_32 == 0:#branch

                continue

            elif cnt_32 == 1:
                typ = 0

                if branch_branch[branch_cnt] != -1:
                    cu_branch = branch_cnt
                    fa_branch = branch_branch[cu_branch]
                    father = branch_lastnode[fa_branch]
                elif branch_branch[branch_cnt] == -1:
                    father =0


            else:
                typ = 0
                father = node_cnt - 1
            x = branch[cnt_32][0]
            y = branch[cnt_32][1]
            z = branch[cnt_32][2]
            node = (node_cnt+1,typ,x,y,z,1,father+1)
            nodes.append(node)
            smallnode_branch.update({node_cnt:branch_cnt})


            if cnt_32 == len(branch)-1:
                branch_lastnode.update({branch_cnt:node_cnt})
            node_cnt = node_cnt + 1

    return nodes


def parse_denoise_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',default=-1)
    parser.add_argument('--need_gauss',default=False)
    parser.add_argument('--in_one_graph', default=False, type=bool)
    parser.add_argument('--only_swc',action='store_true')
    parser.add_argument('--projection_3d', default='xyz',type=str)
    parser.add_argument('--short', default=None, type=int)
    parser.add_argument('--teaching', default=0, type=float,
                        help='teaching force')
    parser.add_argument('--generate_layers', default=-1, type=int,
                        help='the layers to draw, recommended to be no more than 8. -1 for draw'
                        'whole neuron.')

    parser.add_argument('--max_window_length', default=8, type=int,
                        help='the max number of branches on prefix')
    parser.add_argument('--max_src_length', default=32, type=int,
                        help='the max length for generated branches')
    parser.add_argument('--max_dst_length', default=32, type=int,
                        help='--max length for generating branches')

    parser.add_argument('--model_path',default=' ',type=str)
    parser.add_argument('--output_dir',default=' ',type=str)

    parser.add_argument('--log_dir', default=r'.\log')
    parser.add_argument('--data_dir', default=r'')
    parser.add_argument('--align', default=32)
    parser.add_argument('--sort_length', default=False)
    parser.add_argument('--sort_angle', default=False)
    parser.add_argument('--scaling', default=1)
    parser.add_argument('--denoise_dir', default=r'')
    parser.add_argument('--wind_len', default=4)

    args = parser.parse_args()
    return args
def neuron_denoise(neuron):
    ############### log init ###############
    args = parse_denoise_args()

    ############### device set ###############
    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')



    model = ResNet18()
    model.load_state_dict(torch.load('./model_denoise/resnet16model_block.pth'))
    model.eval()
    nodes = generate_a_tree(neuron,model, args, radius=0.2, type_=1)
    return nodes

def auxi(df,model):

    args = parse_denoise_args()
    neuron = load_neuron(df, scaling=1)

    nodes = generate_a_tree(neuron,model, args, radius=0.2, type_=1)
    return nodes



