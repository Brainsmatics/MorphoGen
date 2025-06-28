import random
import torch
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.data_utils import fix_seed, my_collate
from utils.log_utils import parse_train_args, create_log_dir
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided
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
            *[ResidualBlock(self.hidden_channels) for _ in range(8)]
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

def distence(x, y):

    squared_diff = (x - y) ** 2
    distances = torch.sqrt(torch.sum(squared_diff, dim=-1))  #  (batch_size, 16)
    batch_avg_distances = torch.mean(distances, dim=-1)  #  (batch_size,)
    final_loss = torch.mean(batch_avg_distances)
    return final_loss

def normalize(batch_data):

    batch_size = batch_data.shape[0]
    normalized_data = np.zeros_like(batch_data)
    for i in range(batch_size):
        pc = batch_data[i]
        centroid = np.mean(pc, axis=0)#
        pc = pc - centroid
        max_dist = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))#
        pc_norm = pc / max_dist
        return pc_norm
def split_list(lst, ratio):

    temp_list = lst.copy()
    first_list_size = int(len(lst) * ratio)
    first_list = []
    while len(first_list) < first_list_size:
        index = random.randint(0, len(temp_list) - 1)
        element = temp_list.pop(index)
        first_list.append(element)
    second_list = temp_list
    return first_list, second_list


class lstm(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super(lstm, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        output = self.fc(lstm_out)
        return output
def visualsave(points,a,savepath):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    ax.scatter(x, y, z, s=30, c='#e84a5f',
               edgecolors='black',
               linewidths=1, alpha=1)

    ax.plot(x, y, z, c = 'darkblue', lw=2, alpha=1)
    # ax.set_title(a)

    plt.savefig(savepath,
                dpi=1000,
                bbox_inches='tight',
                transparent=True,
                facecolor='none',

                )

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
import  torch.nn as nn
def _gaussian_kernel(size, sigma):
    x = np.arange(size) - size // 2
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    return kernel / kernel.sum()
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
if __name__ == '__main__':
    ############### log init ###############
    args = parse_train_args()
    fix_seed(args.seed)
    log_dir = create_log_dir(args)
    timestamp = time.time()
    args.device = 0
    args.bs = 1
    args.ratio = 0.7

    ############### device set ###############
    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')
    print("----------------")
    print(device)
    print("----------------")
    ############### dadtaset preparation ###############
    model = ResNet18()
    model.load_state_dict(torch.load(r'.\denosie\model_denoise\resnet16model.pth'))
    model.eval()

    noise_set , smooth_set = [],[]

    noise_set = np.load(r'.\denosie\branches\pci_15000_noise16.npy')

    noise_train_set,smooth_train_set,noise_test_set,smooth_test_set=[],[],[],[]
    all_idx = list(range(len(noise_set)))


    noise_test_arr = np.array(noise_set)

    test_loader = DataLoader(noise_test_arr, args.bs, shuffle=False)
    a = 0
    out = r' '
    for data in tqdm(test_loader):
        noise_branch= data.numpy()
        outpath = out + '\\' + str(a)
        branch = torch.from_numpy(noise_branch).float()
        v1 = branch.squeeze(axis=0)

        out_put = model(branch).detach().numpy()

        out_put = branch_gaussian_smooth(out_put)
        npy = np.squeeze(out_put)
        visualsave(npy,a,outpath)
        v2 = out_put.squeeze(axis=0)
        a = a + 1
