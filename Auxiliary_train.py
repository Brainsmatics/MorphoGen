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
        max_dist = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
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

def plt_data(losses,path):
    plt.rcParams["font.family"] = "Times New Roman"
    x = np.arange(len(losses))
    plt.figure(figsize=(8, 5))
    plt.plot(x, losses, '#086972', linewidth=2, label='Training loss')

    # 坐标轴标签
    plt.xlabel("Epochs", fontsize=18)
    plt.ylabel("Training loss", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

class lstm(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super(lstm, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        output = self.fc(lstm_out)
        return output


import torch
import torch.nn as nn


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
            *[ResidualBlock(self.hidden_channels) for _ in range(8)]  # 8
        )
        self.final_conv = nn.Conv1d(self.hidden_channels, input_channels, kernel_size=1)

    def forward(self, x):
        #(bs, 32, 3) → (bs, 3, 32)
        x = x.transpose(1, 2)  # [bs, 3, 32]


        x = self.init_conv(x)  # [bs, 64, 32]
        x = self.res_layers(x)  # [bs, 64, 32]
        x = self.final_conv(x)  # [bs, 3, 32]

        # (bs, 3, 32) → (bs, 32, 3)
        return x.transpose(1, 2)  # [bs, 32, 3]
if __name__ == '__main__':
    ############### log init ###############
    args = parse_train_args()
    fix_seed(args.seed)
    log_dir = create_log_dir(args)
    timestamp = time.time()
    args.device = 0
    args.bs = 150000
    args.ratio = 0.9

    ############### device set ###############
    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')
    print("----------------")
    print(device)
    print("----------------")
    ############### dadtaset preparation ###############

    noise_set , smooth_set = [],[]
    noise_set = np.load('.\denosie\branches\pci_15000_noise16.npy')
    smooth_set = np.load('.\denosie\branches\pci_15000_true16.npy')
    assert len(noise_set) == len(smooth_set)
    noise_train_set,smooth_train_set,noise_test_set,smooth_test_set=[],[],[],[]
    all_idx = list(range(len(noise_set)))
    train_idx,test_idx = split_list(all_idx,args.ratio)
    for id in train_idx:
        noise_train_set.append(noise_set[id])
        smooth_train_set.append(smooth_set[id])
    for id in test_idx:
        noise_test_set.append(noise_set[id])
        smooth_test_set.append(smooth_set[id])

    noise_train_arr = np.array(noise_train_set)
    smooth_train_arr = np.array(smooth_train_set)
    noise_test_arr = np.array(noise_test_set)
    smooth_test_arr = np.array(smooth_test_set)

    train_arr = np.stack((noise_train_arr, smooth_train_arr), axis=1)
    test_arr = np.stack((noise_test_arr, smooth_test_arr), axis=1)
    train_loader = DataLoader(train_arr, args.bs, shuffle=True,collate_fn=my_collate)#train_arr = (2,16,3)
    test_loader = DataLoader(test_arr, args.bs, shuffle=False,collate_fn=my_collate)


    model = ResNet18().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.8)
    best_loss = 10000
    best_epoch = 0
    epoches  = 300
    train_losses,test_losses = [],[]

    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
    print(formatted_time)

    model.train()
    for epoch in range(epoches):
        total_loss = 0
        for data in tqdm(train_loader):
            noise_branch, smooth_branch = data
            noise_branch = torch.from_numpy(noise_branch).float().cuda()
            smooth_branch = torch.from_numpy(smooth_branch).float().cuda()
            out_put = model(noise_branch)
            loss = distence(out_put, smooth_branch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch + 1

            torch.save(model.state_dict(), './model_denoise/resnet16model.pth')
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for data in tqdm(test_loader):
                noise_branch, smooth_branch = data
                noise_branch = torch.from_numpy(noise_branch).float().cuda()
                smooth_branch = torch.from_numpy(smooth_branch).float().cuda()
                output = model(noise_branch)
                test_loss = distence(output, smooth_branch)
                total_test_loss += test_loss.item()

        test_loss = total_test_loss / len(test_loader)
        test_losses .append(test_loss)

        scheduler.step()

        print(f'Current lr: {optimizer.param_groups[0]["lr"]:.2e}')
        print(f"\nEpoch {epoch + 1}/{epoches} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}")
        loss1 = np.array(train_losses)
        loss2 = np.array(test_loss)
        np.save(r'.\denosie\loss_npy\train16.npy', loss1)
        np.save(r'.\denosie\loss_npy\test16.npy', loss2)
        plt_data(train_losses,'')
        plt_data(test_losses,'')

current_time = time.localtime()
formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
print(formatted_time)
print(f'Best model found at epoch {best_epoch} with loss {best_loss:.4f}')


