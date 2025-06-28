import json
import random
import torch
from torch.utils.data import DataLoader
import time
import os
import os
import json
import torch
import numpy as np
import scipy.sparse
from tqdm import tqdm
from matplotlib import pyplot as plt
from copy import deepcopy
from numpy.lib.stride_tricks import as_strided
# from model.model import ConditionalSeq2SeqVAE, ConditionalSeqDecoder, ConditionalSeqEncoder, reconstruction_loss
# from model.tgnn import TGNN
# from model.vmf_batch import vMF
# from utils.data_utils import fix_seed, node_calculation, edge_calculation
# from utils.utils import load_neurons, Tree, Node
# from utils.log_utils import parser_generate_args
# from scripts.measure_branch import angle_metric, branches_metric
# import copy
import argparse
# from utils.data_utils import fix_seed, load_weight, my_collate, tree_construction, ConditionalPrefixSeqDataset
from utils.utils import load_neurons
# from utils.log_utils import create_log_dir
# from scripts.training import train_conditional_one_epoch, evaluateCVAE
import numpy as np
import  torch.nn as nn
def _gaussian_kernel(size, sigma):
    """生成归一化高斯核"""
    x = np.arange(size) - size // 2
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    return kernel / kernel.sum()
def normalize(data):
    pc = data
    max_dist = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))#计算到中心点最大距离
    pc_norm = pc / max_dist
    return pc_norm,max_dist

def denormlize(data,max):
    for c in range(len(data)):
        data = data*max
    return


def branch_gaussian_smooth(branches, sigma=1.5, kernel_size=None):
    """
        向量化高斯平滑实现
        参数：
            branches: 原始分支数据 (B, N, 3) B=64分支, N=32点
            sigma: 高斯核标准差
            kernel_size: 核大小（奇数）
        返回：
            平滑后的分支数据 (B, N, 3)
        """
    # 生成高斯核
    kernel_size = kernel_size or int(6 * sigma) // 2 * 2 + 1
    pad = kernel_size // 2
    kernel = _gaussian_kernel(kernel_size, sigma)

    # 预分配结果数组
    smoothed = np.empty_like(branches)

    # 对每个坐标维度进行向量化处理
    for dim in range(3):
        # 提取当前维度数据 (B, N)
        data = branches[..., dim]

        # 边界反射填充 (B, N+2p)
        padded = np.pad(data, [(0, 0), (pad, pad)], mode='reflect')

        # 创建滑动窗口视图 (B, N, K)
        window_shape = (padded.shape[0], data.shape[1], kernel_size)
        strides = (padded.strides[0], padded.strides[1], padded.strides[1])
        windows = as_strided(padded,
                             shape=window_shape,
                             strides=strides,
                             writeable=False)

        # 执行向量化卷积 (B, N)
        smoothed_dim = np.einsum('...k,k->...', windows, kernel)

        # 存储结果
        smoothed[..., dim] = smoothed_dim

    return smoothed
class ResidualBlock(nn.Module):
    """1D残差块（保持输入输出形状不变）"""

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
        out += residual  # 残差连接
        return self.relu(out)


class ResNet18(nn.Module):
    def __init__(self, input_channels=3, sequence_length=32):
        super().__init__()
        self.hidden_channels = 64  # 隐藏层通道数

        # 初始卷积层（不改变序列长度）
        self.init_conv = nn.Sequential(
            nn.Conv1d(input_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.hidden_channels),
            nn.ReLU()
        )

        # ResNet18主体：4个残差块层（共8个残差块）
        self.res_layers = nn.Sequential(
            *[ResidualBlock(self.hidden_channels) for _ in range(8)]  # 8个残差块
        )

        # 最终输出层（将通道数映射回输入通道数）
        self.final_conv = nn.Conv1d(self.hidden_channels, input_channels, kernel_size=1)

    def forward(self, x):
        # 输入维度调整：(bs, 32, 3) → (bs, 3, 32)
        x = x.transpose(1, 2)  # [bs, 3, 32]

        # 前向传播
        x = self.init_conv(x)  # [bs, 64, 32]
        x = self.res_layers(x)  # [bs, 64, 32]
        x = self.final_conv(x)  # [bs, 3, 32]

        # 输出维度还原：(bs, 3, 32) → (bs, 32, 3)
        return x.transpose(1, 2)  # [bs, 32, 3]
class cnn(nn.Module):
    def __init__(self, input_size=3, hidden_size=64):
        super(cnn, self).__init__()
        # 定义两层卷积
        self.conv_layers = nn.Sequential(
            # 第一层卷积：输入通道=input_size, 输出通道=hidden_size
            nn.Conv1d(
                in_channels=input_size,
                out_channels=hidden_size,
                kernel_size=3,  # 卷积核大小
                padding=1  # 保持序列长度不变
            ),
            nn.ReLU(),  # 激活函数
            # 第二层卷积：通道数保持hidden_size
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU()
        )
        # 全连接层：将通道数映射回input_size
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # 输入x形状: [batch_size, seq_len, input_size]
        # 调整维度为Conv1d需要的 [batch_size, input_size, seq_len]
        x = x.permute(0, 2, 1)

        # 通过卷积层 -> 输出形状 [batch_size, hidden_size, seq_len]
        conv_out = self.conv_layers(x)

        # 调整回维度 [batch_size, seq_len, hidden_size]
        conv_out = conv_out.permute(0, 2, 1)

        # 全连接层输出 -> [batch_size, seq_len, input_size]
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
    # 恢复到原始尺度
    pc = pc_norm * max_dist
    # 恢复到原始位置
    pc = pc + centroid
    return pc
def visual(points,a):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 提取 x, y, z 坐标
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # 绘制 3D 散点图表示各个点
    ax.scatter(x, y, z, c='r', marker='o')

    # 依次连接 32 个点
    ax.plot(x, y, z, c='b')
    ax.set_title(a)


    # 隐藏坐标轴

    # 显示图形
    plt.show()


def generate_a_tree(neuron,model, args, radius=0.2, type_=1):
    # branches, offsets, dataset, layer, nodes = neuron.fetch_branch_seq(align=args.align, move=True, need_angle=args.sort_angle, need_length=args.sort_length)
    ori_branches, offset,node_branch,branch_branch,max_dist =  neuron.easy_fetch_resample(align=args.align, move=True)
    # print('args.align:',args.align)

    branches = []
    # branch 为（32,3）ndarray

    branches = np.stack(ori_branches) # (N, 32, 3)
    branches = torch.from_numpy((branches)).float()

    max_dist, offset = np.stack(max_dist), np.stack(offset) # (521,) (521,3)
    max_dist = max_dist.reshape(offset.shape[0],1,1)
    offset = offset.reshape(max_dist.shape[0],1,offset.shape[-1])

    # for a in range(len(ori_branches)):
    #     if ori_branches[a].shape !=(32,3):
    #         c = ori_branches[a]
    #         b = 1
    #
    #     noise_branch = ori_branches[a].reshape(1,32,3)
    #     noise_branch = ori_branches[a]
    #
    #     visual(noise_branch,a)
    #
    #     branch = torch.from_numpy(noise_branch).float() # shape (1,32,3)

    # print(branches.shape)
    new_branch = model(branches).detach().numpy()

    # from utils.utils import resample_branch_by_step
    # branc = []
    # for bran in new_branch:
    #
    #     bran = resample_branch_by_step(bran, 10, len(bran))
    #     branc.append(bran)
    # new_branch = np.array(branc)

    new_branch = np.squeeze(new_branch)
    new_branch = branch_gaussian_smooth(new_branch, sigma=1.5, kernel_size=None)
    # print(new_branch.shape)
    # visual(new_branch,a)

    # new_branch = new_branch * max_dist[a]
    # new_branch = new_branch + offset[a]
    new_branch = new_branch * max_dist
    branches = new_branch + offset
    #
    # branches.append(new_branch)

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
            elif cnt_32 == 0:#branch起始

                continue

            elif cnt_32 == 1:
                typ = 0

                if branch_branch[branch_cnt] != -1:
                    cu_branch = branch_cnt
                    fa_branch = branch_branch[cu_branch]
                    father = branch_lastnode[fa_branch]
                elif branch_branch[branch_cnt] == -1:
                    father =0


            else:#中间
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
    # print('check')
    #
    # for node in nodes:
    #     if node[6] == -1:
    #         print('aaaaaaaa')
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
    parser.add_argument('--data_dir', default=r'C:\Users\hyzhou\Desktop\final_it\dit_skenew_deepseek_cut0.15')
    parser.add_argument('--align', default=32)
    parser.add_argument('--sort_length', default=False)
    parser.add_argument('--sort_angle', default=False)
    parser.add_argument('--scaling', default=1)
    parser.add_argument('--denoise_dir', default=r'C:\Users\hyzhou\Desktop\final_it\dit_skenew_deepseek_cut0.15_resnet16_smooth')
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

    # 加载模型时处理设备映射和安全警告
    state_dict = torch.load(
        './denoise/model_denoise/resnet16model.pth',
        map_location='cpu',  # 强制先加载到 CPU
        weights_only=True  # 启用安全模式（需 PyTorch >=1.13）
    )
    model.load_state_dict(state_dict)

    # 将模型移动到目标设备
    model = model.to(device)
    model.eval()
    nodes = generate_a_tree(neuron,model, args, radius=0.2, type_=1)
    return nodes
if __name__ == '__main__':
    ############### log init ###############
    args = parse_denoise_args()

    ############### device set ###############
    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')

    ############### dadtaset preparation ###############

    neurons, neuron_files = load_neurons(args.data_dir, scaling=args.scaling, return_filelist=True)
    print('[INFO] neuron loaded')
    all_idx = list(range(len(neurons)))

    print(len(all_idx))
    # dataset = createDataset(neurons=neurons,neuron_files=neuron_files,args=args)
    # train_loader = DataLoader(dataset, args.bs, shuffle=True,collate_fn=my_collate)
    model = ResNet18()
    model.load_state_dict(torch.load('./model_denoise/resnet16model.pth'))
    model.eval()

    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
    print("当前时间是:", formatted_time)

    for i in range(len(neurons)):

        neuron = neurons[i]
        name = neuron_files[i]
        print(i)
        # branches,offset = split(neuron,args)
        # if i == 14:
        #     m =1
        nodes = generate_a_tree(neuron,model, args, radius=0.2, type_=1)
        file_path = os.path.join(args.denoise_dir, str(name))
        with open(file_path, 'w') as f:
            # 写入注释行（可选）

            # 遍历节点列表
            for node in nodes:
                # 将节点信息转换为字符串，用空格分隔
                node_str = " ".join(map(str, node))
                # 写入节点信息到文件，并添加换行符
                f.write(node_str + "\n")
        # df = denoise_neuron.to_swc(scaling=1)
        # df.to_csv(os.path.join(args.denoise_dir, name), header=None, index=None, sep=' ')

    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
    print("当前时间是:", formatted_time)
    # 使用 savez 函数保存列表中的所有 ndarray
    # np.savez('pt_1005_true.npz', *branches)





