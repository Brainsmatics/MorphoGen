
import torch
import os
import time
from utils.data_utils import fix_seed, load_weight, my_collate, tree_construction, ConditionalPrefixSeqDataset
from utils.utils import load_neurons
from utils.log_utils import parse_train_args, create_log_dir

import numpy as np
def normalize(data):
    pc = data
    max_dist = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc_norm = pc / max_dist
    return pc_norm,max_dist
def createDataset(neurons, neuron_files, args):
    branches, offsets, dataset, Tree = [],[],[],[]
    align = 32
    cnt = 0
    all_branches = []
    for neuron in neurons:
        print('load2')
        print(cnt, neuron_files[cnt])
        cnt += 1
        single_branches, _, _, _ ,_= neuron.easy_fetch(align=align, move=True)
        print(len(single_branches))
        all_branches = all_branches +single_branches
    print(len(all_branches))

    return all_branches
if __name__ == '__main__':
    ############### log init ###############
    args = parse_train_args()
    fix_seed(args.seed)
    log_dir = create_log_dir(args)
    timestamp = time.time()

    args.data_dir = r".\denosie\single_resnet_neuron"
    args.device = 0
    args.bs = 1

    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')

    neurons, neuron_files = load_neurons(args.data_dir, scaling=args.scaling, return_filelist=True)
    print('[INFO] neuron loaded')
    all_idx = list(range(len(neurons)))

    branches= createDataset(neurons=neurons,neuron_files=neuron_files,args=args)

    branches = np.array(branches)
    np.save(r'.\denosie\single_branch\single_resnet_branch.npy', branches)





