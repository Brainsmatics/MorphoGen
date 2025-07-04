import os
import torch
import pandas as pd
from pprint import pprint
import torch.nn as nn
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist

import argparse
from torch.distributions import Normal

from utils.file_utils import *
from utils.visualize import *
from tqdm import tqdm

from datasets.shapenet_data_pc import ShapeNet15kPointClouds
from models.dit3d import DiT3D_models
from utils.misc import Evaluator
import numpy as np
from scipy.spatial import distance_matrix, KDTree
from utils.ske_connect import *
import heapq
from sklearn.decomposition import PCA
from utils.cut import filter_short_branches
from utils.utils import load_neuron
from utils.swc_denoise import auxi
'''
models
'''
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + (mean1 - mean2)**2 * torch.exp(-logvar2))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # Assumes data is integers [0, 1]
    assert x.shape == means.shape == log_scales.shape
    px0 = Normal(torch.zeros_like(means), torch.ones_like(log_scales))

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 0.5)
    cdf_plus = px0.cdf(plus_in)
    min_in = inv_stdv * (centered_x - .5)
    cdf_min = px0.cdf(min_in)
    log_cdf_plus = torch.log(torch.max(cdf_plus, torch.ones_like(cdf_plus)*1e-12))
    log_one_minus_cdf_min = torch.log(torch.max(1. - cdf_min,  torch.ones_like(cdf_min)*1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
    x < 0.001, log_cdf_plus,
    torch.where(x > 0.999, log_one_minus_cdf_min,
             torch.log(torch.max(cdf_delta, torch.ones_like(cdf_delta)*1e-12))))
    assert log_probs.shape == x.shape
    return log_probs


class GaussianDiffusion:
    def __init__(self,betas, loss_type, model_mean_type, model_var_type):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))



    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        return (
                self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )


    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(self, denoise_fn, data, t, y, clip_denoised: bool, return_pred_xstart: bool):
        
        model_output = denoise_fn(data, t, y)

        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas.to(data.device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device)),
                'fixedsmall': (self.posterior_variance.to(data.device), self.posterior_log_variance_clipped.to(data.device)),
            }[self.model_var_type]
            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(data)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(data)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(data, t=t, eps=model_output)

            if clip_denoised:
                x_recon = torch.clamp(x_recon, -.5, .5)

            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data, t=t)
        else:
            raise NotImplementedError(self.loss_type)


        assert model_mean.shape == x_recon.shape == data.shape
        assert model_variance.shape == model_log_variance.shape == data.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        eps = eps.to(x_t.device)
        assert x_t.shape == eps.shape
        return (
                self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        )

    ''' samples '''

    def p_sample(self, denoise_fn, data, t, noise_fn, y, clip_denoised=False, return_pred_xstart=False):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(denoise_fn, data=data, t=t, y=y, clip_denoised=clip_denoised,
                                                                 return_pred_xstart=True)
        noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        assert noise.shape == data.shape
        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        assert sample.shape == pred_xstart.shape
        return (sample, pred_xstart) if return_pred_xstart else sample


    def p_sample_loop(self, denoise_fn, shape, device, y,
                      noise_fn=torch.randn, clip_denoised=True, keep_running=False):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """

        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        for t in reversed(range(0, self.num_timesteps if not keep_running else len(self.betas))):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t,t=t_, noise_fn=noise_fn, y=y,
                                  clip_denoised=clip_denoised, return_pred_xstart=False)

        assert img_t.shape == shape
        return img_t

    def reconstruct(self, x0, t, y, denoise_fn, noise_fn=torch.randn, constrain_fn=lambda x, t:x):

        assert t >= 1

        t_vec = torch.empty(x0.shape[0], dtype=torch.int64, device=x0.device).fill_(t-1)
        encoding = self.q_sample(x0, t_vec)

        img_t = encoding

        for k in reversed(range(0,t)):
            img_t = constrain_fn(img_t, k)
            t_ = torch.empty(x0.shape[0], dtype=torch.int64, device=x0.device).fill_(k)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn, y=y,
                                  clip_denoised=False, return_pred_xstart=False, use_var=True).detach()


        return img_t


class Model(nn.Module):
    def __init__(self, args, betas, loss_type: str, model_mean_type: str, model_var_type:str):
        super(Model, self).__init__()
        self.diffusion = GaussianDiffusion(betas, loss_type, model_mean_type, model_var_type)
        
        # DiT-3d
        self.model = DiT3D_models[args.model_type](input_size=args.voxel_size, num_classes=args.num_classes)

    def prior_kl(self, x0):
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, x0, y, clip_denoised=True):
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt =  self.diffusion.calc_bpd_loop(self._denoise, x0, y, clip_denoised)

        return {
            'total_bpd_b': total_bpd_b,
            'terms_bpd': vals_bt,
            'prior_bpd_b': prior_bpd_b,
            'mse_bt':mse_bt
        }


    def _denoise(self, data, t, y):
        B, D,N= data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        out = self.model(data, t, y)

        assert out.shape == torch.Size([B, D, N])
        return out

    def get_loss_iter(self, data, noises=None, y=None):
        B, D, N = data.shape                           # [16, 3, 2048]
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=data.device)

        if noises is not None:
            noises[t!=0] = torch.randn((t!=0).sum(), *noises.shape[1:]).to(noises)

        losses = self.diffusion.p_losses(
            denoise_fn=self._denoise, data_start=data, t=t, noise=noises, y=y)
        assert losses.shape == t.shape == torch.Size([B])
        return losses

    def gen_samples(self, shape, device, y, noise_fn=torch.randn,
                    clip_denoised=True,
                    keep_running=False):
        return self.diffusion.p_sample_loop(self._denoise, shape=shape, device=device, y=y, noise_fn=noise_fn,
                                            clip_denoised=clip_denoised,
                                            keep_running=keep_running)

    def gen_sample_traj(self, shape, device, y, freq, noise_fn=torch.randn,
                    clip_denoised=True,keep_running=False):
        return self.diffusion.p_sample_loop_trajectory(self._denoise, shape=shape, device=device, y=y, noise_fn=noise_fn, freq=freq,
                                                       clip_denoised=clip_denoised,
                                                       keep_running=keep_running)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)


def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == 'warm0.1':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.2':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.5':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    else:
        raise NotImplementedError(schedule_type)
    return betas

def get_constrain_function(ground_truth, mask, eps, num_steps=1):
    '''

    :param target_shape_constraint: target voxels
    :return: constrained x
    '''
    # eps_all = list(reversed(np.linspace(0,np.float_power(eps, 1/2), 500)**2))
    eps_all = list(reversed(np.linspace(0, np.sqrt(eps), 1000)**2 ))
    def constrain_fn(x, t):
        eps_ =  eps_all[t] if (t<1000) else 0
        for _ in range(num_steps):
            x  = x - eps_ * ((x - ground_truth) * mask)


        return x
    return constrain_fn


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
    else:
        output = tensor
    return output
#############################################################################

def get_dataset(dataroot, npoints,category,use_mask=False):
    tr_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=[category], split='train',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=True, use_mask = use_mask)
    te_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=[category], split='val',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
        use_mask=use_mask
    )
    return tr_dataset, te_dataset


def get_dataloader(opt, train_dataset, test_dataset=None):

    if opt.distribution_type == 'multi':
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=opt.world_size,
            rank=opt.rank
        )
        if test_dataset is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset,
                num_replicas=opt.world_size,
                rank=opt.rank
            )
        else:
            test_sampler = None
    else:
        train_sampler = None
        test_sampler = None

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs,sampler=train_sampler,
                                                   shuffle=train_sampler is None, num_workers=int(opt.workers), drop_last=True)

    if test_dataset is not None:
        test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs,sampler=test_sampler,
                                                   shuffle=False, num_workers=int(opt.workers), drop_last=False)
    else:
        test_dataloader = None

    return train_dataloader, test_dataloader, train_sampler, test_sampler

def neuron_swc_generator(skeleton, soma_radius=5.0):

    def detect_soma(points, radius):
        tree = KDTree(points)
        densities = np.array([len(tree.query_ball_point(p, radius)) for p in points])
        return points[np.argmax(densities)]

    soma_center = detect_soma(skeleton, soma_radius)


    class NeuronTree:
        def __init__(self, root):
            self.nodes = {0: {'id': 0, 'pos': root, 'children': [], 'parent': None}}
            self.current_id = 1
            self.direction = np.zeros(3)  

        def add_node(self, parent_id, position):
            self.nodes[self.current_id] = {
                'id': self.current_id,
                'pos': position,
                'children': [],
                'parent': parent_id
            }
            self.nodes[parent_id]['children'].append(self.current_id)
            self.current_id += 1

        def update_direction(self):
       
            if len(self.nodes) > 1:
                positions = np.array([n['pos'] for n in self.nodes.values()])
                pca = PCA(n_components=1)
                pca.fit(positions)
                self.direction = pca.components_[0]
            else:
                self.direction = np.random.randn(3) 
                self.direction /= np.linalg.norm(self.direction) + 1e-6

  
    tree = NeuronTree(soma_center)
    candidate_edges = []
    dists = distance_matrix([soma_center], skeleton)[0]
    for i, d in enumerate(dists):
        if d > 0:
            direction = skeleton[i] - soma_center
            cos_sim = np.dot(direction, tree.direction) / (np.linalg.norm(direction) + 1e-6)
            heapq.heappush(candidate_edges, (d * (1.0 - cos_sim), 0, i))  


    visited = set([0])
    skeleton_flags = np.zeros(len(skeleton), dtype=bool)
    skeleton_flags[np.argmin(dists)] = True  

    while candidate_edges:
        weight, parent_id, skel_idx = heapq.heappop(candidate_edges)

        if skeleton_flags[skel_idx]:
            continue


        if len(tree.nodes[parent_id]['children']) >= 2:
            continue


        tree.add_node(parent_id, skeleton[skel_idx])
        current_id = tree.current_id - 1
        visited.add(current_id)
        skeleton_flags[skel_idx] = True


        if len(visited) % 10 == 0:
            tree.update_direction()

        new_pos = skeleton[skel_idx]
        dists = distance_matrix([new_pos], skeleton)[0]
        for i, d in enumerate(dists):
            if not skeleton_flags[i] and d > 0:
                direction = skeleton[i] - new_pos
                if tree.direction is not None:
                    cos_sim = np.dot(direction, tree.direction) / (np.linalg.norm(direction) + 1e-6)
                else:
                    cos_sim = 0
                heapq.heappush(candidate_edges, (d * (1.2 - cos_sim), current_id, i))

    swc_data = []
    for nid, node in tree.nodes.items():
        swc_entry = [
            nid,  # ID
            3 if nid != 0 else 1,  
            node['pos'][0],  # X
            node['pos'][1],  # Y
            node['pos'][2],  # Z
            soma_radius if nid == 0 else 1.0,  
            -1 if nid == 0 else node['parent'] 
        ]
        swc_data.append(swc_entry)

    return np.array(swc_data)
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

def generate_a_tree(neuron,model, args, radius=0.2, type_=1):
    ori_branches, offset,node_branch,branch_branch,max_dist =  neuron.easy_fetch_resample(align=args.align, move=True)
    # print('args.align:',args.align)

    branches = []
    # branch 为（32,3）ndarray

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
            elif cnt_32 == 0:

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

def generate_eval(model, opt, gpu, outf_syn, evaluator):

    _, test_dataset = get_dataset(opt.dataroot, opt.npoints, opt.category)

    _, test_dataloader, _, test_sampler = get_dataloader(opt, test_dataset, test_dataset)

    aux_model = ResNet18()
    aux_model.load_state_dict(torch.load('./temp/resnet16model.pth'))
    aux_model.eval()

    def new_y_chain(device, num_chain, num_classes):
        return torch.randint(low=0,high=num_classes,size=(num_chain,),device=device)
    
    with torch.no_grad():

        samples = []
        a = 0
        for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Generating Samples'):
            x = data['test_points'].transpose(1,2)
            m, s = data['mean'].float(), data['std'].float()
            y = data['cate_idx']
            gen = model.gen_samples(x.shape, gpu, new_y_chain(gpu,y.shape[0],opt.num_classes), clip_denoised=False).detach().cpu()
            gen = gen.transpose(1,2).contiguous()
            x = x.transpose(1,2).contiguous()

            gen = gen * s + m  #torch.Size([2, 2048, 3])
            x = x * s + m
            for pc in gen: #torch.Size([2048, 3])
                points = pc.numpy()
                point_swc_L1 = L1_medial(points=points, NCenters=2048,iters=1)
                ske = point_swc_L1[fps(point_swc_L1, 1200), :] #(1200,3)
                ###
                connect = neuron_swc_generator(ske) #(1200, 7)
                ###
                cut = [
                    {
                        'n': int(row[0]),
                        'type': int(row[1]),
                        'x': row[2],
                        'y': row[3],
                        'z': row[4],
                        'radius': row[5],
                        'parent': int(row[6])
                    }
                    for row in connect
                ]  #[1200,7]
                cut = filter_short_branches(cut, length_threshold=0.1) ##[1200,7]
                cut = pd.DataFrame(cut)
                nodes = auxi(cut,aux_model)
                file_path = os.path.join(opt.generate_dir, f"pc_{a}.swc")
                print(f"pc_{a}.swc")
                a = a + 1                 
                with open(file_path, 'w') as f:
                    for node in nodes:
                        node_str = " ".join(map(str, node))
                        f.write(node_str + "\n")
    return output_dir


def main(opt):
    output_dir = get_output_dir(opt.generate_dir, opt.experiment_name)
    copy_source(__file__, output_dir)

    opt.dist_url = f'tcp://{opt.node}:{opt.port}'
    print('Using url {}'.format(opt.dist_url))

    if opt.distribution_type == 'multi':
        opt.ngpus_per_node = torch.cuda.device_count()
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(test, nprocs=opt.ngpus_per_node, args=(opt, output_dir))
    else:
        test(opt.gpu, opt, output_dir)

def test(gpu, opt, output_dir):

    logger = setup_logging(output_dir)
    if opt.distribution_type == 'multi':
        should_diag = gpu==0
    else:
        should_diag = True

    outf_syn, = setup_output_subdirs(output_dir, 'syn')

    if opt.distribution_type == 'multi':
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])

        base_rank =  opt.rank * opt.ngpus_per_node
        opt.rank = base_rank + gpu
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)

        opt.bs = int(opt.bs / opt.ngpus_per_node)
        opt.workers = 0

    '''
    create networks
    '''

    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    if opt.distribution_type == 'multi':  # Multiple processes, single GPU per process
        def _transform_(m):
            return nn.parallel.DistributedDataParallel(
                m, device_ids=[gpu], output_device=gpu)

        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        model.multi_gpu_wrapper(_transform_)


    elif opt.distribution_type == 'single':
        def _transform_(m):
            return nn.parallel.DataParallel(m)
        model = model.cuda()
        model.multi_gpu_wrapper(_transform_)

    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        raise ValueError('distribution_type = multi | single | None')

    if should_diag:
        logger.info(opt)

        logger.info("Model = %s" % str(model))
        total_params = sum(param.numel() for param in model.parameters())/1e6
        logger.info("Total_params = %s MB " % str(total_params))    # S4: 32.81 MB

    model.eval()

    evaluator = Evaluator(results_dir=output_dir)    

    with torch.no_grad():
        
        if should_diag:
            logger.info("Resume Path:%s" % opt.model)

        resumed_param = torch.load(opt.model)
        model.load_state_dict(resumed_param['model_state'])

        opt.eval_path = os.path.join(outf_syn, 'samples.pth')
        Path(opt.eval_path).parent.mkdir(parents=True, exist_ok=True)
        
        stats = generate_eval(model, opt, gpu, outf_syn, evaluator)

        if should_diag:
            logger.info(stats)
        

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')
    parser.add_argument('--nc', default=3)
    parser.add_argument('--npoints', default=2048)
    parser.add_argument('--beta_start', default=0.0001)
    parser.add_argument('--beta_end', default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', type=int, default=1000)
    parser.add_argument('--attention', default=True)
    parser.add_argument('--dropout', default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--model_mean_type', default='eps')
    parser.add_argument('--model_var_type', default='fixedsmall')

    '''distributed'''
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--node', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:12345')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distribution_type', default='single', choices=['multi', 'single', None],
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all available GPUs.')
    
    '''eval'''
    parser.add_argument('--eval_path',default='')
    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')
    parser.add_argument('--model_dir', type=str, default=r'/mnt/d/hyzhou/point cloud/temp', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='ct', help='experiment name (used for checkpointing and logging)')
    parser.add_argument('--category', default='it')
    parser.add_argument('--bs', type=int, default=2, help='input batch size')
    parser.add_argument("--model_type", type=str, choices=list(DiT3D_models.keys()), default="DiT-S/4")
    parser.add_argument("--voxel_size", type=int, choices=[16, 32, 64], default=32)
    parser.add_argument('--model', default=r'/data/mosthyzhou/dit_a100/temp/epoch_39999.pth', help="path to model (to continue training)")
    parser.add_argument('--generate_dir', default=r'/data/mosthyzhou/dit_a100/temp/swc')
    parser.add_argument('--dataroot', default=r"/data/mosthyzhou/data/it")
    opt = parser.parse_args()


    return opt
if __name__ == '__main__':
    opt = parse_args()
    set_seed(opt)

    main(opt)
