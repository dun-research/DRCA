import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from math import sqrt
from models.compression_modules import get_compression_module
from models.diff_rank import PerturbedRank, UniformSample
from models.weight_init import weight_init


class DiffContextAwareCompressionModule(nn.Module):
    def __init__(self, in_channels, T, strategy="diff_rank", comp_module="krc", k=4, sigma=0.05, n_sample=500) -> None:
        super().__init__()
        self.strategy = strategy
        self.comp_module = comp_module
        self.k = k
        self.sigma = sigma
        self.n_sample = n_sample
        self.in_channels = in_channels

        self.compression = get_compression_module(comp_module, in_channels)
        
        if strategy == "diff_rank":
            self.rank_net = PerturbedRank(in_channels, k, n_sample, sigma)
        elif strategy == "wo_diff_rank":
            self.rank_net = PerturbedRank(in_channels, k, n_sample, sigma, do_train=False)
        elif strategy == "uniform":
            self.rank_net = UniformSample(k, T)
        else:
            raise ValueError("Unknown strategy: {}".format(strategy))
        self.apply(weight_init)

    def forward(self, x, x_cls, return_index=False):
        B, C, T, H, W = x.shape
        origin_H, origin_W = H, W

        if self.strategy.find("info_rank") >=0:
            x = (x, x_cls)
        frames_topk, frames_back, sorted_inds = self.rank_net(x)
        frames_back_lowres = self.compression(frames_back, frames_topk)
        
        frames_topk = rearrange(frames_topk, "b c k h w -> b h w k c", )
        frames_back_lowres = rearrange(frames_back_lowres, "b c nk h w -> b h w nk c", )
        
        if return_index:
            return frames_topk, frames_back_lowres, sorted_inds
        else:
            return frames_topk, frames_back

