import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from einops import rearrange
from math import sqrt
import torch.distributed as dist
from models.modules.weight_init import weight_init

class UniformSample(nn.Module):
    def __init__(self, k: int, T: int) -> None:
        super().__init__()
        self.k = k
        self.T = T

    def forward(self, frames):
        B,C,T,H,W = frames.shape
        frames = rearrange(frames, "b c t h w -> b t (h w c)")
        indices_topk, indices_back = self.generate_uniform_indices(B)
        frames_topk = extract_from_indices(frames, indices_topk)
        frames_back = extract_from_indices(frames, indices_back)
        sorted_inds = torch.cat([indices_topk, indices_back], dim=-1)
        frames_topk = rearrange(frames_topk, "b k (h w c) -> b c k h w", b=B, k=self.k, h=H, w=W)
        frames_back = rearrange(frames_back, "b nk (h w c) -> b c nk h w", b=B, nk=T-self.k, h=H, w=W)
        return frames_topk, frames_back, sorted_inds

    def generate_uniform_indices(self, b):
        indices_top = np.linspace(0, self.T, self.k+1).astype(np.int32)[:-1]
        indices_back = np.array(list(set(range(self.T)) - set(indices_top.tolist()))).astype(np.int32)
        indices_back.sort()

        indices_top = torch.from_numpy(indices_top).long().unsqueeze(0).repeat(b, 1).cuda()
        indices_back = torch.from_numpy(indices_back).long().unsqueeze(0).repeat(b, 1).cuda()
        return indices_top, indices_back
    

class PerturbedRankFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 1000, sigma: float = 0.05):
        b, d = x.shape
        # for Gaussian: noise and gradient are the same.
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)
        # noise = torch.zeros_like(noise)
        perturbed_x = x[:, None, :] + noise * sigma # b, nS, d
        
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices_topk = topk_results.indices # b, nS, d
        #indices_topk, indices_back = torch.split(indices, [k, d-k], dim=2) #indices[:, :, :k], indices[:, :, k:]
        indices_topk = torch.sort(indices_topk, dim=-1).values
        
        back_results = torch.topk(perturbed_x, k=d-k, dim=-1, largest=False, sorted=False)
        indices_back = back_results.indices # b, nS, d
        #indices_topk, indices_back = torch.split(indices, [k, d-k], dim=2) #indices[:, :, :k], indices[:, :, k:]
        indices_back = torch.sort(indices_back, dim=-1).values

        # b, nS, k, d
        perturbed_output_topk = torch.nn.functional.one_hot(indices_topk, num_classes=d).float()
        indicators_topk = perturbed_output_topk.mean(dim=1) # b, k, d

        perturbed_output_back = torch.nn.functional.one_hot(indices_back, num_classes=d).float()
        indicators_back = perturbed_output_back.mean(dim=1) # b, k, d

        # constants for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        ctx.save_for_backward(perturbed_output_topk, perturbed_output_back, noise)
        return (indicators_topk, indicators_back)

    @staticmethod
    def backward(ctx, grad_output_topk, grad_output_back):
        if grad_output_topk is None:
            print("grad_output_topk is None")
            return tuple([None] * 5)

        perturbed_output_topk, perturbed_output_back, noise_gradient = ctx.saved_tensors

        # import pdb; pdb.Pdb(nosigint=True).set_trace()
        if ctx.sigma <= 1e-20:
            b, _, k, d = perturbed_output_topk.size()
            expected_gradient_topk = torch.zeros(b, k, d).to(grad_output_topk.device)

            b, _, k, d = perturbed_output_back.size()
            expected_gradient_back = torch.zeros(b, k, d).to(grad_output_back.device)
        else:
            expected_gradient_topk = (
                torch.einsum("bnkd,bnd->bkd", perturbed_output_topk, noise_gradient)
                / ctx.num_samples
                / (ctx.sigma)
            )

            expected_gradient_back = (
                torch.einsum("bnkd,bnd->bkd", perturbed_output_back, noise_gradient)
                / ctx.num_samples
                / (ctx.sigma)
            )
        # (grad_output_topk * expected_gradient_topk).sum(1)
        grad_input_topk = torch.einsum("bkd,bkd->bd", grad_output_topk, expected_gradient_topk)

        grad_input_back = torch.einsum("bkd,bkd->bd", grad_output_back, expected_gradient_back)
        #print(f"topk: {grad_input_topk.sum().item()}, back: {grad_input_back.sum().item()}")

        # get final grad_input
        grad_input = grad_input_topk + grad_input_back
        #print(f"grad_input: {grad_input.sum()}, {grad_output_topk.shape}, {grad_output_back.shape}")
        return (grad_input/10., None, None, None, None)
    
    
class PerturbedRank(nn.Module):
    def __init__(self, embed_dim:int, k: int, num_samples: int = 1000, sigma: float = 0.05, do_train: bool = True):
        super(PerturbedRank, self).__init__()
        self.num_samples = num_samples
        self.sigma = sigma
        self.k = k
        self.score_network = ScoreNet(embed_dim)
        self.do_train = do_train
        self.apply(weight_init)

    def forward(self, frames):
        B,C,T,H,W = frames.shape
        scores = self.score_network(frames)

        frames = rearrange(frames, "b c t h w -> b t (h w c)")
        if self.training and self.do_train:
            indicator_topk, indicator_back = self.get_indicator(scores)
            frames_topk = extract_from_indicators(frames, indicator_topk)
            frames_back = extract_from_indicators(frames, indicator_back)
        else:
            indices_topk, indices_back = self.get_indices(scores)
            frames_topk = extract_from_indices(frames, indices_topk)
            frames_back = extract_from_indices(frames, indices_back)
        _, sorted_inds = torch.sort(scores, dim=1,)  
        
        frames_topk = rearrange(frames_topk, "b k (h w c) -> b c k h w", b=B, k=self.k, h=H, w=W)
        frames_back = rearrange(frames_back, "b nk (h w c) -> b c nk h w", b=B, nk=T-self.k, h=H, w=W)
        return frames_topk, frames_back, sorted_inds
    
    def get_indicator(self, score):
        indicator_topk, indicator_back = PerturbedRankFunction.apply(score, self.k, self.num_samples, self.sigma)
        indicator_topk = einops.rearrange(indicator_topk, "b k d -> b d k")
        indicator_back = einops.rearrange(indicator_back, "b k d -> b d k")
        return indicator_topk, indicator_back

    def get_indices(self, score):
        k = self.k
        topk_results = torch.topk(score, k=score.shape[1], dim=-1, sorted=True)
        indices = topk_results.indices # b, k
        indices_topk, indices_back = indices[..., :k], indices[..., k:]
        indices_topk = torch.sort(indices_topk, dim=-1).values
        indices_back = torch.sort(indices_back, dim=-1).values
        return indices_topk, indices_back



class ScoreNet(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim: int=384):
        super().__init__()
        embed_dim *= 2
        self.downsample = torch.nn.Conv3d(embed_dim//2, embed_dim//2, (1,3,3), (1,2,2))
        self.bn = torch.nn.BatchNorm3d(embed_dim//2)

        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        
        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1)
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = x.detach()  # stop gradient here
        x = self.bn(self.downsample(x))
        
        x = rearrange(x, 'b c t h w -> b t (h w) c')
        avg = torch.mean(x, dim=2, keepdim=False)
        max = torch.max(x, dim=2).values
        x = torch.cat((avg, max), dim=2)
        
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:,:, :C//2]
        
        global_x = torch.mean(x[:,:, C//2:], dim=1, keepdim=True)
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1)

        score =  self.out_conv(x)
        score = score.squeeze(-1)
        score = self.min_max_norm(score)
        return score

    def min_max_norm(self, x):
        flatten_score_min = x.min(axis=-1, keepdim=True).values
        flatten_score_max = x.max(axis=-1, keepdim=True).values
        norm_flatten_score = (x - flatten_score_min) / (flatten_score_max - flatten_score_min + 1e-5)
        return norm_flatten_score


def batched_index_select(input, dim, index):
    for i in range(1, len(input.shape)):
        if i != dim:
            index = index.unsqueeze(i)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def extract_from_indices(x, indices):
    batch_size, _, channels = x.shape
    k = indices.shape[-1]
    patches = x
    patches = batched_index_select(patches, 1, indices)
    patches = patches.contiguous().view(batch_size, k, channels)
    return patches


def extract_from_indicators(x, indicators):
    indicators = rearrange(indicators, "b d k -> b k d")
    patches = torch.bmm(indicators, x)
    return patches
