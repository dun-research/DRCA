import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import ATTENTION, FEEDFORWARD_NETWORK
from einops import rearrange
from mmaction.models.common import DividedTemporalAttentionWithNorm

import torch.functional as F

from mmcv.cnn import (Linear, build_activation_layer, build_norm_layer)
from mmcv.runner.base_module import BaseModule, Sequential
from mmcv.utils import deprecated_api_warning
from mmcv.runner.base_module import BaseModule
from mmcv.cnn import build_norm_layer, constant_init
from mmcv.cnn.bricks.registry import ATTENTION, FEEDFORWARD_NETWORK
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.utils import digit_version


def _weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)


@ATTENTION.register_module()
class ResolutionAlignDividedTemporalAttentionWithNorm(DividedTemporalAttentionWithNorm):
    def __init__(self, embed_dims, num_heads, num_frames, attn_drop=0, proj_drop=0, dropout_layer=dict(type='DropPath', drop_prob=0.1), norm_cfg=dict(type='LN'), init_cfg=None, **kwargs):
        super().__init__(embed_dims, num_heads, num_frames, attn_drop, proj_drop, dropout_layer, norm_cfg, init_cfg, **kwargs)
       
        self.temporal_downsample = nn.AvgPool2d(kernel_size=2)
        self.temporal_upsample = torch.nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, query, key=None, value=None, residual=None, **kwargs):
        assert residual is None, (
            'Always adding the shortcut in the forward function')

        if isinstance(query, torch.Tensor):
            # single resolution
            return super().forward(query, key, value, residual, **kwargs)
        else:
            # multiple resolutions
            (cls_token, x_high, x_low) = query[:3]
            if len(query) == 4: 
                sort_index = query[-1]
            else:
                sort_index = None
            B, h_high, w_high, t_high, C = x_high.shape
            B, h_low, w_low, t_low, _ = x_low.shape

            xt_high_identity = x_high
            xt_low_identity = x_low

            # downsample
            x_high = rearrange(x_high, "b h w t c -> (b t) c h w",)
            x_high_ds = self.temporal_downsample(x_high)
            #print(f"x_high_ds:{x_high_ds.size()}, x_high:{x_high.size()}, x_low:{x_low.size()}")
            new_h, new_w = x_high_ds.size(2), x_high_ds.size(3) 

            # concat x_high and x_low
            x_high_ds = rearrange(x_high_ds, "(b t) c h w -> b (h w) t c", t=t_high)
            x_low = rearrange(x_low, "b h w t c -> b (h w) t c", t=t_low)
            query_t = torch.cat([x_high_ds, x_low,], dim=2)
            query_t = rearrange(query_t, "b p t c -> b (p t) c")

            # compute residual
            res_temporal = self.compute_residual(query_t)

            # split x_high and x_low
            res_temporal = rearrange(res_temporal, "(b h w) t c -> b h w t c", 
                                                    b=B, h=h_low, w=w_low, t=t_high+t_low, c=C,)
            res_temporal_high, res_temporal_low = torch.split(res_temporal, [t_high, t_low], dim=3)

            # upsample res_temporal_high
            res_temporal_high = rearrange(res_temporal_high, "b h w t c -> (b t) c h w", h=new_h, w=new_w)
            res_temporal_high = self.temporal_upsample(res_temporal_high)
            res_temporal_high = rearrange(res_temporal_high, "(b t) c h w -> b h w t c", t=t_high)
            
            new_xt_high = xt_high_identity + res_temporal_high
            new_xt_low = xt_low_identity + res_temporal_low
            if sort_index is None:
                return cls_token, new_xt_high, new_xt_low           
            else:
                return cls_token, new_xt_high, new_xt_low,sort_index       

    def compute_residual(self, query_t, ):
        # query_t [batch_size, num_patches * num_frames, embed_dims]
        b, pt, m = query_t.size()
        p, t = pt // self.num_frames, self.num_frames

        # res_temporal [batch_size * num_patches, num_frames, embed_dims]
        query_t = self.norm(query_t.reshape(b * p, t, m)).permute(1, 0, 2)
        res_temporal = self.attn(query_t, query_t, query_t)[0].permute(1, 0, 2)
        res_temporal = self.dropout_layer(
            self.proj_drop(res_temporal.contiguous()))
        res_temporal = self.temporal_fc(res_temporal)
        return res_temporal



@ATTENTION.register_module()
class ResolutionAlignDividedSpatialAttentionWithNorm(BaseModule):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 num_frames,
                 single_resolution=True,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='DropPath', drop_prob=0.1),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.norm = build_norm_layer(norm_cfg, self.embed_dims)[1]
        
        if digit_version(torch.__version__) < digit_version('1.9.0'):
            kwargs.pop('batch_first', None)

        if not single_resolution:
            self.norm_low = build_norm_layer(norm_cfg, self.embed_dims)[1]
            self.attn_low = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                             **kwargs)
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    def forward_one(self, query, key=None, value=None, residual=None, **kwargs):
        assert residual is None, (
            'Always adding the shortcut in the forward function')

        identity = query
        init_cls_token = query[:, 0, :].unsqueeze(1)
        query_s = query[:, 1:, :]

        # query_s [batch_size, num_patches * num_frames, embed_dims]
        b, pt, m = query_s.size()
        p, t = pt // self.num_frames, self.num_frames

        # cls_token [batch_size * num_frames, 1, embed_dims]
        cls_token = init_cls_token.repeat(1, t, 1).reshape(b * t,
                                                           m).unsqueeze(1)

        # query_s [batch_size * num_frames, num_patches + 1, embed_dims]
        query_s = rearrange(query_s, 'b (p t) m -> (b t) p m', p=p, t=t)
        query_s = torch.cat((cls_token, query_s), 1)

        # res_spatial [batch_size * num_frames, num_patches + 1, embed_dims]
        query_s = self.norm(query_s).permute(1, 0, 2)
        res_spatial = self.attn(query_s, query_s, query_s)[0].permute(1, 0, 2)
        res_spatial = self.dropout_layer(
            self.proj_drop(res_spatial.contiguous()))

        # cls_token [batch_size, 1, embed_dims]
        cls_token = res_spatial[:, 0, :].reshape(b, t, m)
        cls_token = torch.mean(cls_token, 1, True)

        # res_spatial [batch_size * num_frames, num_patches + 1, embed_dims]
        res_spatial = rearrange(
            res_spatial[:, 1:, :], '(b t) p m -> b (p t) m', p=p, t=t)
        res_spatial = torch.cat((cls_token, res_spatial), 1)

        new_query = identity + res_spatial
        return new_query

    def forward(self, query, key=None, value=None, residual=None, **kwargs):
        assert residual is None, (
            'Always adding the shortcut in the forward function')

        if isinstance(query, torch.Tensor):
            # single resolution
            return self.forward_one(query, key, value, residual, **kwargs)
        else:
            # multiple resolutions
            (x_cls, xs_high, xs_low) = query

            x_high_shape = xs_high.shape
            x_low_shape = xs_low.shape
            b = xs_high.shape[0]

            xs_high = rearrange(xs_high, "b h w t c -> b (h w t) c")
            xs_low = rearrange(xs_low, "b h w t c -> b (h w t) c")

            res_spatial_high, cls_token_high =  self.compute_residual(xs_high, x_cls, t=x_high_shape[3],mode="high")
            res_spatial_low, cls_token_low = self.compute_residual(xs_low, x_cls,t=x_low_shape[3], mode="low")

            cls_token = torch.cat([cls_token_high, cls_token_low], dim=1)
            cls_token = torch.mean(cls_token, 1, True) # averaging for every frame

            res_spatial = torch.cat([cls_token, res_spatial_high, res_spatial_low], dim=1)
            identity = torch.cat([x_cls, xs_high, xs_low], dim=1)

            new_query = identity + res_spatial     
            new_cls, new_high, new_low = torch.split(new_query, [1, xs_high.shape[1], xs_low.shape[1]], dim=1)
            new_high = new_high.reshape(*x_high_shape)
            new_low = new_low.reshape(*x_low_shape)
            return new_cls, new_high, new_low

    def compute_residual(self, query_s, init_cls_token, t, mode):
        # query_s [batch_size, num_patches * num_frames, embed_dims]
        b, pt, m = query_s.size()
        p = pt // t

        # cls_token [batch_size * num_frames, 1, embed_dims]
        cls_token = init_cls_token.repeat(1, t, 1).reshape(b * t, m).unsqueeze(1)

        # query_s [batch_size * num_frames, num_patches + 1, embed_dims]
        query_s = rearrange(query_s, 'b (p t) m -> (b t) p m', p=p, t=t)
        query_s = torch.cat((cls_token, query_s), 1)

        if mode == "high":
            query_s = self.norm(query_s).permute(1, 0, 2)
            res_spatial = self.attn(query_s, query_s, query_s)[0].permute(1, 0, 2)
        else:
            query_s = self.norm_low(query_s).permute(1, 0, 2)
            res_spatial = self.attn_low(query_s, query_s, query_s)[0].permute(1, 0, 2)

        res_spatial = self.dropout_layer(
            self.proj_drop(res_spatial.contiguous()))
        
        cls_token = res_spatial[:, 0, :].reshape(b, t, m)
        res_spatial = rearrange(
            res_spatial[:, 1:, :], '(b t) p m -> b (p t) m', p=p, t=t)
        return res_spatial, cls_token


@FEEDFORWARD_NETWORK.register_module()
class ResolutionAlignFFN(FFN):
    """FFN with pre normalization layer.

    FFNWithNorm is implemented to be compatible with `BaseTransformerLayer`
    when using `DividedTemporalAttentionWithNorm` and
    `DividedSpatialAttentionWithNorm`.

    FFNWithNorm has one main difference with FFN:

    - It apply one normalization layer before forwarding the input data to
        feed-forward networks.

    Args:
        embed_dims (int): Dimensions of embedding. Defaults to 256.
        feedforward_channels (int): Hidden dimension of FFNs. Defaults to 1024.
        num_fcs (int, optional): Number of fully-connected layers in FFNs.
            Defaults to 2.
        act_cfg (dict): Config for activate layers.
            Defaults to `dict(type='ReLU')`
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Defaults to 0..
        add_residual (bool, optional): Whether to add the
            residual connection. Defaults to `True`.
        dropout_layer (dict | None): The dropout_layer used when adding the
            shortcut. Defaults to None.
        init_cfg (dict): The Config for initialization. Defaults to None.
        norm_cfg (dict): Config dict for normalization layer. Defaults to
            `dict(type='LN')`.
    """

    def __init__(self, *args, norm_cfg=dict(type='LN'), **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = build_norm_layer(norm_cfg, self.embed_dims)[1]

    def forward(self, x, residual=None):
        assert residual is None, ('Cannot apply pre-norm with FFNWithNorm')
        if isinstance(x, torch.Tensor):
            return super().forward(self.norm(x), x)
        else:
            (cls_token, x_high, x_low) = x[:3]
            if len(x) ==4:
                sort_index = x[-1]
            else:
                sort_index = None

            x_high_shape = x_high.shape
            x_low_shape = x_low.shape

            x_high = rearrange(x_high, "b h w t c -> b (h w t) c")
            x_low = rearrange(x_low, "b h w t c -> b (h w t) c")
            len_high, len_low = x_high.size(1), x_low.size(1)
            
            x = torch.cat([cls_token, x_high, x_low], dim=1)
            x = super().forward(self.norm(x), x)
            
            x_cls, x_high, x_low = torch.split(x, [1, len_high, len_low], dim=1)
            x_high = x_high.reshape(*x_high_shape)
            x_low = x_low.reshape(*x_low_shape)
            if sort_index == None:
                return x_cls, x_high, x_low
            else:
                return x_cls, x_high, x_low, sort_index


