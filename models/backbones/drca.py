
from collections import OrderedDict
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmcv.cnn import  build_norm_layer
from mmcv.runner import _load_checkpoint, load_state_dict
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmaction.utils import get_root_logger
from mmaction.models.builder import BACKBONES
from mmaction.models.backbones.timesformer import PatchEmbed
from models.modules.reso_align_layers import ResolutionAlignTransformerLayerSequence


@BACKBONES.register_module()
class DRCA(nn.Module):
    def __init__(self,
                 num_frames,
                 img_size,
                 patch_size,
                 pretrained=None,
                 embed_dims=768,
                 num_heads=12,
                 num_transformer_layers=12,
                 in_channels=3,
                 dropout_ratio=0.,
                 transformer_layers=None,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 # compression args
                 comp_insert_layer=-1, # if comp_insert_layer is -1, don't use compression
                 comp_k=4,
                 comp_strategy='diff_rank',
                 comp_module="sal_ref",
                 sigma=0.05,
                 num_samples=500,
                 **kwargs):
        super().__init__(**kwargs)

        assert transformer_layers is None or isinstance(
            transformer_layers, (dict, list))
        
        self.num_frames = num_frames
        self.pretrained = pretrained
        self.embed_dims = embed_dims
        self.num_transformer_layers = num_transformer_layers



        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims)
        num_patches = self.patch_embed.num_patches
        self.img_size = img_size
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dims))
        self.drop_after_pos = nn.Dropout(p=dropout_ratio)

        self.time_embed = nn.Parameter(
                torch.zeros(1, num_frames, embed_dims))
        self.drop_after_time = nn.Dropout(p=dropout_ratio)

        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

        single_resolutions = [True for i in range(num_transformer_layers)] 
        if comp_insert_layer >=0:
            n = comp_insert_layer
            for i in range(n, num_transformer_layers):
                single_resolutions[i] = False
            
        if transformer_layers is None:
            # stochastic depth decay rule
            dpr = np.linspace(0, 0.1, num_transformer_layers)
            _transformerlayers_cfg = [
                dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='ResolutionAlignDividedTemporalAttentionWithNorm',
                            embed_dims=embed_dims,
                            num_heads=num_heads,
                            num_frames=num_frames,
                            dropout_layer=dict(
                                type='DropPath', drop_prob=dpr[i]),
                            norm_cfg=dict(type='LN', eps=1e-6)),
                        dict(
                            type='ResolutionAlignDividedSpatialAttentionWithNorm',
                            embed_dims=embed_dims,
                            num_heads=num_heads,
                            num_frames=num_frames,
                            single_resolution=single_resolutions[i],
                            dropout_layer=dict(
                                type='DropPath', drop_prob=dpr[i]),
                            norm_cfg=dict(type='LN', eps=1e-6))
                    ],
                    ffn_cfgs=dict(
                            type='ResolutionAlignFFN',
                        single_resolution=single_resolutions[i],
                        embed_dims=embed_dims,
                        feedforward_channels=embed_dims * 4,
                        num_fcs=2,
                        act_cfg=dict(type='GELU'),
                        dropout_layer=dict(
                            type='DropPath', drop_prob=dpr[i]),
                        norm_cfg=dict(type='LN', eps=1e-6)),
                    operation_order=('self_attn', 'self_attn', 'ffn'))
                for i in range(num_transformer_layers)
            ]

        compression_args = dict(
                            comp_insert_layer=comp_insert_layer,
                            comp_k=comp_k,
                            comp_strategy=comp_strategy,
                            comp_module=comp_module,
                            sigma=sigma,
                            num_samples=num_samples,
                            embed_dim=embed_dims,
                            temporal_size=num_frames,
                            )

        self.transformer_layers = ResolutionAlignTransformerLayerSequence(_transformerlayers_cfg, num_transformer_layers, compression_args)

    def init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.time_embed, std=.02)

        if pretrained:
            self.pretrained = pretrained
        
            
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            state_dict = _load_checkpoint(self.pretrained)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            if not self.pretrained.startswith('imagenet'):
                self.load_state_dict(state_dict)

            self.attention_type = 'divided_space_time'
            if self.attention_type == 'divided_space_time':
                # modify the key names of norm layers
                old_state_dict_keys = list(state_dict.keys())
                for old_key in old_state_dict_keys:
                    if 'norms' in old_key:
                        new_key = old_key.replace('norms.0',
                                                  'attentions.0.norm')
                        new_key = new_key.replace('norms.1', 'ffns.0.norm')
                        state_dict[new_key] = state_dict.pop(old_key)

                # copy the parameters of space attention to time attention
                old_state_dict_keys = list(state_dict.keys())
                for old_key in old_state_dict_keys:
                    if 'attentions.0' in old_key:
                        new_key = old_key.replace('attentions.0',
                                                  'attentions.1')
                        state_dict[new_key] = state_dict[old_key].clone()

                # init pos_embedding
                ## Resizing the positional embeddings in case they don't match
                num_patches = (self.img_size // self.patch_size) * (self.img_size // self.patch_size)
                if num_patches + 1 != state_dict['pos_embed'].size(1):
                    pos_embed = state_dict['pos_embed']
                    cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
                    other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
                    new_pos_embed = F.interpolate(other_pos_embed, size=(num_patches), mode='nearest')
                    new_pos_embed = new_pos_embed.transpose(1, 2)
                    new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
                    state_dict['pos_embed'] = new_pos_embed
                    logger.info(f"resize pos_embed from {pos_embed.size()} to {new_pos_embed.size()}")

                load_state_dict(self, state_dict, strict=False, logger=logger)

        
    def forward(self, x):
        """Defines the computation performed at every call."""
        # x [batch_size * num_frames, num_patches, embed_dims]
        batches = x.shape[0]
        x = self.patch_embed(x)

        # x [batch_size * num_frames, num_patches + 1, embed_dims]
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.drop_after_pos(x)

        # Add Time Embedding
        # x [batch_size, num_patches * num_frames + 1, embed_dims]
        cls_tokens = x[:batches, 0, :].unsqueeze(1)
        x = rearrange(x[:, 1:, :], '(b t) p m -> (b p) t m', b=batches)
        x = x + self.time_embed
        x = self.drop_after_time(x)
        x = rearrange(x, '(b p) t m -> b (p t) m', b=batches)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.transformer_layers(x, None, None)
        x = self.norm(x)

        # Return Class Token
        return x[:, 0]

    def get_rank_index(self, x):
        batches = x.shape[0]
        x = self.patch_embed(x)

        # x [batch_size * num_frames, num_patches + 1, embed_dims]
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.drop_after_pos(x)

        # Add Time Embedding
        # x [batch_size, num_patches * num_frames + 1, embed_dims]
        cls_tokens = x[:batches, 0, :].unsqueeze(1)
        x = rearrange(x[:, 1:, :], '(b t) p m -> (b p) t m', b=batches)
        x = x + self.time_embed
        x = self.drop_after_time(x)
        x = rearrange(x, '(b p) t m -> b (p t) m', b=batches)
        x = torch.cat((cls_tokens, x), dim=1)

        index = self.transformer_layers.get_rank_index(x)
        return index
