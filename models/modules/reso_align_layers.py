import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import build_transformer_layer
import copy
import numpy as np
from einops import rearrange
from models.modules.dccm import DiffContextAwareCompressionModule


class ResolutionAlignTransformerLayerSequence(nn.Module):
    def __init__(self, transformerlayers=None, num_layers=None, compression_args=None, init_cfg=None):
        super().__init__()
        if isinstance(transformerlayers, dict):
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)
            ]
        else:
            assert isinstance(transformerlayers, list) and \
                   len(transformerlayers) == num_layers
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(build_transformer_layer(transformerlayers[i]))
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm

        compression_args = copy.deepcopy(compression_args)
        comp_module = compression_args['comp_module']
        comp_insert_layer = compression_args['comp_insert_layer']
        comp_k = compression_args['comp_k']
        comp_strategy = compression_args['comp_strategy']
        embed_dim = compression_args['embed_dim']
        embedding_temporal_size = compression_args['temporal_size']
        num_samples=compression_args['num_samples']
        sigma = compression_args['sigma']

        self.num_frames = embedding_temporal_size
        self.comp_insert_layer = comp_insert_layer
        self.sigma = sigma


        if comp_insert_layer > 0:
            self.dccm = DiffContextAwareCompressionModule(
                                                        in_channels=int(embed_dim),
                                                        T=embedding_temporal_size,
                                                        strategy=comp_strategy,
                                                        comp_module=comp_module,
                                                        k=comp_k,
                                                        sigma=sigma,
                                                        n_sample=num_samples)
        else:
            self.dccm = None

    def forward(self, query, key=None, value=None,):
        # query: (bs, 1+T*P, c)
 
        B, N, c = query.shape 
        T = self.num_frames
        P = (N - 1) // T
        H = W = int( np.sqrt(P))

        multi_resolution = False
        sort_index =None
        for i, layer in enumerate(self.layers):
           
            if self.dccm is not None and i == self.comp_insert_layer:
                cls_token, ts_tokens = query[:, 0].unsqueeze(1), query[:, 1:]
                ts_tokens = rearrange(ts_tokens, "b (h w t) c -> b c t h w", b=B, t=T, c=c, h=H, w=W)
                # split token to different resolutions
                # ts_tokens_high: (B, H1, W1, T1, C), ts_tokens_low:(B, H2, W2, T2, C)
                ts_tokens_high, ts_tokens_low, sort_index = self.dccm(ts_tokens, cls_token.squeeze(1),return_index=True)
                multi_resolution = True
                query = (cls_token, ts_tokens_high, ts_tokens_low)

            query = layer(query)
            #print(f"layers[{i}]: {query[0].sum()}")

        if multi_resolution:
            x_cls, x_high, x_low = query
            x_high = rearrange(x_high, "b h w t c -> b (h w t) c")
            x_low = rearrange(x_low, "b h w t c -> b (h w t) c")
            x = torch.cat([x_cls, x_high, x_low], dim=1)
            query = x
        return query
    
    def get_rank_index(self, query):
        # query: (bs, 1+T*P, c)
        B, N, c = query.shape 
        T = self.num_frames
        P = (N - 1) // T
        H = W = int( np.sqrt(P))
        if self.dccm is None:
            return torch.arange(T).unsqueeze(0).repeat(B, 1).to(query.device)

        for i, layer in enumerate(self.layers):
            if i == self.comp_insert_layer:
                cls_token, ts_tokens = query[:, 0].unsqueeze(1), query[:, 1:]
                ts_tokens = rearrange(ts_tokens, "b (h w t) c -> b c t h w", b=B, t=T, c=c, h=H, w=W)
                # split token to different resolutions
                # ts_tokens_high: (B, H1, W1, T1, C), ts_tokens_low:(B, H2, W2, T2, C)
                ts_tokens_high, ts_tokens_low, score_index = self.dccm(ts_tokens, cls_token.squeeze(1), return_index=True)
                query = (cls_token, ts_tokens_high, ts_tokens_low)
                break
            query = layer(query)

        return score_index

