import numpy
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from einops import rearrange
from torch.nn import init
from math import sqrt

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop_rate=0.0,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        if self.drop_rate > 0.0:
            self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        x = self.fc2(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        return x


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def attention_pool(tensor, pool, hw_shape, has_cls_embed=True, norm=None):
    if pool is None:
        return tensor, hw_shape
    tensor_dim = tensor.ndim
    tensor_shape = tensor.shape
    #print(f"tensor: {tensor.shape}, pool:{pool}")
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    elif tensor_dim == 5:
        tensor = tensor.reshape((-1,)+tensor_shape[2:])
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape
    #H, W = hw_shape
    H = W = int(sqrt(L))
    tensor = tensor.reshape(B * N, H, W, C).permute(0, 3, 1, 2).contiguous()

    tensor = pool(tensor)
    #print(f"poolout: {tensor.shape}")
    hw_shape = [tensor.shape[2], tensor.shape[3]]
    L_pooled = tensor.shape[2] * tensor.shape[3]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
    if norm is not None:
        tensor = norm(tensor)

    if tensor_dim == 3:
        tensor = tensor.squeeze(1)
    elif tensor_dim == 5:
        tensor = tensor.reshape((tensor_shape[0], tensor_shape[1]) + tensor.shape[1:])
    return tensor, hw_shape


def cal_rel_pos_spatial(
    attn,
    q,
    has_cls_embed,
    q_shape,
    k_shape,
    rel_pos_h,
    rel_pos_w,
):
    """
    Spatial Relative Positional Embeddings.
    """
    sp_idx = 1 if has_cls_embed else 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
        torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio
    )
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, :, None]
        + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn

class RefAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        input_size,
        num_heads=8,
        qkv_bias=False,
        kernel_q=(1, 1),
        kernel_kv=(1, 1),
        stride_q=(1, 1),
        stride_kv=(1, 1),
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        mode="conv",
        pool_first=False,
        rel_pos_spatial=False,
        rel_pos_zero_init=False,
        residual_pooling=True,
    ):
        super().__init__()
        self.pool_first = pool_first

        self.num_heads = num_heads
        self.dim_out = dim_out
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5
        self.has_cls_embed = has_cls_embed
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv =[int(kv // 2) for kv in kernel_kv]

        if pool_first:
            self.q = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.k = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.v = nn.Linear(dim, dim_out, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim_out, dim_out)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if numpy.prod(kernel_q) == 1 and numpy.prod(stride_q) == 1:
            kernel_q = ()
        if numpy.prod(kernel_kv) == 1 and numpy.prod(stride_kv) == 1:
            kernel_kv = ()
        self.mode = mode

        if mode in ("avg", "max"):
            pool_op = nn.MaxPool2d if mode == "max" else nn.AvgPool2d
            self.pool_q = (
                pool_op(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None
            )
            self.pool_k = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            self.pool_v = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
        elif mode == "conv" or mode == "conv_unshared":
            if pool_first:
                dim_conv = dim // num_heads if mode == "conv" else dim
            else:
                dim_conv = dim_out // num_heads if mode == "conv" else dim_out
            self.pool_q = (
                nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_q) > 0
                else None
            )
            self.norm_q = norm_layer(dim_conv) if len(kernel_q) > 0 else None
            self.pool_k = (
                nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_k = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
            self.pool_v = (
                nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_v = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

        # relative pos embedding
        self.rel_pos_spatial = rel_pos_spatial
        if self.rel_pos_spatial:
            assert input_size[0] == input_size[1]

            size = input_size[0]
            q_size = size // stride_q[1] if len(stride_q) > 0 else size
            kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size
            rel_sp_dim = 2 * max(q_size, kv_size) - 1

            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))

            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

        self.residual_pooling = residual_pooling

    def forward(self, x, x_ref, hw_shape):
        """
            x: (B, T1, N, C)
            x_ref: (B, T2, N, C)
        """
        B, T1, N, _ = x.shape
        B, T2, N, _ = x_ref.shape

        if self.pool_first:
            if self.mode == "conv_unshared":
                fold_dim = 1
            else:
                fold_dim = self.num_heads
            
            x = x.reshape(B, T1, N, fold_dim, -1) #.permute(0,  2, 1, 3)
            x_ref = x_ref.reshape(B, T2, N, fold_dim, -1) #.permute(0, 2, 1, 3)
            x = rearrange(x, "b t n f c -> b t f n c")
            x_ref = rearrange(x_ref, "b t n f c -> b t f n c")
            # q = k = v = x
            q = x
            k = v = x_ref

        else:
            raise NotImplemented
            assert self.mode != "conv_unshared"

            qkv = (
                self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]

        # print(f"step1: q({q.shape}), k({k.shape}), v({v.shape})")

        q, q_shape = attention_pool(
            q,
            self.pool_q,
            hw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
        )
        k, k_shape = attention_pool(
            k,
            self.pool_k,
            hw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
        )
        v, v_shape = attention_pool(
            v,
            self.pool_v,
            hw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
        )

        # print(f"step2: q({q.shape}), k({k.shape}), v({v.shape})")

        if self.pool_first:
            q_N = numpy.prod(q_shape) + 1 if self.has_cls_embed else numpy.prod(q_shape)
            k_N = numpy.prod(k_shape) + 1 if self.has_cls_embed else numpy.prod(k_shape)
            v_N = numpy.prod(v_shape) + 1 if self.has_cls_embed else numpy.prod(v_shape)
            # print(f"q_N={q_N}, k_N={k_N}, v_N={v_N}")

            # q = q.permute(0, 2, 1, 3).reshape(B, q_N, -1)
            # q = self.q(q).reshape(B, q_N, self.num_heads, -1).permute(0, 2, 1, 3)

            # v = v.permute(0, 2, 1, 3).reshape(B, v_N, -1)
            # v = self.v(v).reshape(B, v_N, self.num_heads, -1).permute(0, 2, 1, 3)

            # k = k.permute(0, 2, 1, 3).reshape(B, k_N, -1)
            # k = self.k(k).reshape(B, k_N, self.num_heads, -1).permute(0, 2, 1, 3)

            q = rearrange(q, "b t nh qn nd ->  (b t) qn (nh nd)")
            q = self.q(q)
            q = rearrange(q, "(b t) qn (nh nd) -> b t nh qn nd", nh=self.num_heads, b=B)

            v = rearrange(v, "b t nh qn nd ->  (b t) qn (nh nd)")
            v = self.v(v)
            v = rearrange(v, "(b t) qn (nh nd) -> b t nh qn nd", nh=self.num_heads, b=B)

            k = rearrange(k, "b t nh qn nd ->  (b t) qn (nh nd)")
            k = self.k(k)
            k = rearrange(k, "(b t) qn (nh nd) -> b t nh qn nd", nh=self.num_heads, b=B)

        # step3: q(torch.Size([3, 6, 4, 49, 8])), k(torch.Size([3, 2, 4, 196, 8])), v(torch.Size([3, 2, 4, 196, 8]))
        # print(f"step3: q({q.shape}), k({k.shape}), v({v.shape})")
        N = q.shape[2]

        # k.transpose(-2, -1)=torch.Size([2, 4, 8, 196]), attn: torch.Size([2, 4, 49, 196])
        q = rearrange(q, "b t nh qn nd -> b nh (t qn) nd")
        k = rearrange(k, "b t nh kn nd -> b nh (t kn) nd")
        # print(f"q: {q.shape}, k: {k.shape}")
        attn = (q * self.scale) @ k.transpose(-2, -1)

        # print(f"k.transpose(-2, -1)={k.transpose(-2, -1).shape}, attn: {attn.shape}")
        if self.rel_pos_spatial:
            print(f"do rel_pos_spatial")
            attn = cal_rel_pos_spatial(
                attn,
                q,
                self.has_cls_embed,
                q_shape,
                k_shape,
                self.rel_pos_h,
                self.rel_pos_w,
            )
        else:
            # print("skip rel_pos_spatial")
            pass

        # print(f"rel_pos_spatial - attn: {attn.shape}")

        attn = attn.softmax(dim=-1)
        v = rearrange(v, "b t nh vn nd -> b nh (t vn) nd")
        x = attn @ v

        # print(f"after-attn-x: {x.shape}")


        if self.residual_pooling:
            if self.has_cls_embed:
                x[:, :, 1:, :] += q[:, :, 1:, :]
            else:
                x = x + q

        x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
        x = self.proj(x)

        # final-attn-x: torch.Size([3, 294, 32])
        # print(f"final-attn-x: {x.shape}")
        x = rearrange(x, "b (t xn) c -> b t xn c", t=T1)

        return x, q_shape


class RefBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        input_size,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        kernel_q=(1, 1),
        kernel_kv=(1, 1),
        stride_q=(1, 1),
        stride_kv=(1, 1),
        mode="conv",
        has_cls_embed=True,
        
        rel_pos_spatial=False,
        rel_pos_zero_init=False,
        residual_pooling=True,
        dim_mul_in_att=False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.dim_mul_in_att = dim_mul_in_att

        att_dim = dim_out if dim_mul_in_att else dim
        self.attn = RefAttention(
            dim,
            att_dim,
            num_heads=num_heads,
            input_size=input_size,
            qkv_bias=qkv_bias,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=norm_layer,
            has_cls_embed=has_cls_embed,
            mode=mode,
            pool_first=True,
            rel_pos_spatial=rel_pos_spatial,
            rel_pos_zero_init=rel_pos_zero_init,
            residual_pooling=residual_pooling,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(att_dim)
        mlp_hidden_dim = int(att_dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed

        mlp_dim_out = dim_out
        self.mlp = Mlp(
            in_features=att_dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
        )
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        if len(stride_q) > 0 and numpy.prod(stride_q) > 1:
            kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
            stride_skip = stride_q
            padding_skip = [int(skip // 2) for skip in kernel_skip]
            self.pool_skip = nn.MaxPool2d(
                kernel_skip, stride_skip, padding_skip, ceil_mode=False
            )
        else:
            self.pool_skip = None

    def forward(self, x, x_ref):
        
        hw_shape = (x.shape[3], x.shape[4])
        x = rearrange(x, 'b c t h w -> b t (h w) c')
        x_ref = rearrange(x_ref, 'b c t h w -> b t (h w) c')
        x_norm = self.norm1(x)
        x_block, hw_shape_new = self.attn(x_norm, x_ref, hw_shape)

        # print(f"DiffAttention: x_norm({x_norm.shape}) -> x_block({x_block.shape})")

        if self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)

        x_res, _ = attention_pool(
            x, self.pool_skip, hw_shape, has_cls_embed=self.has_cls_embed
        )
        # print(f"attention_pool: x({x.shape}) -> x_res({x_res.shape})")
        
        x = x_res + self.drop_path(x_block)
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)

        if not self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp)
        # input :torch.Size([2, 196, 32]) ->  output: torch.Size([3, 6, 49, 32])
        x = rearrange(x, "b t (h w) c -> b c t h w", h=hw_shape_new[0], w=hw_shape_new[1])
        return x


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        init.constant_(module.weight, 1)
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        init.normal_(module.weight, 0, 0.01)
        if module.bias is not None:
            init.constant_(module.bias, 0)


if __name__ == "__main__":
    print("\n======================================\n")
    kr_block = RefBlock(dim=32,
                dim_out=32,
                num_heads=4,
                input_size=(7, 7),
                mode="conv",
                has_cls_embed=False,
                kernel_q=(1, 1),
                kernel_kv=(3, 3),
                stride_q=(1, 1),
                stride_kv=(2, 2),
            )

    x = torch.rand(3, 32, 6, 7, 7)
    x_ref = torch.rand(3, 32, 2, 7, 7,)
    out = kr_block(x, x_ref)
    print(f"input :{x.shape} ->  output: {out.shape}")
