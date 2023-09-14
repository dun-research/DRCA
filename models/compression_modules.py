import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.weight_init import weight_init
from models.ref_compress import RefBlock
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor


def conv3x3_2d(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1_2d(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv1x1x1_3d(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

norm_layer = nn.BatchNorm2d
class Bottleneck2D(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        width: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        # groups: int = 1,
        # base_width: int = 32,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 32.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1_2d(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3_2d(width, width, stride, 1, dilation)
        self.bn2 = norm_layer(width)
        # self.conv3 = conv1x1_2d(width, planes * self.expansion)
        # self.bn3 = norm_layer(planes * self.expansion)
        self.conv3 = conv1x1_2d(width, inplanes)
        self.bn3 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # print(f"out-bn3: {out.shape}")
        if self.downsample is not None:
            identity = self.downsample(x)
        # print(f"identity: {identity.shape}")
        out += identity
        out = self.relu(out)

        return out


class Bottleneck3D(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        width: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        # groups: int = 1,
        # base_width: int = 32,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        # width = int(planes * (base_width / 32.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1x1_3d(inplanes, width, )
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv3d(width, width, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False)
        self.bn2 = norm_layer(width)
        # self.conv3 = conv1x1_2d(width, planes * self.expansion)
        # self.bn3 = norm_layer(planes * self.expansion)
        self.conv3 = conv1x1x1_3d(width, inplanes)
        self.bn3 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        print(f"out-bn3: {out.shape}")
        if self.downsample is not None:
            identity = self.downsample(x)
        print(f"identity: {identity.shape}")
        out += identity
        out = self.relu(out)

        return out

class CompressionConv2D(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(in_channels),
        ) 
        self.apply(weight_init)

    def forward(self, x, x_ref=None):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        out = self.downsample(x)
        W = x.size(-1)
        out = rearrange(out, '(b t) c h w -> b c t h w', b=B,)
        return out
    

class ResBlock2D(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
                conv1x1_2d(in_channels, in_channels, 2),
                norm_layer(in_channels),
            )
        self.bottle_neck = Bottleneck2D(in_channels, in_channels, width=in_channels*2, stride=2, downsample=self.downsample)
    
    def forward(self, x, x_ref):
        B = x.size(0)
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        out = self.bottle_neck(x)
        W = x.size(-1)
        out = rearrange(out, '(b t) c h w -> b c t h w', b=B,)
        return out



class ResBlock3D(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False ),
                nn.BatchNorm3d(in_channels),
            )
        self.bottle_neck = Bottleneck3D(in_channels, in_channels, width=in_channels*2, stride=2, downsample=self.downsample)
    
    def forward(self, x, x_ref):
        # x.shape=(b c t h w)
        B = x.size(0)
        out = self.bottle_neck(x)
        return out


class CompressionConv3D(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
                nn.Conv3d(in_channels, in_channels*2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
                nn.BatchNorm3d(in_channels*2),
                nn.Conv3d(in_channels*2, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(in_channels),
        ) 
        self.apply(weight_init)

    def forward(self, x, x_ref=None):
        B, C, T, H, W = x.shape
        out = self.reduce(x)
        # print(f"out: {out.shape}" )
        return out


class PatchMerge3D(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, in_channels, norm_layer=nn.LayerNorm):
        super().__init__()
        dim = in_channels
        # swap temporal information
        self.in_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(1, 1, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(in_channels),
            nn.Conv3d(in_channels, in_channels, kernel_size=(1, 1, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm3d(in_channels),
        )
        self.reduction = nn.Linear(4 * dim, dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.apply(weight_init)

    def forward(self, x, x_ref=None):
        B, D, H, W, C = x.shape
        x = self.in_conv(x)
        x = rearrange(x, 'b c t h w -> b t  h w c')
        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        x = rearrange(x, 'b t  h w c -> b c t h w ')
 
        return x

 
def get_saliency_frame_reference_compression(in_channels):
    model =  RefBlock(dim=in_channels,
                dim_out=in_channels,
                num_heads=4,
                input_size=(7, 7),
                mode="conv",
                has_cls_embed=False,
                kernel_q=(3, 3),
                kernel_kv=(1, 1),
                stride_q=(2, 2),
                stride_kv=(1, 1),
            ) 
    return model


def get_compression_module(name, in_channels):
    if name == "conv2d":
        return CompressionConv2D(in_channels)
    elif name == "conv3d":
        return CompressionConv3D(in_channels)
    elif name == "patch_merge":
        return PatchMerge3D(in_channels)
    elif name == "resblock_2d":
        return ResBlock2D(in_channels)
    elif name == "resblock_3d":
        return ResBlock3D(in_channels)
    elif name == "sal_ref":
        return get_saliency_frame_reference_compression(in_channels)
    else:
        raise NotImplementedError(f"Unknown compression module: {name}")