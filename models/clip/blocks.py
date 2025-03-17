import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# ODConv2D-Based Block for Adaptive Feature Extraction
class ODConvBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv = ODConv2d(width, width, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(width)  

        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

# Attention Module for ODConv2D
class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.norm = LayerNorm2D(attention_channel)  # ✅ Replace with LayerNorm

        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True) if in_planes != out_planes else None

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_channel_attention(self, x):
        return torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)

    def get_filter_attention(self, x):
        return torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature) if self.filter_fc else 1.0

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.norm(x)
        x = self.relu(x)
        return self.get_channel_attention(x), self.get_filter_attention(x)

# ODConv2D Implementation with Adaptive Filtering
class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, reduction=0.0625, kernel_num=4):
        super(ODConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups, reduction=reduction, kernel_num=kernel_num)

        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes // groups, kernel_size, kernel_size), requires_grad=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        channel_attn, filter_attn = self.attention(x)
        x = x * channel_attn
        aggregate_weight = self.weight.sum(dim=0) * filter_attn
        return F.conv2d(x, weight=aggregate_weight, bias=None, stride=1, padding=self.kernel_size // 2, groups=self.groups)
    
class LayerNorm2D(nn.Module):
    """Custom LayerNorm for CNN feature maps (Vision-based)"""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = torch.sqrt(x.var(dim=(2, 3), keepdim=True, unbiased=False) + self.eps)
        std = torch.clamp(std, min=1e-3)  # Prevents instability
        
        out = (x - mean) / std * self.weight + self.bias

        # Debugging checks
        if torch.isnan(out).any():
            print("⚠️ NaN detected in LayerNorm2D output!")

        return out

# LayerNorm for 2D Inputs
class LayerNorm(nn.LayerNorm):
    def forward(self, x):

        # Ensure x is 3D [Batch, Sequence, Feature]
        if x.dim() == 3:  # Expected shape for text encoder
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        print("x.shape before unpacking:", x.shape)

        # If the shape is incorrect, reshape it
        B, T, C = x.shape  # Batch, Tokens, Channels
        x = x.reshape(B * T, C)  # Flatten across batch & tokens
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.reshape(B, T, C)  # Restore shape

        print("LayerNorm Output Shape:", x.shape)  # Debugging step
        return x




# Density-Aware Pooling for CLIP Feature Aggregation
class DensityAwarePooling(nn.Module):
    def __init__(self, embed_dim, output_dim):
        super().__init__()
        self.attention = Attention(embed_dim, output_dim, kernel_size=1)
        self.conv = nn.Conv2d(embed_dim, output_dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(output_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        channel_attn, filter_attn = self.attention(x)
        x = x * channel_attn * filter_attn
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: Tensor):
        return self.resblocks(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)