
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Tuple, Union, Any, List, Iterable, Optional
from .blocks import Bottleneck, AttentionPool2d
class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """
    def __init__(
        self,
        layers: Tuple[int, int, int, int],
        output_dim: int,
        input_resolution: int = 224,
        width: int = 64,
        heads: int = 8,
        features_only: bool = False,
        out_indices: Optional[Iterable[int]] = None,
        reduction: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        input_resolution = (input_resolution, input_resolution) if isinstance(input_resolution, int) else input_resolution
        assert isinstance(input_resolution, tuple) and len(input_resolution) == 2, f"input_resolution should be a tuple of length 2, but got {input_resolution}"
        self.input_resolution = input_resolution
        self.downsampling_rate = 32  # the rate at which the input is downsampled by the network

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=1 if reduction <= 16 else 2)

        self.features_only = features_only
        if features_only:
            self.out_indices = out_indices if out_indices is not None else range(5)
            self.out_indices = [idx + 5 if idx < 0 else idx for idx in self.out_indices]  # map negative indices to positive indices
            self.out_indices = sorted(set(self.out_indices))  # remove duplicates and sort
            assert min(self.out_indices) >= 0 and max(self.out_indices) <= 4, f"out_indices={self.out_indices} is invalid for a ResNet with 5 stages"
            self.channels = width * 32  # the ResNet feature dimension
        else:
            self.out_indices = None
            embed_dim = width * 32  # the ResNet feature dimension
            self.attnpool = AttentionPool2d((input_resolution[0] // 32) * (input_resolution[1] // 32), embed_dim, heads, output_dim)
            self.channels = output_dim

        self.reduction = self.downsampling_rate // 2 if reduction <= 16 else self.downsampling_rate
        self.clip_embed_dim = output_dim

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def _stem(self, x: Tensor) -> Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x: Tensor) -> Union[Tensor, List[Tensor]]:
        x = x.type(self.conv1.weight.dtype)
        x = self._stem(x)

        feats = [x] if self.features_only and 0 in self.out_indices else []

        x = self.layer1(x)
        if self.features_only and 1 in self.out_indices:
            feats.append(x)

        x = self.layer2(x)
        if self.features_only and 2 in self.out_indices:
            feats.append(x)

        x = self.layer3(x)
        if self.features_only and 3 in self.out_indices:
            feats.append(x)

        x = self.layer4(x)
        if self.features_only and 4 in self.out_indices:
            feats.append(x)

        if self.features_only:
            if len(self.out_indices) == 1:
                return feats[0]
            else:
                return feats
        else:
            x = self.attnpool(x)
            return x