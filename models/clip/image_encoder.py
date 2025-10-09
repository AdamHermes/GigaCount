import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional, Tuple, Union
from .blocks import ODConvBlock, DensityAwarePooling

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone,self).__init__()
        
        feats =list(convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).features.children())
        
        self.stem = nn.Sequential(*feats[0:2])
        self.stage1 = nn.Sequential(*feats[2:4])
        self.stage2 = nn.Sequential(*feats[4:6])
        #self.stage3 = nn.Sequential(*feats[6:12])
        
    def forward(self, x):
        x = x.float()
        x = self.stem(x)
        feature1 = x
        x = self.stage1(x)
        feature2 = x
        x = self.stage2(x)
        
        
        
        return feature1, feature2, x

# Run your training with this modified forward function


class ModifiedConvNext(nn.Module):
    def __init__(
        self,
        base_width: int,
        layers: List[int],
        output_dim: int,
        input_resolution: int = 224,
        heads: int = 8,
        features_only: bool = False,
        out_indices: Optional[List[int]] = None,
        reduction: int = 32,
    ) -> None:
        super().__init__()
        widths = [base_width * (2**i) for i in range(4)]
        norm = nn.BatchNorm2d
        self.features_only = features_only
        self.inplanes = widths[0]
        self.downsampling_rate = 32  # Match ResNet's downsampling rate

        # Stem (Downsampling to 1/4 resolution)
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=4, stride=4),
            norm(self.inplanes),
        )

        # Staged feature extraction layers
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for idx, (layer, width) in enumerate(zip(layers, widths)):
            self.stages.append(nn.Sequential(*[ODConvBlock(width) for _ in range(layer)]))
            if idx < 3:
                self.downsamples.append(
                    nn.Sequential(
                        norm(width),
                        nn.Conv2d(width, widths[idx + 1], kernel_size=2, stride=2),
                    )
                )
        
        # Feature extraction compatibility
        if features_only:
            self.out_indices = out_indices if out_indices is not None else range(5)
            self.out_indices = sorted(set([idx + 5 if idx < 0 else idx for idx in self.out_indices]))
            assert min(self.out_indices) >= 0 and max(self.out_indices) <= 4, f"Invalid out_indices: {self.out_indices}"
            self.channels = widths[-1]
        else:
            self.out_indices = None
            self.attnpool = DensityAwarePooling(widths[-1], output_dim)
            self.channels = output_dim

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = self.stem(x)
        feats = [x] if self.features_only and 0 in self.out_indices else []
        
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if self.features_only and (idx + 1) in self.out_indices:
                feats.append(x)
            if idx < 3:
                x = self.downsamples[idx](x)
        
        if self.features_only:
            return feats if len(feats) > 1 else feats[0]
        else:
            x = self.attnpool(x)
            return x



class ModifiedConvNextMultiScale(nn.Module):
    def __init__(self, base_width: int, layers: List[int], output_dim: int, input_resolution: int):
        super().__init__()
        self.convnext = ModifiedConvNext(base_width, layers, output_dim, input_resolution, features_only=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        features = self.convnext(x)  # Returns a list of feature maps

        # Ensure at least 3 feature maps exist
        if len(features) < 3:
            raise ValueError(f"Expected at least 3 feature maps, but got {len(features)}")

        shallow = features[0]  # Early feature map
        mid = features[len(features) // 2]  # Middle-stage feature map
        deep = features[-2]  # Deepest feature map
        print("Feature 1 Shape: ", shallow.shape)
        print("Feature 2 Shape: ", mid.shape)
        print("Feature 3 Shape: ", deep.shape)
        return shallow, mid, deep

