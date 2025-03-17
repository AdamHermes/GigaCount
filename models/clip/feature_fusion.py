import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureFusion(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        """
        Multi-scale feature fusion module to aggregate different levels of feature maps.
        
        Parameters:
        - in_channels_list (list): List of input channel dimensions for different feature scales.
        - out_channels (int): Output feature dimension after fusion.
        """
        super().__init__()
        self.conv1x1 = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False) for in_ch in in_channels_list
        ])
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels_list), out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        )
        

    def forward(self, features):


        # Determine target size (shallow feature size)
        target_size = features[0].shape[2:]  # (H, W) of the largest feature map

        # Resize all feature maps to match target size
        resized_features = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False) for f in features]



        # Concatenate along channel dimension
        fused_output = torch.cat(resized_features, dim=1)


        return fused_output

