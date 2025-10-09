import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ODConv2d, ChannelAttention, SpatialAttention
class ccsm(nn.Module):
    def __init__(self, channel, channel2, num_filters):
        super(ccsm, self).__init__()
        self.ch_att_s = ChannelAttention(channel)
        self.sa_s = SpatialAttention(7)
        self.conv1 = nn.Sequential(
            ODConv2d(channel, channel, kernel_size=1, stride=1, padding=0), 
            nn.ReLU(),
            nn.BatchNorm2d(num_features = channel))
        self.conv2 = nn.Sequential(
            ODConv2d(channel, channel2, kernel_size=1, stride=1, padding=0), 
            nn.ReLU(),
            nn.BatchNorm2d(num_features = channel2))
        
        self.conv3 = nn.Sequential(
            ODConv2d(channel2, channel2, kernel_size=1, stride=1, padding=0), 
            nn.ReLU(),
            nn.BatchNorm2d(num_features = channel2))
        self.conv4 = nn.Sequential(
            ODConv2d(channel2, num_filters, kernel_size=1, stride=1, padding=0), 
            nn.ReLU(),
            nn.BatchNorm2d(num_features = num_filters))
           
    def forward(self, x):
        x = self.ch_att_s(x)*x
        pool1 = x
        x = self.conv1(x)
        x = x + pool1
        x = self.conv2(x)
        pool2 = x
        x = self.conv3(x)
        x = x + pool2
        x = self.conv4(x)
        
        x = self.sa_s(x)*x

        
        return x
    

class FeatureFusion(nn.Module):
    def __init__(self, num_filters1, num_filters2, num_filters3):
        super(FeatureFusion, self).__init__()
        self.upsample_1 = nn.ConvTranspose2d(in_channels=num_filters2, out_channels=num_filters2, kernel_size=4, padding=1, stride=2)
        self.upsample_2 = nn.ConvTranspose2d(in_channels=num_filters3, out_channels=num_filters3, kernel_size=4, padding=0, stride=4)
        self.final = nn.Sequential(
            nn.Conv2d(num_filters1+num_filters2+num_filters3, 1, kernel_size=1, padding=0),
            nn.ReLU(),
        )
        
    def forward(self, x1, x2, x3):
        x2 = self.upsample_1(x2)
        x3 = self.upsample_2(x3)

        x = torch.cat([x1, x2, x3], dim=1)
        
        return x
