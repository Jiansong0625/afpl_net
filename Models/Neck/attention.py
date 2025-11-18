"""
Attention Modules for Enhanced Feature Learning

Implements channel and spatial attention mechanisms to help the network
focus on lane-relevant features and suppress background noise.
"""

import torch
from torch import nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM)
    
    Adaptively recalibrates channel-wise feature responses by explicitly
    modeling interdependencies between channels.
    
    Reference: SENet (Squeeze-and-Excitation Networks)
    """
    
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Average pooling and max pooling
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        
        # Combine and apply sigmoid
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM)
    
    Generates a spatial attention map that highlights where to focus
    in the spatial dimension.
    
    Reference: CBAM (Convolutional Block Attention Module)
    """
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        
        return x * out


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    
    Sequentially applies channel and spatial attention for enhanced
    feature representation. Particularly useful for lane detection
    where both "what" (channel) and "where" (spatial) matter.
    """
    
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class CoordAttention(nn.Module):
    """
    Coordinate Attention Module
    
    Encodes channel relationships and long-range dependencies with precise
    positional information. Particularly effective for lane detection where
    spatial location is critical.
    
    Reference: Coordinate Attention for Efficient Mobile Network Design (CVPR 2021)
    """
    
    def __init__(self, in_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, in_channels // reduction)
        
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # Encode along height and width
        x_h = self.pool_h(x)  # [N, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [N, C, W, 1]
        
        # Concatenate and transform
        y = torch.cat([x_h, x_w], dim=2)  # [N, C, H+W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # Split back
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        # Attention weights
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        out = identity * a_h * a_w
        return out
