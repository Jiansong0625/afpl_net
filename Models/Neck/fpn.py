from torch import nn
import torch.nn.functional as F
from .attention import CBAM


class FPN(nn.Module):
    def __init__(self, cfg):
        super(FPN, self).__init__()
        assert isinstance(cfg.fpn_in_channel, list)
        self.in_channels = cfg.fpn_in_channel
        self.out_channel = cfg.neck_dim
        self.num_level = len(self.in_channels)
        self.use_attention = getattr(cfg, 'fpn_use_attention', False)
        
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.attention_modules = nn.ModuleList() if self.use_attention else None

        for i in range(self.num_level):
            l_conv = nn.Conv2d(self.in_channels[i], self.out_channel, 1, 1, 0)
            fpn_conv = nn.Conv2d(self.out_channel, self.out_channel, 3, 1, 1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            
            # Add attention module after each FPN conv for better feature refinement
            if self.use_attention:
                self.attention_modules.append(CBAM(self.out_channel, reduction=16))

    def forward(self, inputs):
        assert len(inputs) == self.num_level

        # Lateral connections
        laterals = []
        for i in range(self.num_level):
           laterals.append(self.lateral_convs[i](inputs[i])) 

        # Top-down pathway with fusion
        for i in range(self.num_level - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode='nearest')
        
        # Generate output features with optional attention
        outs = []
        for i in range(self.num_level):
            out = self.fpn_convs[i](laterals[i])
            
            # Apply attention if enabled for enhanced feature learning
            if self.use_attention:
                out = self.attention_modules[i](out)
            
            outs.append(out)

        return outs