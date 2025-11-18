"""
AFPL-Net: Anchor-Free Polar Lane Network

A single-stage, anchor-free lane detection network that combines:
- Global pole concept from Polar R-CNN
- Per-pixel prediction from PolarMask
- Centerness for quality estimation
- Angular clustering for NMS-free post-processing
"""

import torch
from torch import nn
from .Backbone.build import build_backbone
from .Neck.build import build_neck
from .Head.afpl_head import AFPLHead
from .Head.afpl_head_multiscale import MultiScaleAFPLHead


class AFPLNet(nn.Module):
    """
    Single-Stage Anchor-Free Polar Lane Network
    
    Architecture:
        Backbone (ResNet/DLA) → FPN → AFPL Head (3 parallel branches)
        
    Key differences from Polar R-CNN:
        - Single-stage (no RPN + ROI, just direct prediction)
        - Anchor-free (no need for 20 predefined anchors)
        - Per-pixel prediction (every feature map location predicts)
        - Angular clustering (NMS-free post-processing)
    """
    
    def __init__(self, cfg=None):
        super().__init__()
        
        self.cfg = cfg
        
        # Backbone: ResNet, DLA, etc.
        self.backbone = build_backbone(cfg)
        
        # Neck: FPN for multi-scale features
        self.neck = build_neck(cfg)
        
        # AFPL Head: Choose between original and multi-scale version
        use_multiscale = getattr(cfg, 'use_multiscale_head', False)
        if use_multiscale:
            self.afpl_head = MultiScaleAFPLHead(cfg)
        else:
            self.afpl_head = AFPLHead(cfg)
        
    def forward(self, sample_batch):
        """
        Forward pass
        
        Args:
            sample_batch: Dictionary with 'img' key during training,
                         or tensor directly during inference
                         
        Returns:
            Dictionary with predictions from AFPL head
        """
        # Extract input image - handle both dict and tensor inputs
        if isinstance(sample_batch, dict):
            x = sample_batch['img']
        else:
            # Direct tensor input (typically during inference)
            x = sample_batch
            
        # Backbone: extract multi-scale features
        backbone_feats = self.backbone(x)[1:]  # Skip first low-level features
        
        # Neck: FPN for feature fusion
        fpn_feats = self.neck(backbone_feats)
        
        # AFPL Head: per-pixel predictions
        pred_dict = self.afpl_head(fpn_feats)
        
        # During inference, add post-processing
        if not self.training:
            # Post-process to get lane curves
            downsample_factor = self.cfg.downsample_strides[0] if hasattr(self.cfg, 'downsample_strides') else 8
            lanes = self.afpl_head.post_process(pred_dict, downsample_factor)
            pred_dict['lane_list'] = lanes
            
        return pred_dict
    
    def get_lanes(self, img_tensor):
        """
        Convenience method for inference
        
        Args:
            img_tensor: Input image tensor [B, C, H, W]
            
        Returns:
            List of detected lanes (one list per batch item)
        """
        self.eval()
        with torch.no_grad():
            pred_dict = self.forward(img_tensor)
        return pred_dict['lane_list']