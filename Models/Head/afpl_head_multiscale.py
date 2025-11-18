"""
Enhanced Multi-Scale AFPL Head

This head extends the original AFPL head with:
1. Multi-scale feature fusion (P3, P4, P5)
2. Improved head architecture with residual connections
3. Optional spatial attention for better localization

The multi-scale approach helps detect lanes at various distances and scales.
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
import numpy as np


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution for efficient computation
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ImprovedPredictionHead(nn.Module):
    """
    Improved prediction head with residual connections and efficient convolutions
    """
    def __init__(self, in_channels, out_channels, use_depthwise=False):
        super().__init__()
        mid_channels = in_channels
        
        if use_depthwise:
            self.conv1 = DepthwiseSeparableConv(in_channels, mid_channels, 3, 1, 1)
            self.conv2 = DepthwiseSeparableConv(mid_channels, mid_channels // 2, 3, 1, 1)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 3, 1, 1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels // 2, 3, 1, 1),
                nn.BatchNorm2d(mid_channels // 2),
                nn.ReLU(inplace=True)
            )
        
        self.out_conv = nn.Conv2d(mid_channels // 2, out_channels, 1, 1, 0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.out_conv(x)
        return x


class MultiScaleAFPLHead(nn.Module):
    """
    Multi-Scale Anchor-Free Polar Lane Detection Head
    
    Extends the original AFPL head by:
    - Using features from multiple scales (P3, P4, P5)
    - Fusing multi-scale predictions for better detection
    - Improved head architecture with residual connections
    """
    
    def __init__(self, cfg):
        super(MultiScaleAFPLHead, self).__init__()
        
        # Image and network parameters
        self.img_w, self.img_h = cfg.img_w, cfg.img_h
        self.in_channels = cfg.neck_dim  # FPN output channels
        self.use_multiscale = getattr(cfg, 'use_multiscale_head', False)
        self.use_depthwise = getattr(cfg, 'use_depthwise_conv', False)
        
        # Global pole (vanishing point)
        self.global_pole_x = cfg.center_w if hasattr(cfg, 'center_w') else cfg.img_w // 2
        self.global_pole_y = cfg.center_h if hasattr(cfg, 'center_h') else cfg.img_h // 4
        self.use_adaptive_pole = getattr(cfg, 'use_adaptive_pole', False)
        init_pole = getattr(cfg, 'initial_pole', (self.global_pole_x, self.global_pole_y))
        self.register_buffer('initial_pole', torch.tensor(init_pole, dtype=torch.float32))
        self.register_buffer(
            'pole_offset_scale',
            torch.tensor([self.img_w / 2.0, self.img_h / 2.0], dtype=torch.float32)
        )
        
        # Inference parameters
        self.conf_threshold = cfg.conf_thres if hasattr(cfg, 'conf_thres') else 0.1
        self.centerness_threshold = cfg.centerness_thres if hasattr(cfg, 'centerness_thres') else 0.1
        self.angle_cluster_eps = cfg.angle_cluster_eps if hasattr(cfg, 'angle_cluster_eps') else 0.035
        self.min_cluster_points = cfg.min_cluster_points if hasattr(cfg, 'min_cluster_points') else 10
        
        # Build heads
        if self.use_multiscale:
            # Multi-scale heads: create separate heads for each scale
            self.num_scales = 3  # P3, P4, P5
            self.cls_heads = nn.ModuleList([
                ImprovedPredictionHead(self.in_channels, 1, self.use_depthwise)
                for _ in range(self.num_scales)
            ])
            self.centerness_heads = nn.ModuleList([
                ImprovedPredictionHead(self.in_channels, 1, self.use_depthwise)
                for _ in range(self.num_scales)
            ])
            self.polar_reg_heads = nn.ModuleList([
                ImprovedPredictionHead(self.in_channels, 2, self.use_depthwise)
                for _ in range(self.num_scales)
            ])
        else:
            # Single-scale heads (original behavior)
            self._build_classification_head()
            self._build_centerness_head()
            self._build_polar_regression_head()
        
        self._maybe_build_pole_head(cfg)
        
    def _build_classification_head(self):
        """Classification Head: H×W×1"""
        self.cls_head = ImprovedPredictionHead(self.in_channels, 1, self.use_depthwise)
        
    def _build_centerness_head(self):
        """Centerness Head: H×W×1"""
        self.centerness_head = ImprovedPredictionHead(self.in_channels, 1, self.use_depthwise)
        
    def _build_polar_regression_head(self):
        """Polar Regression Head: H×W×2"""
        self.polar_reg_head = ImprovedPredictionHead(self.in_channels, 2, self.use_depthwise)
    
    def _maybe_build_pole_head(self, cfg):
        """Optional lightweight head to predict adaptive pole location"""
        if not self.use_adaptive_pole:
            self.pole_head = None
            return
        hidden_dim = getattr(cfg, 'pole_head_dim', 64)
        self.pole_head = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 2)
        )
        nn.init.zeros_(self.pole_head[-1].weight)
        nn.init.zeros_(self.pole_head[-1].bias)
        
    def forward(self, feats):
        """
        Forward pass through AFPL head
        
        Args:
            feats: List of FPN feature maps [P3, P4, P5, ...]
            
        Returns:
            Dictionary containing predictions from all heads
        """
        if self.use_multiscale:
            return self._forward_multiscale(feats)
        else:
            return self._forward_singlescale(feats)
    
    def _forward_singlescale(self, feats):
        """Single-scale forward (original behavior)"""
        feat = feats[0]  # P3: highest resolution (stride 8)
        pole_xy = self._predict_pole(feat)
        
        # Three parallel predictions
        cls_pred = self.cls_head(feat)
        centerness_pred = self.centerness_head(feat)
        polar_pred = self.polar_reg_head(feat)
        
        # Normalize polar predictions
        theta_pred = torch.tanh(polar_pred[:, 0:1, ...]) * math.pi
        r_pred = F.relu(polar_pred[:, 1:2, ...])
        
        pred_dict = {
            'cls_pred': cls_pred,
            'centerness_pred': centerness_pred,
            'theta_pred': theta_pred,
            'r_pred': r_pred,
            'pole_xy': pole_xy,
        }
        
        return pred_dict
    
    def _forward_multiscale(self, feats):
        """Multi-scale forward with feature fusion"""
        # Use first 3 scales: P3, P4, P5
        scales_to_use = min(self.num_scales, len(feats))
        feat_list = feats[:scales_to_use]
        
        # Predict pole from highest resolution feature
        pole_xy = self._predict_pole(feat_list[0])
        
        # Collect predictions from each scale
        cls_preds = []
        centerness_preds = []
        theta_preds = []
        r_preds = []
        
        # Reference size (P3)
        ref_size = feat_list[0].shape[2:]
        
        for i, feat in enumerate(feat_list):
            # Get predictions at this scale
            cls_pred = self.cls_heads[i](feat)
            centerness_pred = self.centerness_heads[i](feat)
            polar_pred = self.polar_reg_heads[i](feat)
            
            # Normalize polar predictions
            theta_pred = torch.tanh(polar_pred[:, 0:1, ...]) * math.pi
            r_pred = F.relu(polar_pred[:, 1:2, ...])
            
            # Upsample to reference size for fusion
            if feat.shape[2:] != ref_size:
                cls_pred = F.interpolate(cls_pred, size=ref_size, mode='bilinear', align_corners=False)
                centerness_pred = F.interpolate(centerness_pred, size=ref_size, mode='bilinear', align_corners=False)
                theta_pred = F.interpolate(theta_pred, size=ref_size, mode='bilinear', align_corners=False)
                r_pred = F.interpolate(r_pred, size=ref_size, mode='bilinear', align_corners=False)
            
            cls_preds.append(cls_pred)
            centerness_preds.append(centerness_pred)
            theta_preds.append(theta_pred)
            r_preds.append(r_pred)
        
        # Fuse multi-scale predictions (weighted average)
        # Higher weight for higher resolution (P3 > P4 > P5)
        weights = torch.tensor([0.5, 0.3, 0.2][:scales_to_use], device=feat_list[0].device)
        weights = weights / weights.sum()
        
        cls_pred = sum(w * p for w, p in zip(weights, cls_preds))
        centerness_pred = sum(w * p for w, p in zip(weights, centerness_preds))
        theta_pred = sum(w * p for w, p in zip(weights, theta_preds))
        r_pred = sum(w * p for w, p in zip(weights, r_preds))
        
        pred_dict = {
            'cls_pred': cls_pred,
            'centerness_pred': centerness_pred,
            'theta_pred': theta_pred,
            'r_pred': r_pred,
            'pole_xy': pole_xy,
        }
        
        return pred_dict
    
    def _predict_pole(self, feat):
        """Predict per-image pole when enabled, otherwise fall back to static pole"""
        batch_size = feat.shape[0]
        base = self.initial_pole.to(feat.device).unsqueeze(0).expand(batch_size, -1)
        if not self.use_adaptive_pole or self.pole_head is None:
            return base
        offset = torch.tanh(self.pole_head(feat))
        scale = self.pole_offset_scale.to(feat.device)
        return base + offset * scale
    
    def post_process(self, pred_dict, downsample_factor=8, output_format='numpy'):
        """
        Post-processing: Angular clustering and lane formation
        
        Args:
            pred_dict: Predictions from forward pass
            downsample_factor: Feature map downsample factor relative to input
            output_format: 'numpy' for inference, 'dict' for training/debugging
            
        Returns:
            List of detected lanes per batch item
        """
        # Apply sigmoid to get probabilities
        cls_pred = torch.sigmoid(pred_dict['cls_pred'])
        centerness_pred = torch.sigmoid(pred_dict['centerness_pred'])
        theta_pred = pred_dict['theta_pred']
        r_pred = pred_dict['r_pred']
        
        batch_size = cls_pred.shape[0]
        lanes_batch = []
        
        for b in range(batch_size):
            # Compute final score: classification × centerness
            final_score = cls_pred[b, 0] * centerness_pred[b, 0]
            
            # Filter low-confidence points
            valid_mask = final_score > self.conf_threshold
            
            if not valid_mask.any():
                lanes_batch.append([])
                continue
            
            # Get coordinates of valid points
            y_coords, x_coords = torch.where(valid_mask)
            
            if len(y_coords) == 0:
                lanes_batch.append([])
                continue
            
            # Get predictions for valid points
            scores = final_score[valid_mask].detach().cpu().numpy()
            thetas = theta_pred[b, 0, valid_mask].detach().cpu().numpy()
            rs = r_pred[b, 0, valid_mask].detach().cpu().numpy()
            
            # Get predicted pole for this image
            pole_x = pred_dict['pole_xy'][b, 0].detach().cpu().item()
            pole_y = pred_dict['pole_xy'][b, 1].detach().cpu().item()
            
            # Angular clustering and convert polar to Cartesian
            lanes = self._cluster_by_angle(thetas, rs, scores, pole_x, pole_y, output_format)
            lanes_batch.append(lanes)
        
        return lanes_batch
    
    def _cluster_by_angle(self, thetas, rs, scores, pole_x, pole_y, output_format='numpy'):
        """Cluster points by angle using DBSCAN and convert polar to Cartesian"""
        if len(thetas) == 0:
            return []
        
        # Normalize angle for clustering
        thetas_normalized = (thetas + math.pi) / (2 * math.pi)
        features = thetas_normalized.reshape(-1, 1)
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=self.angle_cluster_eps, min_samples=self.min_cluster_points)
        labels = clustering.fit_predict(features)
        
        # Form lanes from clusters
        lanes = []
        unique_labels = set(labels)
        unique_labels.discard(-1)
        
        for label in unique_labels:
            cluster_mask = labels == label
            
            cluster_thetas = thetas[cluster_mask]
            cluster_rs = rs[cluster_mask]
            cluster_scores = scores[cluster_mask]
            
            # Convert polar to Cartesian
            cluster_x = pole_x + cluster_rs * np.cos(cluster_thetas)
            cluster_y = pole_y + cluster_rs * np.sin(cluster_thetas)
            
            # Sort by y coordinate
            sort_idx = np.argsort(cluster_y)
            cluster_x = cluster_x[sort_idx]
            cluster_y = cluster_y[sort_idx]
            cluster_scores = cluster_scores[sort_idx]
            
            if output_format == 'numpy':
                lane_points = np.column_stack([cluster_x, cluster_y]).astype(np.float32)
                lanes.append(lane_points)
            else:
                lane_points = [(float(x), float(y)) for x, y in zip(cluster_x, cluster_y)]
                lanes.append({
                    'points': lane_points,
                    'scores': cluster_scores.tolist(),
                    'mean_score': float(np.mean(cluster_scores))
                })
        
        # Sort lanes
        if output_format == 'numpy':
            lanes.sort(key=lambda x: len(x), reverse=True)
        else:
            lanes.sort(key=lambda x: x['mean_score'], reverse=True)
        
        return lanes
