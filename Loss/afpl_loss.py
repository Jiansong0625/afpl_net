"""
Loss functions for AFPL-Net

Implements three loss components:
1. Focal Loss for classification (lane/non-lane)
2. BCE Loss for centerness (point quality)
3. Smooth L1 Loss for polar regression (θ, r)
"""

import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (RetinaNet)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits [B, 1, H, W] or [B, H, W]
            target: Ground truth binary mask [B, H, W] or same as pred
        """
        # Align shapes: squeeze channel dim if present
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)  # [B,H,W]
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)

        pred = pred.reshape(-1)              # [N]
        target = target.reshape(-1).float()  # [N]

        # BCE with logits (per-element)
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # Probabilities
        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)

        # Focal factors
        focal_term = (1 - p_t).pow(self.gamma)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        loss = alpha_t * focal_term * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CenternessLoss(nn.Module):
    """
    BCE loss for centerness prediction, optionally masked by positives
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: logits [B,1,H,W] or [B,H,W]
            target: [B,H,W] in [0,1]
            mask: optional weight [B,H,W] for per-pixel loss scaling
        """
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)

        pred = pred.reshape(-1)
        target = target.reshape(-1).float()

        if mask is not None:
            if mask.dim() == 4 and mask.size(1) == 1:
                mask = mask.squeeze(1)
            mask = mask.reshape(-1)
            pos = mask > 0
            pred = pred[pos]
            target = target[pos]

        if pred.numel() == 0:
            return pred.new_tensor(0.0)

        loss = F.binary_cross_entropy_with_logits(pred, target, reduction=self.reduction)
        return loss


def smooth_l1_with_beta(input, target, beta=1.0, reduction='mean'):
    # Manual Smooth L1 to avoid version issues with beta param
    diff = torch.abs(input - target)
    loss = torch.where(diff < beta, 0.5 * (diff ** 2) / beta, diff - 0.5 * beta)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


class PeriodicL1Loss(nn.Module):
    """周期性 L1，用于角度差的稳健衡量"""
    def __init__(self, beta=1.0, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred, target):
        delta = pred - target
        sin_term = torch.sin(delta)
        cos_term = torch.cos(delta) - 1.0
        loss = torch.abs(sin_term) + torch.abs(cos_term)
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class PolarRegressionLoss(nn.Module):
    """
    Smooth L1 Loss for polar coordinate regression of (theta, r)
    Only computed on positive samples (mask)
    """
    def __init__(self, beta=1.0, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.periodic_l1 = PeriodicL1Loss(beta=beta, reduction=reduction)

    def forward(self, theta_pred, r_pred, theta_target, r_target, mask):
        """
        Args:
            theta_pred: [B,1,H,W] logits mapped to [-pi,pi] upstream
            r_pred: [B,1,H,W] >= 0
            theta_target: [B,H,W]
            r_target: [B,H,W]
            mask: [B,H,W] positive mask
        """
        theta_pred = theta_pred.squeeze(1).reshape(-1)
        r_pred = r_pred.squeeze(1).reshape(-1)
        theta_target = theta_target.reshape(-1)
        r_target = r_target.reshape(-1)
        if mask.dim() == 4 and mask.size(1) == 1:
            mask = mask.squeeze(1)
        mask = mask.reshape(-1)

        pos = mask > 0
        theta_pred = theta_pred[pos]
        r_pred = r_pred[pos]
        theta_target = theta_target[pos]
        r_target = r_target[pos]

        if theta_pred.numel() == 0:
            return theta_pred.new_tensor(0.0)
        theta_loss = self.periodic_l1(theta_pred, theta_target)
        r_loss = smooth_l1_with_beta(r_pred, r_target, beta=self.beta, reduction=self.reduction)
        return theta_loss + r_loss


class AFPLLoss(nn.Module):
    """
    Overall loss for AFPL-Net: Focal (cls) + BCE (centerness) + Smooth L1 (theta,r)
    """
    def __init__(self, cfg):
        super().__init__()
        self.cls_weight = getattr(cfg, 'cls_loss_weight', 1.0)
        self.centerness_weight = getattr(cfg, 'centerness_loss_weight', 1.0)
        self.regression_weight = getattr(cfg, 'regression_loss_weight', 1.0)
        self.use_adaptive_pole = getattr(cfg, 'use_adaptive_pole', False)
        self.pole_loss_weight = getattr(cfg, 'pole_loss_weight', 0.0)

        self.focal_loss = FocalLoss(
            alpha=getattr(cfg, 'cls_loss_alpha', 0.25),
            gamma=getattr(cfg, 'cls_loss_gamma', 2.0)
        )
        self.centerness_loss = CenternessLoss()
        self.polar_regression_loss = PolarRegressionLoss(
            beta=getattr(cfg, 'regression_beta', 1.0)
        )
        self.pole_loss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, pred_dict, target_dict):
        """
        Args:
            pred_dict: {'cls_pred','centerness_pred','theta_pred','r_pred'}
            target_dict: {'cls_gt','centerness_gt','theta_gt','r_gt'}
        Returns:
            total_loss, loss_dict
        """
        cls_pred = pred_dict['cls_pred']
        centerness_pred = pred_dict['centerness_pred']
        theta_pred = pred_dict['theta_pred']
        r_pred = pred_dict['r_pred']

        cls_gt = target_dict['cls_gt']
        centerness_gt = target_dict['centerness_gt']
        theta_gt = target_dict['theta_gt']
        r_gt = target_dict['r_gt']
        pole_gt = target_dict.get('pole_gt') if isinstance(target_dict, dict) else None

        # positive mask from cls_gt
        pos_mask = (cls_gt > 0.5).float()

        centerness_pos_mask = (centerness_gt > 0).float()

        loss_cls = self.focal_loss(cls_pred, cls_gt)
        loss_centerness = self.centerness_loss(
            centerness_pred,
            centerness_gt,
            mask=centerness_pos_mask
        )

        # Normalize r to stabilize early training
        with torch.no_grad():
            r_scale = r_gt.max().clamp(min=1.0)  # dynamic per-batch scale
        loss_reg = self.polar_regression_loss(theta_pred, r_pred / r_scale, theta_gt, r_gt / r_scale, pos_mask)

        total = self.cls_weight * loss_cls + \
                self.centerness_weight * loss_centerness + \
                self.regression_weight * loss_reg
        
        loss_pole = cls_pred.new_tensor(0.0)
        if self.use_adaptive_pole and pole_gt is not None and self.pole_loss_weight > 0:
            loss_pole = self.pole_loss(pred_dict['pole_xy'], pole_gt)
            total = total + self.pole_loss_weight * loss_pole

        loss_dict = {
            'loss': total,
            'loss_cls': loss_cls.detach(),
            'loss_centerness': loss_centerness.detach(),
            'loss_reg': loss_reg.detach(),
        }
        if self.use_adaptive_pole:
            loss_dict['loss_pole'] = loss_pole.detach()
        return total, loss_dict
