"""
Enhanced Loss Functions for AFPL-Net

Improvements over the original loss:
1. IoU-based loss for better localization
2. Dynamic loss weight balancing
3. Improved hard sample mining
4. Better numerical stability
"""

import torch
from torch import nn
import torch.nn.functional as F
from .afpl_loss import FocalLoss, CenternessLoss, PeriodicL1Loss, smooth_l1_with_beta


class IoULoss(nn.Module):
    """
    IoU-based loss for regression
    
    Helps improve localization accuracy by directly optimizing IoU.
    Particularly useful for lane detection where spatial accuracy is critical.
    """
    
    def __init__(self, reduction='mean', loss_type='iou'):
        super().__init__()
        self.reduction = reduction
        self.loss_type = loss_type  # 'iou', 'giou', 'diou'
        
    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: [N, 4] (x1, y1, x2, y2)
            target_boxes: [N, 4] (x1, y1, x2, y2)
        """
        # Calculate intersection
        lt = torch.max(pred_boxes[:, :2], target_boxes[:, :2])
        rb = torch.min(pred_boxes[:, 2:], target_boxes[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        
        # Calculate union
        area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        area_target = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union = area_pred + area_target - inter
        
        # IoU
        iou = inter / (union + 1e-7)
        
        if self.loss_type == 'iou':
            loss = 1 - iou
        elif self.loss_type == 'giou':
            # GIoU: generalized IoU
            # Calculate enclosing box
            enclose_lt = torch.min(pred_boxes[:, :2], target_boxes[:, :2])
            enclose_rb = torch.max(pred_boxes[:, 2:], target_boxes[:, 2:])
            enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
            enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]
            giou = iou - (enclose_area - union) / (enclose_area + 1e-7)
            loss = 1 - giou
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AdaptiveFocalLoss(nn.Module):
    """
    Adaptive Focal Loss with dynamic alpha and gamma
    
    Automatically adjusts focus on hard samples based on training progress.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer('avg_pos_ratio', torch.tensor(0.1))
        
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits [B, 1, H, W] or [B, H, W]
            target: Ground truth binary mask [B, H, W]
        """
        # Align shapes
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)

        pred = pred.reshape(-1)
        target = target.reshape(-1).float()
        
        # Update positive ratio (exponential moving average)
        with torch.no_grad():
            pos_ratio = target.mean()
            self.avg_pos_ratio = 0.99 * self.avg_pos_ratio + 0.01 * pos_ratio
        
        # Adaptive alpha based on class imbalance
        adaptive_alpha = self.alpha * (1 - self.avg_pos_ratio) / self.avg_pos_ratio
        adaptive_alpha = torch.clamp(adaptive_alpha, 0.1, 0.9)

        # BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # Probabilities
        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)

        # Focal factors
        focal_term = (1 - p_t).pow(self.gamma)
        alpha_t = adaptive_alpha * target + (1 - adaptive_alpha) * (1 - target)

        loss = alpha_t * focal_term * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class EnhancedPolarRegressionLoss(nn.Module):
    """
    Enhanced Polar Regression Loss with:
    1. Separate weighting for theta and r
    2. Distance-aware loss weighting (far lanes get higher weight)
    3. Better handling of periodic angle differences
    """
    
    def __init__(self, beta=1.0, theta_weight=1.0, r_weight=1.0, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.theta_weight = theta_weight
        self.r_weight = r_weight
        self.reduction = reduction
        self.periodic_l1 = PeriodicL1Loss(beta=beta, reduction=reduction)

    def forward(self, theta_pred, r_pred, theta_target, r_target, mask):
        """
        Args:
            theta_pred: [B,1,H,W]
            r_pred: [B,1,H,W]
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
        
        # Distance-aware weighting: far lanes (larger r) get slightly higher weight
        # This helps the model focus on harder cases
        with torch.no_grad():
            r_weights = torch.sqrt(r_target / (r_target.mean() + 1e-6))
            r_weights = torch.clamp(r_weights, 0.5, 2.0)
        
        theta_loss = self.periodic_l1(theta_pred, theta_target)
        r_loss = smooth_l1_with_beta(r_pred, r_target, beta=self.beta, reduction='none')
        r_loss = (r_loss * r_weights).mean() if self.reduction == 'mean' else (r_loss * r_weights).sum()
        
        total_loss = self.theta_weight * theta_loss + self.r_weight * r_loss
        return total_loss


class DynamicLossBalancer:
    """
    Dynamically balances multiple loss components during training
    
    Uses uncertainty-based weighting to automatically balance losses.
    Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses"
    """
    
    def __init__(self, num_losses):
        self.num_losses = num_losses
        # Log variance parameters (learnable)
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
    
    def forward(self, losses):
        """
        Args:
            losses: List of loss values
        Returns:
            Weighted sum of losses
        """
        weighted_losses = []
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            weighted_losses.append(weighted_loss)
        return sum(weighted_losses), weighted_losses


class EnhancedAFPLLoss(nn.Module):
    """
    Enhanced overall loss for AFPL-Net with improved components
    """
    def __init__(self, cfg):
        super().__init__()
        self.cls_weight = getattr(cfg, 'cls_loss_weight', 1.0)
        self.centerness_weight = getattr(cfg, 'centerness_loss_weight', 1.0)
        self.regression_weight = getattr(cfg, 'regression_loss_weight', 1.0)
        self.use_adaptive_pole = getattr(cfg, 'use_adaptive_pole', False)
        self.pole_loss_weight = getattr(cfg, 'pole_loss_weight', 0.0)
        self.use_adaptive_focal = getattr(cfg, 'use_adaptive_focal_loss', False)
        self.use_enhanced_regression = getattr(cfg, 'use_enhanced_regression_loss', False)

        # Loss components
        if self.use_adaptive_focal:
            self.focal_loss = AdaptiveFocalLoss(
                alpha=getattr(cfg, 'cls_loss_alpha', 0.25),
                gamma=getattr(cfg, 'cls_loss_gamma', 2.0)
            )
        else:
            self.focal_loss = FocalLoss(
                alpha=getattr(cfg, 'cls_loss_alpha', 0.25),
                gamma=getattr(cfg, 'cls_loss_gamma', 2.0)
            )
        
        self.centerness_loss = CenternessLoss()
        
        if self.use_enhanced_regression:
            self.polar_regression_loss = EnhancedPolarRegressionLoss(
                beta=getattr(cfg, 'regression_beta', 1.0),
                theta_weight=getattr(cfg, 'theta_loss_weight', 1.0),
                r_weight=getattr(cfg, 'r_loss_weight', 1.0)
            )
        else:
            from .afpl_loss import PolarRegressionLoss
            self.polar_regression_loss = PolarRegressionLoss(
                beta=getattr(cfg, 'regression_beta', 1.0)
            )
        
        self.pole_loss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, pred_dict, target_dict):
        """
        Args:
            pred_dict: {'cls_pred','centerness_pred','theta_pred','r_pred','pole_xy'}
            target_dict: {'cls_gt','centerness_gt','theta_gt','r_gt','pole_gt'}
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

        # Positive mask
        pos_mask = (cls_gt > 0.5).float()
        centerness_pos_mask = (centerness_gt > 0).float()

        # Compute individual losses
        loss_cls = self.focal_loss(cls_pred, cls_gt)
        loss_centerness = self.centerness_loss(
            centerness_pred,
            centerness_gt,
            mask=centerness_pos_mask
        )

        # Normalize r for stability
        with torch.no_grad():
            r_scale = r_gt.max().clamp(min=1.0)
        loss_reg = self.polar_regression_loss(
            theta_pred, r_pred / r_scale, 
            theta_gt, r_gt / r_scale, 
            pos_mask
        )

        # Combine losses
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
