import torch
from torch import nn
from .roi_loss.build import build_roi_loss
from .rpn_loss.build import build_rpn_loss

class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Check if using AFPL-Net (single-stage)
        if hasattr(cfg, 'cfg_name') and 'afplnet' in cfg.cfg_name.lower():
            # Choose between original and enhanced AFPL loss
            use_enhanced_loss = getattr(cfg, 'use_adaptive_focal_loss', False) or \
                              getattr(cfg, 'use_enhanced_regression_loss', False)
            
            if use_enhanced_loss:
                from .afpl_loss_enhanced import EnhancedAFPLLoss
                self.afpl_loss = EnhancedAFPLLoss(cfg)
            else:
                from .afpl_loss import AFPLLoss
                self.afpl_loss = AFPLLoss(cfg)
            
            self.is_afpl = True
        else:
            # Two-stage Polar R-CNN losses
            self.roi_loss = build_roi_loss(cfg)
            self.rpn_loss = build_rpn_loss(cfg)
            self.is_afpl = False
    
    def forward(self, pred_dict, target_dict):
        if self.is_afpl:
            # AFPL-Net: single loss function
            loss, loss_msg = self.afpl_loss(pred_dict, target_dict)
            return loss, loss_msg
        else:
            # Polar R-CNN: ROI + RPN losses
            loss_roi, loss_roi_msg = self.roi_loss(pred_dict, target_dict)
            if self.rpn_loss is not None:
                loss_rpn, loss_rpn_msg = self.rpn_loss(pred_dict, target_dict)
                loss_roi = loss_roi + loss_rpn
                loss_roi_msg.update(loss_rpn_msg)
            
            loss = loss_roi
            loss_msg = loss_roi_msg
            loss_msg['loss'] = loss
            return loss, loss_msg

        

        