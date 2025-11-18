from .anchor_based_lane_detector import TwoStageAnchorBasedLaneDetector
from .afpl_net import AFPLNet

def build_model(cfg):
    # Check if using AFPL-Net (single-stage, anchor-free)
    if hasattr(cfg, 'cfg_name') and 'afplnet' in cfg.cfg_name.lower():
        model = AFPLNet(cfg=cfg)
    else:
        # Default: Two-stage anchor-based Polar R-CNN
        model = TwoStageAnchorBasedLaneDetector(cfg=cfg)
    return model