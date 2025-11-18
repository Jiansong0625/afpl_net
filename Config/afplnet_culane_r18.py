"""
Configuration file for AFPL-Net on CULane dataset

AFPL-Net: Anchor-Free Polar Lane Network
- Single-stage, anchor-free architecture
- Per-pixel polar coordinate prediction
- Global pole (vanishing point) based approach
- Angular clustering for NMS-free post-processing
"""

cfg_name = 'afplnet_culane_r18'

############### import package ######################
import math
import cv2

############### dataset choice ######################
dataset = 'culane'
data_root = r'E:\PolarRCNN-master\Culane'  # Use raw string to avoid escape sequence warning

############### image parameter #########################
ori_img_h = 590
ori_img_w = 1640
cut_height = 270
img_h = 320
img_w = 800

# Global pole (vanishing point) - typically at top-center of image
center_h = 25  # y-coordinate of global pole
center_w = 386  # x-coordinate of global pole (roughly center)
initial_pole = (center_w, center_h)  # default pole before augmentation

# Adaptive pole options
use_adaptive_pole = True
pole_head_dim = 64
pole_loss_weight = 0.1

max_lanes = 4

############## data augment ###############################
train_augments = [
    dict(name='Resize', parameters=dict(height=img_h, width=img_w, interpolation=cv2.INTER_CUBIC, p=1.0)),
    dict(name='HorizontalFlip', parameters=dict(p=0.5)),
    dict(name='RandomBrightnessContrast', parameters=dict(brightness_limit=(-0.15, 0.15), contrast_limit=(-0, 0), p=0.6)),
    dict(name='HueSaturationValue', parameters=dict(hue_shift_limit=(-10, 10), sat_shift_limit=(-10, 10), val_shift_limit=(-0, 0), p=0.7)),
    dict(name='OneOf', transforms=[dict(name='MotionBlur', parameters=dict(blur_limit=(3, 5)), p=1.0),
                                   dict(name='MedianBlur', parameters=dict(blur_limit=(3, 5)), p=1.0)], p=0.2),
    dict(name='Affine', parameters=dict(translate_percent=dict(x=(-0.1, 0.1), y=(-0.1, 0.1)), rotate=(-9, 9), scale=(0.8, 1.2), interpolation=cv2.INTER_CUBIC, p=0.7)),
    dict(name='Resize', parameters=dict(height=img_h, width=img_w, interpolation=cv2.INTER_CUBIC, p=1.0)),
]

############### lane parameter #########################
num_offsets = 72
offset_stride = 4.507

######################network parameter#################################
#####backbone#####
backbone = 'resnet18'
pretrained = True

#####neck#####
neck = 'fpn'
fpn_in_channel = [128, 256, 512]
neck_dim = 64
downsample_strides = [8, 16, 32]

#####AFPL head parameters#####
# No RPN head - single stage!
# No ROI head - direct prediction!

############## AFPL-Net specific parameters #############
# Inference thresholds
conf_thres = 0.1  # Classification confidence threshold
centerness_thres = 0.1  # Centerness threshold (for filtering low-quality points)

# Angular clustering parameters (for NMS-free post-processing)
angle_cluster_eps = 0.035  # DBSCAN epsilon for angle clustering (~2 degrees in normalized space)
min_cluster_points = 10  # Minimum points to form a lane cluster

############## train parameter ###############################
batch_size = 16
epoch_num = 32
random_seed = 3404

######################optimizer parameter#################################
lr = 6e-4
warmup_iter = 800

######################loss parameter######################################
# AFPL loss weights
cls_loss_weight = 1.0  # Weight for Focal Loss (classification)
cls_loss_alpha = 0.25  # Alpha parameter for Focal Loss
cls_loss_gamma = 2.0  # Gamma parameter for Focal Loss

centerness_loss_weight = 1.5  # Weight for centerness BCE loss
regression_loss_weight = 2.0  # Weight for polar regression (Smooth L1)
regression_beta = 1.0  # Beta parameter for Smooth L1 loss

######################postprocess parameter######################################
# No NMS needed! Angular clustering handles instance separation
is_nmsfree = True  # AFPL-Net is always NMS-free

# For compatibility with evaluation code
nms_thres = 50  # Not used, but kept for compatibility
conf_thres_nmsfree = conf_thres

centerness_debug_dir = r"E:\PolarRCNN-master\Debug\centerness"  # Debug output directory
enable_centerness_debug = False  # Set to True to save centerness visualizations
