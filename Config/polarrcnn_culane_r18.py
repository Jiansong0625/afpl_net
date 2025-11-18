cfg_name = 'polarrcnn_culane_r18'
############### import package ######################
import math
import cv2

############### dataset choice ######################
dataset =  'culane'
data_root = r'E:\PolarRCNN-master\Culane'  # Use raw string to avoid escape sequence warning 

############### image parameter #########################
ori_img_h =  590
ori_img_w =  1640
cut_height =  270
img_h = 320
img_w = 800
center_h = 25
center_w = 386
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

# 自适应分支模块配置（带动态门控的多分支卷积）
use_adaptive_branch = True  # 是否启用自适应分支模块（替代标准卷积）
# 控制哪些阶段启用自适应分支：[layer1, layer2, layer3, layer4]
# True表示使用自适应分支模块，False表示使用标准3x3卷积
inceptionnext_stages = [False, False, True, True]  # layer3和layer4启用

# 自适应分支模块的超参数
adaptive_square_kernel = 3      # 方形卷积核大小（标准卷积分支）
adaptive_band_kernel = 11       # 带状卷积核大小（水平+垂直分支）
adaptive_branch_ratio = 0.125   # 每个卷积分支占用的通道比例（剩余通道为identity）
adaptive_use_gating = True      # 是否启用动态门控网络（通道级动态路由）
adaptive_gate_reduction = 16    # 门控网络的降维比例（越大越轻量，建议8-32）

#####neck#####
neck = 'fpn'
fpn_in_channel = [128, 256, 512]
neck_dim = 64
downsample_strides = [8, 16, 32]

#####rpn head#####
rpn_head = 'local_polar_head'
rpn_inchannel = neck_dim
polar_map_size = (4, 10)
num_training_priors = polar_map_size[0]*polar_map_size[1]
num_testing_priors = 20
angle_noise_p = 0.025
rho_noise_p = 0.25

#####roi head#####
roi_head = 'global_polar_head'
num_feat_samples = 36
fc_hidden_dim = 192
prior_feat_channels = 64
num_line_groups = 6
gnn_inter_dim = 128
iou_dim = 5
o2o_angle_thres = math.pi/6
o2o_rho_thres = 50

############## train parameter ###############################
batch_size = 16
epoch_num = 32
random_seed = 3404

######################optimizer parameter#################################
lr = 6e-4
warmup_iter = 1000

######################loss parameter######################################
rpn_loss = 'polarmap_loss'
roi_loss = 'tribranch_loss'

#####cost function#####
reg_cost_weight = 6
reg_cost_weight_o2o = 6
cls_cost_weight = 1
angle_prior_thres = math.pi/5
rho_prior_thres = 80
cost_iou_width = 30
ota_iou_width = 7.5

#####loss function #####
g_weight = 1
iou_loss_weight = 2
cls_loss_weight = 0.33
cls_loss_alpha = 0.47
cls_loss_alpha_o2o = 0.3
rank_loss_weight = 0.7
end_loss_weight = 0.03
aux_loss_weight = 0.2
polarmap_loss_weight = 5
loss_iou_width = 7.5

######################postprocess parameter######################################
nms_thres = 50
conf_thres = 0.48
conf_thres_o2o = conf_thres
conf_thres_nmsfree = 0.46
is_nmsfree = True
# is_nmsfree = False


# batch_size = 24
# epoch_num = 1
# warmup_iter = 1
# batch_size = 40
# epoch_num=8
# conf_thres = 0.66