import torch
from torch import nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url


model_urls = {
    'resnet18':
    'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34':
    'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50':
    'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101':
    'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152':
    'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d':
    'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d':
    'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2':
    'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2':
    'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class AdaptiveBranchModule(nn.Module):
    """
    自适应分支模块：带动态路由的多分支深度卷积
    使用门控网络根据输入特征动态分配各分支的权重
    """
    def __init__(self, in_channels, out_channels=None, square_kernel_size=3, 
                 band_kernel_size=11, branch_ratio=0.125, stride=1, 
                 use_gating=True, gate_reduction=16):
        super().__init__()
        # 目标通道数
        target_c = out_channels if out_channels is not None else in_channels
        self.need_proj = (in_channels != target_c) or (stride != 1)
        self.use_gating = use_gating
        
        # 1x1投影层（用于通道变换和下采样）
        self.proj = nn.Conv2d(in_channels, target_c, kernel_size=1, stride=stride, bias=False) if self.need_proj else nn.Identity()
        
        # 计算每个分支的通道数
        gc = max(1, int(target_c * branch_ratio))
        branch_stride = 1 if self.need_proj else stride
        
        # 三个深度卷积分支
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, stride=branch_stride, 
                                   padding=square_kernel_size//2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), stride=branch_stride,
                                  padding=(0, band_kernel_size//2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), stride=branch_stride,
                                  padding=(band_kernel_size//2, 0), groups=gc)
        
        self.split_indexes = (target_c - 3 * gc, gc, gc, gc)
        
        # 门控网络：用于动态路由
        if self.use_gating:
            # 全局平均池化 + 小型MLP生成分支权重
            gate_channels = max(target_c // gate_reduction, 4)
            self.gate_net = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # GAP: [B, C, H, W] -> [B, C, 1, 1]
                nn.Conv2d(target_c, gate_channels, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(gate_channels, 4, 1, bias=False),  # 4个分支（id + 3个卷积分支）
                nn.Sigmoid()  # 输出每个分支的权重 [B, 4, 1, 1]
            )
    
    def forward(self, x):
        # 先做投影（变通道/下采样）
        x = self.proj(x)
        
        # 分割成4部分
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        
        # 各分支卷积
        out_id = x_id
        out_hw = self.dwconv_hw(x_hw)
        out_w = self.dwconv_w(x_w)
        out_h = self.dwconv_h(x_h)
        
        # 如果启用门控，动态加权各分支
        if self.use_gating:
            # 计算门控权重 [B, 4, 1, 1]
            gate_weights = self.gate_net(x)
            
            # 分别提取每个分支的权重
            w_id = gate_weights[:, 0:1, :, :]
            w_hw = gate_weights[:, 1:2, :, :]
            w_w = gate_weights[:, 2:3, :, :]
            w_h = gate_weights[:, 3:4, :, :]
            
            # 加权各分支输出
            out_id = out_id * w_id
            out_hw = out_hw * w_hw
            out_w = out_w * w_w
            out_h = out_h * w_h
        
        # 合并所有分支
        return torch.cat([out_id, out_hw, out_w, out_h], dim=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None,
                 use_adaptive_branch=False,
                 adaptive_branch_config=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')

        # 使用自适应分支模块或标准卷积
        if use_adaptive_branch and adaptive_branch_config is not None:
            self.conv1 = AdaptiveBranchModule(
                inplanes, planes,
                square_kernel_size=adaptive_branch_config.get('square_kernel_size', 3),
                band_kernel_size=adaptive_branch_config.get('band_kernel_size', 11),
                branch_ratio=adaptive_branch_config.get('branch_ratio', 0.125),
                stride=stride,
                use_gating=adaptive_branch_config.get('use_gating', True),
                gate_reduction=adaptive_branch_config.get('gate_reduction', 16)
            )
            self.conv2 = AdaptiveBranchModule(
                planes, planes,
                square_kernel_size=adaptive_branch_config.get('square_kernel_size', 3),
                band_kernel_size=adaptive_branch_config.get('band_kernel_size', 11),
                branch_ratio=adaptive_branch_config.get('branch_ratio', 0.125),
                stride=1,
                use_gating=adaptive_branch_config.get('use_gating', True),
                gate_reduction=adaptive_branch_config.get('gate_reduction', 16)
            )
        else:
            self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
            self.conv2 = conv3x3(planes, planes, dilation=dilation)

        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None,
                 in_channels=None,
                 adaptive_branch_stages=None,
                 adaptive_branch_config=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        if adaptive_branch_stages is None:
            adaptive_branch_stages = [False, False, False, False]
        assert len(adaptive_branch_stages) == 4, "adaptive_branch_stages should be a list of 4 boolean values."
        self.adaptive_branch_stages = adaptive_branch_stages
        self.adaptive_branch_config = adaptive_branch_config

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                 replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3,
                               self.inplanes,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_channels = in_channels
        self.layer1 = self._make_layer(block, in_channels[0], layers[0],
                                       use_adaptive_branch=self.adaptive_branch_stages[0])
        self.layer2 = self._make_layer(block,
                                       in_channels[1],
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       use_adaptive_branch=self.adaptive_branch_stages[1])
        self.layer3 = self._make_layer(block,
                                       in_channels[2],
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       use_adaptive_branch=self.adaptive_branch_stages[2])
        if in_channels[3] > 0:
            self.layer4 = self._make_layer(
                block,
                in_channels[3],
                layers[3],
                stride=2,
                dilate=replace_stride_with_dilation[2],
                use_adaptive_branch=self.adaptive_branch_stages[3])
        self.expansion = block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, use_adaptive_branch=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer, 
                  use_adaptive_branch=use_adaptive_branch,
                  adaptive_branch_config=self.adaptive_branch_config))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      norm_layer=norm_layer,
                      use_adaptive_branch=use_adaptive_branch,
                      adaptive_branch_config=self.adaptive_branch_config))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out_layers = []
        for name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if not hasattr(self, name):
                continue
            layer = getattr(self, name)
            x = layer(x)
            out_layers.append(x)

        return out_layers


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        print('pretrained model: ', model_urls[arch])
        state_dict = load_state_dict_from_url(model_urls[arch])
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained,
                   progress, **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained,
                   progress, **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained,
                   progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained,
                   progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained,
                   progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained,
                   progress, **kwargs)


class ResNetWrapper(nn.Module):
    def __init__(self,
                 resnet='resnet18',
                 pretrained=True,
                 replace_stride_with_dilation=[False, False, False],
                 out_conv=False,
                 fea_stride=8,
                 out_channel=128,
                 in_channels=[64, 128, 256, 512],
                 cfg=None):
        super(ResNetWrapper, self).__init__()
        self.cfg = cfg
        self.in_channels = in_channels

        # 获取自适应分支配置
        adaptive_branch_config = None
        adaptive_branch_stages = [False, False, False, False]
        
        if hasattr(cfg, 'use_adaptive_branch') and cfg.use_adaptive_branch:
            # 使用配置中的 inceptionnext_stages 作为 adaptive_branch_stages
            adaptive_branch_stages = getattr(cfg, 'inceptionnext_stages', [False, False, False, False])
            adaptive_branch_config = {
                'square_kernel_size': getattr(cfg, 'adaptive_square_kernel', 3),
                'band_kernel_size': getattr(cfg, 'adaptive_band_kernel', 11),
                'branch_ratio': getattr(cfg, 'adaptive_branch_ratio', 0.125),
                'use_gating': getattr(cfg, 'adaptive_use_gating', True),
                'gate_reduction': getattr(cfg, 'adaptive_gate_reduction', 16),
            }

        self.model = eval(resnet)(
            pretrained=pretrained,
            replace_stride_with_dilation=replace_stride_with_dilation,
            in_channels=self.in_channels,
            adaptive_branch_stages=adaptive_branch_stages,
            adaptive_branch_config=adaptive_branch_config
        )
        self.out = None
        if out_conv:
            out_channel = 512
            for chan in reversed(self.in_channels):
                if chan < 0: continue
                out_channel = chan
                break
            self.out = conv1x1(out_channel * self.model.expansion,
                               cfg.featuremap_out_channel)

    def forward(self, x):
        x = self.model(x)
        if self.out:
            x[-1] = self.out(x[-1])
        return x
