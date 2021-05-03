import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def deconv4x4(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=stride, padding=1, bias=False)

class InvBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(InvBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.upsample layers upsample the input when stride != 1
        if stride == 1:
            self.conv1 = conv3x3(inplanes, planes, stride)
        else:
            self.conv1 = deconv4x4(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class InvBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, upsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(InvBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.upsample layers upsample the input when stride != 1
        self.conv1 = conv1x1(in_planes, in_planes)
        self.bn1 = norm_layer(in_planes)
        if stride == 1:
            self.conv2 = conv3x3(in_planes, in_planes, stride)
        else:
            self.conv2 = deconv4x4(in_planes, out_planes, stride)
        self.bn2 = norm_layer(out_planes)
        self.conv3 = conv1x1(out_planes, out_planes)
        self.bn3 = norm_layer(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
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

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class InvResNet(nn.Module):

    def __init__(self, block, layers, output_channels=3, input_dims=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(InvResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(512)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer5 = self._make_layer(block, 32, 16, layers[0], stride=2)
        self.layer4 = self._make_layer(block, 64, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer2 = self._make_layer(block, 256, 128, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer1 = self._make_layer(block, 512, 256, layers[4], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.fc = nn.Linear(input_dims, 512)
        self.lin = nn.ConvTranspose2d(16, output_channels, kernel_size=4, stride=2, padding=1, bias=False)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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

    def _make_layer(self, block, in_planes, out_planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1:
            upsample = nn.Sequential(
                deconv4x4(in_planes, out_planes, stride),
                norm_layer(out_planes),
            )

        layers = []
        for _ in range(1, blocks):
            layers.append(block(in_planes, in_planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        layers.append(block(in_planes, out_planes, stride, upsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
                            
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.fc(x)
        x = x.view(x.size(0), -1, 1,1)
        x = self.conv1(x)
        x = self.bn1(x)
        self.x0 = self.relu(x)

        self.x1 = self.layer1(self.x0)
        self.x2 = self.layer2(self.x1)
        self.x3 = self.layer3(self.x2)
        self.x4 = self.layer4(self.x3)
        self.x5 = self.layer5(self.x4)

        x = self.lin(self.x5)
        x = torch.sigmoid(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _invresnet(block, layers, **kwargs):
    model = InvResNet(block, layers, **kwargs)
    return model


def invresnet18(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _invresnet(InvBasicBlock, [2, 2, 2, 2, 2], **kwargs)


def invresnet34(**kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _invresnet(InvBasicBlock, [3, 4, 6, 4, 3], **kwargs)


def invresnet50(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _invresnet(InvBottleneck, [3, 4, 6, 4, 3], **kwargs)


def invresnet101(**kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _invresnet(InvBottleneck, [3, 4, 23, 4, 3], **kwargs)


def invresnet152(**kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _invresnet(InvBottleneck, [3, 8, 36, 8, 3], **kwargs)


