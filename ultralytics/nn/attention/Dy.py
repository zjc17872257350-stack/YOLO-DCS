#gaijin。。。。。。。。。。。。。。。。。
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

import torch
from torch import nn


class UltraLightDepthwiseSeparableConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, act=True, g=1):
        super().__init__()

        # 深度可分离卷积
        self.depthwise = nn.Conv2d(c1, c1, kernel_size=k, stride=s, 
                                   padding=k // 2, groups=c1, bias=False)
        self.bn_dw = nn.BatchNorm2d(c1)

        # 使用逐点卷积进行通道压缩（减少参数量）
        self.pointwise = nn.Conv2d(c1, c2, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(c2)

        # 激活函数（默认ReLU，轻量）
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        # 深度卷积
        x = self.act(self.bn_dw(self.depthwise(x)))

        # 点卷积，通道压缩
        x = self.bn_pw(self.pointwise(x))

        return x



class LightweightConvV2(nn.Module):
   
    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__()
        self.c_mid = c2 // 2

        # Depthwise convolution
        self.depthwise = nn.Conv2d(c1, c1, kernel_size=k, stride=s, 
                                   padding=k // 2, groups=c1, bias=False)
        self.bn_dw = nn.BatchNorm2d(c1)

        # Pointwise convolution (reduce channels)
        self.pointwise_reduce = nn.Conv2d(c1, self.c_mid, kernel_size=1, bias=False)
        self.bn_pw_reduce = nn.BatchNorm2d(self.c_mid)

        # Pointwise convolution (expand channels)
        self.pointwise_expand = nn.Conv2d(self.c_mid, c2, kernel_size=1, bias=False)
        self.bn_pw_expand = nn.BatchNorm2d(c2)

        # Activation function
        self.act = nn.Mish() if act else nn.Identity()

    def forward(self, x):
        # Depthwise convolution
        x = self.act(self.bn_dw(self.depthwise(x)))

        # Reduce channels
        x = self.act(self.bn_pw_reduce(self.pointwise_reduce(x)))

        # Expand channels
        x = self.bn_pw_expand(self.pointwise_expand(x))

        # Channel shuffle for interaction
        b, c, h, w = x.shape
        x = x.reshape(b, 2, c // 2, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, c, h, w)

        return x
        
        
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Corrected torch.max usage
        avg_out = torch.mean(x, dim=(2, 3), keepdim=True)
        max_out, _ = torch.max(x, dim=2, keepdim=True)  # Corrected dimension for max operation
        max_out, _ = torch.max(max_out, dim=3, keepdim=True)  # Apply max on the second dimension (width)
        out = avg_out + max_out
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Corrected max usage
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out) * x

class AdaptiveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super(AdaptiveConv, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, k, padding=k//2) for k in kernel_sizes])

    def forward(self, x):
        return sum([conv(x) for conv in self.convs])

class FeatureGuidance(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(FeatureGuidance, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=(2, 3), keepdim=True)
        max_out, _ = torch.max(x, dim=2, keepdim=True)  # Corrected max usage
        max_out, _ = torch.max(max_out, dim=3, keepdim=True)  # Apply max on the second dimension (width)
        out = avg_out + max_out
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return self.sigmoid(out) * x


class DynamicFeatureEnhancement(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)

        # Initial convolution layer to create 3 * c channels
        self.cv1 = Conv(c1, 3 * self.c, 1, 1)  
        self.cv2 = Conv(3 * self.c, c1, 1)

        # Adaptive convolution
        self.adaptive_conv = AdaptiveConv(self.c, self.c)

        # Feature guidance
        self.feature_guidance = FeatureGuidance(self.c)

        # Attention mechanisms
        self.channel_attn = ChannelAttention(self.c)
        self.spatial_attn = SpatialAttention()

        # Feedforward network
        self.ffn = nn.Sequential(
            UltraLightDepthwiseSeparableConv(self.c, self.c * 2, 1),
            UltraLightDepthwiseSeparableConv(self.c * 2, self.c, 1, act=False)
        )

    def forward(self, x):
        # Split the output of the initial convolution into 3 parts
        x_split = self.cv1(x)
        a, b, d = x_split.chunk(3, dim=1)  # split into 3 equal parts along channel dimension
        
        # Apply adaptive convolution
        b = self.adaptive_conv(b)
        
        # Feature guidance enhancement
        b = self.feature_guidance(b)
        
        # Apply channel and spatial attention
        b = self.channel_attn(b)
        d = self.spatial_attn(a)
        
        # Apply feedforward network
        b = b + self.ffn(b)
        
        # Concatenate the 3 parts along the channel dimension
        return self.cv2(torch.cat((a, b, d), dim=1))
