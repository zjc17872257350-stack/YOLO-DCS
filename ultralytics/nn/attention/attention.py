###################### ContextAggregation  ####     START   by  AI&CV  ###############################
 
from mmcv.cnn import ConvModule
from mmengine.model import caffe2_xavier_init, constant_init
 
from ultralytics.nn.modules.conv import Conv
import torch
import torch.nn as nn

 
class ContextAggregation(nn.Module):
    """
    Context Aggregation Block.
    Args:
        in_channels (int): Number of input channels.
        reduction (int, optional): Channel reduction ratio. Default: 1.
        conv_cfg (dict or None, optional): Config dict for the convolution
            layer. Default: None.
    """
 
    def __init__(self, in_channels, reduction=1):
        super(ContextAggregation, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = max(in_channels // reduction, 1)
 
        conv_params = dict(kernel_size=1, act_cfg=None)
 
        self.a = ConvModule(in_channels, 1, **conv_params)
        self.k = ConvModule(in_channels, 1, **conv_params)
        self.v = ConvModule(in_channels, self.inter_channels, **conv_params)
        self.m = ConvModule(self.inter_channels, in_channels, **conv_params)
 
        self.init_weights()
 
    def init_weights(self):
        for m in (self.a, self.k, self.v):
            caffe2_xavier_init(m.conv)
        constant_init(self.m.conv, 0)
 
    def forward(self, x):
        # n, c = x.size(0)
        n = x.size(0)
        c = self.inter_channels
        # n, nH, nW, c = x.shape
 
        # a: [N, 1, H, W]
        a = self.a(x).sigmoid()
 
        # k: [N, 1, HW, 1]
        k = self.k(x).view(n, 1, -1, 1).softmax(2)
 
        # v: [N, 1, C, HW]
        v = self.v(x).view(n, 1, c, -1)
 
        # y: [N, C, 1, 1]
        y = torch.matmul(v, k).view(n, c, 1, 1)
        y = self.m(y) * a
 
        return x + y
 
 
class PSContextAggregation(nn.Module):
 
    def __init__(self, c1, c2, e=0.1):
        super().__init__()
        assert (c1 == c2)
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
 
        self.attn = ContextAggregation(self.c)
        self.ffn = nn.Sequential(
            Conv(self.c, self.c * 2, 1),
            Conv(self.c * 2, self.c, 1, act=False)
        )
 
    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))
###################### ContextAggregation  ####     END   by  AI&CV  ###############################