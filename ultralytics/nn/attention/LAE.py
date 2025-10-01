import torch
import torch.nn as nn
from einops import rearrange

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class LAE(nn.Module):
    # Light-weight Adaptive Extraction
    def __init__(self, c1, c2, group=16) -> None:
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)
        self.attention = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv(c1, c1, k=1)  # Self-attention on the same channel size
        )

        self.ds_conv = Conv(c1, c2, k=3, s=1, g=(c1 // group))  # Depthwise separable convolution

    def forward(self, x):
        # Compute attention
        att = self.attention(x)
        att = self.softmax(att.view(x.size(0), x.size(1), -1))  # Flatten spatial dimensions
        att = att.view_as(x)  # Restore to original shape

        # Apply depthwise separable convolution and combine with attention
        x = self.ds_conv(x)
        x = x * att  # Element-wise multiplication with attention map
        return x
