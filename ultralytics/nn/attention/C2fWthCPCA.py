import torch
import torch.nn as nn
import torch.nn.functional as F

class CPCAChannelAttention(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(CPCAChannelAttention, self).__init__()
        # 确保Conv2d参数正确：输入通道数、输出通道数、卷积核大小、步长等
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))  # 使用全局平均池化
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)

        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))  # 使用全局最大池化
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)

        x = x1 + x2  # 将两个池化的结果加起来
        x = x.view(-1, self.input_channels, 1, 1)  # 调整形状
        return x

class C2fWithCPCA(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=1, internal_neurons=16):
        super(C2fWithCPCA, self).__init__()

        # 基础的卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)  # 使用stride=1，padding=1
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)  # 使用stride=1，padding=1
        self.bn2 = nn.BatchNorm2d(out_channels)

        # CPCAChannelAttention模块
        self.cpca_attention = CPCAChannelAttention(out_channels, internal_neurons)

        # 用于整合前向传播的通道注意力
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # 第一层卷积
        out = F.relu(self.bn1(self.conv1(x)))

        # 第二层卷积
        out = F.relu(self.bn2(self.conv2(out)))

        # 获取CPCA通道注意力
        attention = self.cpca_attention(out)

        # 使用注意力加权输出
        out = out * attention

        # 通过1x1卷积调整输出
        out = self.bn3(self.conv3(out))

        return out

# 测试代码
if __name__ == "__main__":
    input_tensor = torch.randn(8, 64, 32, 32)  # 假设输入为8个样本，64个通道，32x32大小
    model = C2fWithCPCA(in_channels=64, out_channels=128, stride=1)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # 输出应该是(8, 128, 32, 32)
