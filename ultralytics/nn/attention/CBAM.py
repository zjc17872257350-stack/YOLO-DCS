import torch
from torch import nn


# ͨ��ע����ģ��
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # ����Ӧƽ���ػ�
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # ����Ӧ���ػ�

        # ������������ڴӳػ����������ѧϰע����Ȩ��
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # ��һ������㣬��ά
        self.relu1 = nn.ReLU()  # ReLU�����
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # �ڶ�������㣬��ά
        self.sigmoid = nn.Sigmoid()  # Sigmoid�����������յ�ע����Ȩ��

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # ��ƽ���ػ����������д���
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # �����ػ����������д���
        out = avg_out + max_out  # �����ֳػ���������Ȩ����Ϊ���
        return self.sigmoid(out)  # ʹ��sigmoid���������ע����Ȩ��


# �ռ�ע����ģ��
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'  # ���Ĵ�Сֻ����3��7
        padding = 3 if kernel_size == 7 else 1  # ���ݺ��Ĵ�С�������

        # ��������ڴ����ӵ�ƽ���ػ������ػ�����ͼ��ѧϰ�ռ�ע����Ȩ��
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # Sigmoid�����������յ�ע����Ȩ��

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # ����������ͼִ��ƽ���ػ�
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # ����������ͼִ�����ػ�
        x = torch.cat([avg_out, max_out], dim=1)  # �����ֳػ�������ͼ��������
        x = self.conv1(x)  # ͨ������㴦�����Ӻ������ͼ
        return self.sigmoid(x)  # ʹ��sigmoid���������ע����Ȩ��


# CBAMģ��
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)  # ͨ��ע����ʵ��
        self.sa = SpatialAttention(kernel_size)  # �ռ�ע����ʵ��

    def forward(self, x):
        out = x * self.ca(x)  # ʹ��ͨ��ע������Ȩ��������ͼ
        result = out * self.sa(out)  # ʹ�ÿռ�ע������һ����Ȩ����ͼ
        return result  # �������յ�����ͼ


# ʾ��ʹ��
if __name__ == '__main__':
    block = CBAM(64)  # ����һ��CBAMģ�飬����ͨ��Ϊ64
    input = torch.rand(4, 256, 64, 64)  # �������һ����������ͼ
    output = block(input)  # ͨ��CBAMģ�鴦����������ͼ
    print(input.size(), output.size())  # ��ӡ����������shape������֤
