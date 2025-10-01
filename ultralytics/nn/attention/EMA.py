import torch  # ���� PyTorch ��
from torch import nn  # �� PyTorch �е���������ģ��
 
class EMA(nn.Module):  # ����һ���̳��� nn.Module �� EMA ��
    def __init__(self, channels, c2=None, factor=32):  # ���캯������ʼ������
        super(EMA, self).__init__()  # ���ø���Ĺ��캯��
        self.groups = factor  # �����������Ϊ factor��Ĭ��ֵΪ 32
        assert channels // self.groups > 0  # ȷ��ͨ�������Ա���������
        self.softmax = nn.Softmax(-1)  # ���� softmax �㣬�������һ��ά��
        self.agp = nn.AdaptiveAvgPool2d((1, 1))  # ��������Ӧƽ���ػ��㣬�����СΪ 1x1
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # ��������Ӧƽ���ػ��㣬ֻ�ڿ���ϳػ�
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # ��������Ӧƽ���ػ��㣬ֻ�ڸ߶��ϳػ�
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)  # �������һ����
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)  # ���� 1x1 �����
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)  # ���� 3x3 �����
 
    def forward(self, x):  # ����ǰ�򴫲�����
        b, c, h, w = x.size()  # ��ȡ���������Ĵ�С�����Ρ�ͨ�����߶ȺͿ��
        group_x = x.reshape(b * self.groups, -1, h, w)  # ����������������״Ϊ (b * ����, c // ����, �߶�, ���)
        x_h = self.pool_h(group_x)  # �ڸ߶��Ͻ��гػ�
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # �ڿ���Ͻ��гػ�������ά��
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # ���ػ����ƴ�Ӳ�ͨ�� 1x1 �����
        x_h, x_w = torch.split(hw, [h, w], dim=2)  # �����������߶ȺͿ�ȷָ�
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())  # �������һ��������ϸ߶ȺͿ�ȵļ�����
        x2 = self.conv3x3(group_x)  # ͨ�� 3x3 �����
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # �� x1 ���гػ�����״�任����Ӧ�� softmax
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # �� x2 ������״Ϊ (b * ����, c // ����, �߶� * ���)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # �� x2 ���гػ�����״�任����Ӧ�� softmax
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # �� x1 ������״Ϊ (b * ����, c // ����, �߶� * ���)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)  # ����Ȩ��
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)  # Ӧ��Ȩ�ز�����״�ָ�Ϊԭʼ��С