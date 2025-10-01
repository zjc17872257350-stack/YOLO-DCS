import torch
import torch.nn as nn
import torch.nn.functional as F

class CPCAChannelAttention(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(CPCAChannelAttention, self).__init__()
        # ȷ��Conv2d������ȷ������ͨ���������ͨ����������˴�С��������
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))  # ʹ��ȫ��ƽ���ػ�
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)

        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))  # ʹ��ȫ�����ػ�
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)

        x = x1 + x2  # �������ػ��Ľ��������
        x = x.view(-1, self.input_channels, 1, 1)  # ������״
        return x

class C2fWithCPCA(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=1, internal_neurons=16):
        super(C2fWithCPCA, self).__init__()

        # �����ľ��
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)  # ʹ��stride=1��padding=1
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)  # ʹ��stride=1��padding=1
        self.bn2 = nn.BatchNorm2d(out_channels)

        # CPCAChannelAttentionģ��
        self.cpca_attention = CPCAChannelAttention(out_channels, internal_neurons)

        # ��������ǰ�򴫲���ͨ��ע����
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # ��һ����
        out = F.relu(self.bn1(self.conv1(x)))

        # �ڶ�����
        out = F.relu(self.bn2(self.conv2(out)))

        # ��ȡCPCAͨ��ע����
        attention = self.cpca_attention(out)

        # ʹ��ע������Ȩ���
        out = out * attention

        # ͨ��1x1����������
        out = self.bn3(self.conv3(out))

        return out

# ���Դ���
if __name__ == "__main__":
    input_tensor = torch.randn(8, 64, 32, 32)  # ��������Ϊ8��������64��ͨ����32x32��С
    model = C2fWithCPCA(in_channels=64, out_channels=128, stride=1)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # ���Ӧ����(8, 128, 32, 32)
