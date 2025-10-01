import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardConvNet(nn.Module):
    def __init__(self, 
                 inp_channels=31, 
                 out_channels=31, 
                 dim=48,
                 num_blocks=3,  # 改为整数
                 bias=False):
        super(StandardConvNet, self).__init__()

        if isinstance(num_blocks, int):
            num_blocks = [num_blocks] * 4  # 转为列表
        

        # 输入嵌入
        self.patch_embed = nn.Conv2d(inp_channels, dim, kernel_size=3, stride=1, padding=1, bias=bias)

        # 第一层编码
        self.encoder_level1 = self._make_layer(dim, dim, num_blocks[0], bias)

        # 第二层编码
        self.down1_2 = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, bias=bias)
        self.encoder_level2 = self._make_layer(dim * 2, dim * 2, num_blocks[1], bias)

        # 第三层编码
        self.down2_3 = nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=2, padding=1, bias=bias)
        self.encoder_level3 = self._make_layer(dim * 4, dim * 4, num_blocks[2], bias)

        # 解码部分
        self.up3_2 = nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = self._make_layer(dim * 2, dim * 2, num_blocks[1], bias)

        self.up2_1 = nn.ConvTranspose2d(dim * 2, dim, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias)
        self.reduce_chan_level1 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.decoder_level1 = self._make_layer(dim, dim, num_blocks[0], bias)

        # 输出层
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def _make_layer(self, in_channels, out_channels, num_blocks, bias):
        
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias) for _ in range(num_blocks)]
        return nn.Sequential(*layers)

    def forward(self, inp_img):
        # 编码阶段
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        # 解码阶段
        out_dec_level3 = out_enc_level3

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        # 输出层
        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1



